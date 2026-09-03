"""Where the threshold comes from, measured rather than asserted.

Everything this project claims rests on one number: the cosine similarity above
which two faces are called the same person. 0.38 for ArcFace is a defensible
figure taken from the literature, but "defensible figure taken from the
literature" is not evidence, and a reader has no way to tell it apart from a
number that was tuned until the demo worked.

So it gets measured, on this machine, against real data:

* **Impostor pairs** come free from the identity index. Every pair of distinct
  Wikidata humans in it is a pair of different people, and 3,583 identities
  make 6.4 million such pairs. The false-positive rate at any threshold is a
  count over that whole set - no sampling, no extrapolation.

* **Genuine pairs** need two photographs of one person, which the index does
  not have: it keeps one portrait each. They are harvested here from the fact
  that different language Wikipedias illustrate the same person with different
  photographs - a real second capture, different year, angle and lighting,
  still labelled by Wikidata rather than by this code.

Two honesty notes that the report repeats, because they cut in opposite
directions:

* Some cross-language portraits are crops or retouches of one file rather than
  a separate photograph. Those are genuine pairs that are easier than they
  look, so the true-positive rate here is an upper bound. Byte-identical files
  are dropped, which catches the easiest case and not the rest.

* The subject of a lead image is taken to be its largest face, which is wrong
  for the occasional group photograph. That mislabels a genuine pair as a
  failure, pushing the true-positive rate down. It is left in rather than
  hand-corrected, because a hand-corrected set is not a measurement.

The genuine set is restricted to people born after ``BORN_AFTER``, and that
filter is not cosmetic. Without it the hardest "same person" pairs were every
pair of portraits of Alexander the Great: a Roman bust against a Pompeian
mosaic against a coin. Those are not two photographs of a face, and scoring a
face recogniser against them measured whether two sculptors agreed rather than
whether the model works. Their presence cost 14 points of true-positive rate.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .concurrency import prefetch
from .config import MODELS_DIR, Config
from .evidence import sha256_hex
from .face import decode_image, largest_face
from .identify import Identity, IdentityIndex
from .search.http import fetch_image, make_session
from .search.matcher import PREFETCH_FACTOR, download_workers

CALIBRATION_PATH = MODELS_DIR / "calibration.json"

# Wikipedias big enough to have their own picture editors, so their lead image
# is often a different photograph rather than a copy of the English one.
PORTRAIT_LANGS = ("en", "es", "fr", "de", "ru", "ja", "pt", "it", "ar", "fa",
                  "id", "tr", "pl", "hi", "ta")

HARVEST_WORKERS = 8
BATCH = 40

# A note on `--limit`, because the obvious instinct is to turn it up.
#
# Each sampled identity yields roughly nine portraits across fifteen
# Wikipedias, so the download count is about `limit * 9` individual files from
# Wikimedia. At 400 that is a few thousand and takes a bit under an hour on
# CPU; at 1500 it is ~12,900 and the encoder stops being the constraint
# entirely - even on a GPU it projected past two and a half hours, all of it
# waiting on someone else's servers.
#
# 400 is where this settled, and not only for the runtime: the genuine sample
# it produces (2,069 pairs from 323 people) already moved the headline figures
# by a quarter of a point against a sample a sixth of the size, so a larger one
# buys precision nobody needs at the cost of a sustained load on a free
# service. Raise it if you have a reason; it is not free to anyone but you.

# Above this, two "different" index identities are almost certainly the same
# human twice - one duplicated Wikidata entity, or one painting illustrating
# two historical figures. Reported separately rather than deleted, because
# which of the two it is matters to whoever reads the number.
ARTEFACT = 0.99

# Genuine pairs must be photographs of a face. Before roughly this year the
# lead image is a painting, a bust or a coin, and two artists' impressions of
# one head are not a second capture of it.
BORN_AFTER = 1900


@dataclass
class Distribution:
    pairs: int
    mean: float
    sd: float
    minimum: float
    maximum: float
    quantiles: dict[str, float]

    @classmethod
    def of(cls, v: np.ndarray, qs: Iterable[float]) -> Distribution:
        return cls(
            pairs=int(v.size),
            mean=float(v.mean()),
            sd=float(v.std()),
            minimum=float(v.min()),
            maximum=float(v.max()),
            quantiles={f"p{q:g}": float(np.quantile(v, q / 100)) for q in qs},
        )


@dataclass
class Calibration:
    backend: str
    model: str
    threshold: float
    index_identities: int
    sampled_requested: int
    sampled_photographic: int
    sampled_identities: int
    born_after: int | None
    portraits_encoded: int
    genuine: Distribution
    impostor: Distribution
    tpr: float
    fpr: float
    # None when every impostor pair turned out to be an artefact, so there is
    # nothing left to compute a rate over. Reporting nan in a table of measured
    # rates would be worse than reporting nothing.
    fpr_excluding_artefacts: float | None
    artefact_pairs: int
    eer: float
    eer_threshold: float
    curve: list[dict[str, float]] = field(default_factory=list)
    thresholds_for_fpr: dict[str, float] = field(default_factory=dict)
    artefact_examples: list[dict[str, Any]] = field(default_factory=list)
    hardest_genuine: list[dict[str, Any]] = field(default_factory=list)
    # The identity index is a different question from the match threshold, and
    # a harder one. Defaulted so a calibration written before this existed
    # still loads rather than failing to parse.
    # A hash of the index the impostor side was measured over. Only the count
    # used to tie the two together, so rebuilding the index with a different
    # 3,583 faces left every impostor statistic silently describing a set that
    # no longer existed. Defaulted so a calibration written before this loads.
    index_sha256: str = ""
    identify_threshold: float | None = None
    false_name_rate: float | None = None
    false_name_rate_excluding_artefacts: float | None = None
    wrongly_named: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Calibration:
        return cls(**{**d,
                      "genuine": Distribution(**d["genuine"]),
                      "impostor": Distribution(**d["impostor"])})

    # Both resolve the default at call time rather than binding it as a
    # parameter default, which would freeze the module constant at import and
    # make the path unredirectable - the same reason the chain client reads
    # STATE_PATH inside its methods.
    def save(self, path: Path | None = None) -> None:
        path = path or CALIBRATION_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1))

    def rates_at(self, threshold: float) -> tuple[float, float] | None:
        """(tpr, fpr) at ``threshold``, from the nearest measured curve point.

        Nearest rather than interpolated: these are counts over two finite
        populations, and inventing a value between two of them would be
        presenting arithmetic as measurement. Returns None outside the range
        that was actually measured, rather than extrapolating.
        """
        if not self.curve:
            return None
        step = 0.01
        nearest = min(self.curve, key=lambda c: abs(c["threshold"] - threshold))
        if abs(nearest["threshold"] - threshold) > step:
            return None
        return nearest["tpr"], nearest["fpr"]

    @classmethod
    def load(cls, path: Path | None = None) -> Calibration:
        path = path or CALIBRATION_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"no calibration at {path} - measure one with: sigil calibrate"
            )
        return cls.from_dict(json.loads(path.read_text()))


# -------------------------------------------------------------------- harvest


def _chunks(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _get_json(session, url: str, params: dict) -> dict:
    """A lost batch costs coverage, not the run - so it is skipped, not raised."""
    try:
        r = session.get(url, params=params, timeout=40)
        if r.status_code != 200:
            return {}
        return r.json()
    except Exception:  # noqa: BLE001 - a non-JSON error page lands here too
        return {}


def _in_parallel(items, work):
    with ThreadPoolExecutor(max_workers=HARVEST_WORKERS) as pool:
        for _item, result in prefetch(pool, items, work, HARVEST_WORKERS * 2):
            yield result


def _birth_year(ent: dict) -> int | None:
    """Wikidata P569, as a signed year. None when absent or unparseable."""
    for claim in (ent.get("claims") or {}).get("P569", []):
        time = ((claim.get("mainsnak") or {}).get("datavalue") or {}) \
            .get("value", {}).get("time")
        # "+1926-04-21T00:00:00Z", and "-0356-07-20T00:00:00Z" for BC.
        if isinstance(time, str) and len(time) > 5:
            try:
                return int(time[:5])
            except ValueError:
                continue
    return None


def sitelinks(
    session, qids: list[str], born_after: int | None = BORN_AFTER
) -> dict[str, dict[str, str]]:
    """qid -> {wiki: article title}, for every language it has an article in.

    Identities with no recorded birth year, or one before ``born_after``, are
    dropped here rather than downstream: fetching a dozen busts of Alexander
    the Great only to discard them is a minute of network for nothing.
    """
    out: dict[str, dict[str, str]] = {}

    def fetch(batch: list[str]) -> dict:
        return _get_json(session, "https://www.wikidata.org/w/api.php", {
            "action": "wbgetentities", "format": "json",
            "ids": "|".join(batch), "props": "sitelinks|claims",
        })

    for payload in _in_parallel(list(_chunks(qids, BATCH)), fetch):
        for qid, ent in (payload.get("entities") or {}).items():
            if born_after is not None:
                year = _birth_year(ent)
                if year is None or year < born_after:
                    continue
            out[qid] = {k: v["title"] for k, v in (ent.get("sitelinks") or {}).items()}
    return out


def lead_images(
    session,
    links: dict[str, dict[str, str]],
    langs: Iterable[str] = PORTRAIT_LANGS,
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, set[str]]:
    """qid -> the set of distinct lead-image URLs across those Wikipedias."""
    say = on_progress or (lambda _: None)
    found: dict[str, set[str]] = {q: set() for q in links}

    for lang in langs:
        wiki = f"{lang}wiki"
        by_title = {links[q][wiki]: q for q in links if wiki in links[q]}
        if not by_title:
            continue

        def fetch(batch: list[str], lang=lang) -> dict:
            return _get_json(session, f"https://{lang}.wikipedia.org/w/api.php", {
                "action": "query", "format": "json", "prop": "pageimages",
                "piprop": "original", "titles": "|".join(batch), "redirects": 1,
            })

        before = sum(len(v) for v in found.values())
        for payload in _in_parallel(list(_chunks(list(by_title), BATCH)), fetch):
            for pg in ((payload.get("query") or {}).get("pages") or {}).values():
                src = (pg.get("original") or {}).get("source")
                qid = by_title.get(pg.get("title", ""))
                if src and qid:
                    found[qid].add(src)
        gained = sum(len(v) for v in found.values()) - before
        say(f"  {lang}: {len(by_title)} articles -> {gained} new portraits")
    return found


def encode_portraits(
    encoder,
    urls_by_qid: dict[str, set[str]],
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, list[np.ndarray]]:
    """Download and encode every portrait, keeping the largest face in each.

    Byte-identical files are dropped per identity: the same photograph served
    to two Wikipedias is not a second capture, and scoring it as a genuine pair
    would measure the encoder against itself.
    """
    say = on_progress or (lambda _: None)
    session = make_session()
    jobs = [(q, u) for q, urls in urls_by_qid.items() for u in sorted(urls)]
    out: dict[str, list[np.ndarray]] = {}
    seen: dict[str, set[str]] = {}

    def grab(job: tuple[str, str]) -> bytes | None:
        return fetch_image(session, job[1], 30.0)

    # Portrait downloads feeding one encoder, same shape as the search path,
    # so the same rule for how many at once. See sigil/search/matcher.py.
    workers = download_workers(encoder, Config())
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, ((qid, _url), blob) in enumerate(
            prefetch(pool, jobs, grab, workers * PREFETCH_FACTOR), 1
        ):
            if done % 100 == 0:
                kept = sum(len(v) for v in out.values())
                say(f"  {done}/{len(jobs)} portraits · {kept} usable faces")
            if not blob:
                continue
            digest = sha256_hex(blob)
            if digest in seen.setdefault(qid, set()):
                continue
            seen[qid].add(digest)
            img = decode_image(blob)
            if img is None:
                continue
            face = largest_face(encoder.detect_and_encode(img))
            if face is None:
                continue
            out.setdefault(qid, []).append(face.embedding.astype(np.float32))
    return out


# -------------------------------------------------------------------- measure


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1.0, n)


def genuine_similarities(
    by_qid: dict[str, list[np.ndarray]],
) -> tuple[np.ndarray, list[tuple[str, int, int]]]:
    """Every within-identity pair, with the (qid, i, j) it came from."""
    sims: list[float] = []
    keys: list[tuple[str, int, int]] = []
    for qid in sorted(by_qid):
        vecs = by_qid[qid]
        if len(vecs) < 2:
            continue
        m = _unit(np.stack(vecs))
        s = m @ m.T
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                sims.append(float(s[i, j]))
                keys.append((qid, i, j))
    return np.asarray(sims, dtype=np.float64), keys


def index_digest(index: IdentityIndex) -> str:
    """Content hash of the vectors a calibration was measured over.

    The count alone cannot distinguish one index of 3,583 faces from another,
    and the impostor rates describe a specific set of faces rather than a
    number of them.
    """
    return sha256_hex(np.ascontiguousarray(
        index.vectors.astype(np.float32)).tobytes())


def impostor_similarities(index: IdentityIndex) -> tuple[np.ndarray, tuple]:
    """Every cross-identity pair in the index: different QIDs, different people."""
    m = _unit(index.vectors.astype(np.float32))
    s = m @ m.T
    iu = np.triu_indices(len(m), k=1)
    return s[iu].astype(np.float64), iu


def false_name_rate(
    index: IdentityIndex, threshold: float, artefact: float = ARTEFACT
) -> tuple[float, float, list[dict[str, Any]]]:
    """How often the index would put a *wrong name* to a face, per query.

    This is not the pair-level false-positive rate, and the difference is the
    whole point. `sigil identify` compares one probe against every face in the
    index at once, so it gets thousands of chances to be wrong per question
    asked. A per-pair rate of 1e-5 across 3,583 candidates is roughly a 3.5%
    chance of a wrong name on any single query - two numbers that sound alike
    and differ by three orders of magnitude.

    So it is measured the way the tool is used: every indexed face is queried
    against every *other* one, and a query counts as wrong if anything else
    clears the threshold. Returns (rate, rate ignoring artefacts, examples).

    Pairs at or above ``artefact`` are one human indexed twice - a duplicated
    Wikidata entity, or one painting illustrating two people. Naming that face
    with the other entry's label is a correct answer to a mislabelled index,
    so it is reported both ways rather than silently counted as a failure.
    """
    m = _unit(index.vectors.astype(np.float32))
    s = m @ m.T
    np.fill_diagonal(s, -2.0)  # a face is not its own impostor

    best = s.argmax(axis=1)
    top = s[np.arange(len(s)), best]
    wrong = top >= threshold
    genuine_dupe = top >= artefact

    names = [i.name for i in index.identities]
    order = np.argsort(top)[::-1]
    examples = [
        {"queried": names[int(i)], "named": names[int(best[i])],
         "similarity": round(float(top[i]), 4),
         "duplicate_entry": bool(genuine_dupe[i])}
        for i in order[:8] if wrong[i]
    ]
    return (
        float(wrong.mean()),
        float((wrong & ~genuine_dupe).mean()),
        examples,
    )


def _rates(genuine: np.ndarray, impostor: np.ndarray, t: float) -> tuple[float, float]:
    return float((genuine >= t).mean()), float((impostor >= t).mean())


def _threshold_at_fpr(impostor: np.ndarray, target: float) -> float:
    """The similarity a pair must clear for the false-positive rate to be ``target``."""
    return round(float(np.quantile(impostor, 1.0 - target)), 4)


def measure(
    encoder,
    index: IdentityIndex,
    by_qid: dict[str, list[np.ndarray]],
    threshold: float,
    requested: int = 0,
    born_after: int | None = BORN_AFTER,
    identify_threshold: float | None = None,
) -> Calibration:
    """Turn the two similarity populations into the numbers worth reporting."""
    genuine, gkeys = genuine_similarities(by_qid)
    if genuine.size == 0:
        raise RuntimeError(
            "no identity ended up with two usable portraits - nothing to measure"
        )
    impostor, iu = impostor_similarities(index)
    # An index with fewer than two faces has no cross-identity pair, so there
    # is no impostor population at all. Reachable: a build interrupted early
    # leaves a partial index, and every statistic below would otherwise be a
    # numpy warning followed by "zero-size array to reduction operation".
    if impostor.size == 0:
        raise RuntimeError(
            f"the identity index holds {len(index)} face(s), so there are no "
            "cross-identity pairs to measure a false-accept rate over. Build a "
            "larger index with: sigil index build"
        )

    artefacts = impostor >= ARTEFACT
    clean = impostor[~artefacts]
    tpr, fpr = _rates(genuine, impostor, threshold)

    curve = []
    for t in np.arange(0.05, 0.96, 0.01):
        a, b = _rates(genuine, impostor, float(t))
        curve.append({"threshold": round(float(t), 2), "tpr": a, "fpr": b})

    # Equal error rate: where the miss rate and the false-alarm rate cross.
    # It is the one summary number that does not depend on a chosen threshold.
    at = int(np.argmin([abs((1 - c["tpr"]) - c["fpr"]) for c in curve]))
    eer = (1 - curve[at]["tpr"] + curve[at]["fpr"]) / 2

    if identify_threshold is None:
        from .pipeline import IDENTITY_THRESHOLD

        identify_threshold = IDENTITY_THRESHOLD.get(encoder.name, 0.45)
    name_rate, name_rate_clean, wrongly = false_name_rate(index, identify_threshold)

    names = [i.name for i in index.identities]
    top = np.argsort(impostor)[::-1][:8]
    ident_by_qid = {i.qid: i for i in index.identities}
    hard = np.argsort(genuine)[:8]

    return Calibration(
        backend=encoder.name,
        model=encoder.model,
        threshold=threshold,
        index_identities=len(index),
        index_sha256=index_digest(index),
        sampled_requested=requested,
        sampled_photographic=len(by_qid),
        born_after=born_after,
        sampled_identities=sum(1 for v in by_qid.values() if len(v) >= 2),
        portraits_encoded=sum(len(v) for v in by_qid.values()),
        genuine=Distribution.of(genuine, (1, 5, 25, 50, 75)),
        impostor=Distribution.of(impostor, (50, 99, 99.9, 99.99)),
        tpr=tpr,
        fpr=fpr,
        fpr_excluding_artefacts=(
            float((clean >= threshold).mean()) if clean.size else None
        ),
        artefact_pairs=int(artefacts.sum()),
        eer=float(eer),
        eer_threshold=curve[at]["threshold"],
        curve=curve,
        thresholds_for_fpr={
            f"1e-{e}": _threshold_at_fpr(impostor, 10.0 ** -e) for e in (4, 5, 6)
        },
        artefact_examples=[
            {"a": names[int(iu[0][k])], "b": names[int(iu[1][k])],
             "similarity": round(float(impostor[int(k)]), 4)}
            for k in top
        ],
        hardest_genuine=[
            {"name": _name(ident_by_qid.get(gkeys[int(k)][0])),
             "similarity": round(float(genuine[int(k)]), 4)}
            for k in hard
        ],
        identify_threshold=identify_threshold,
        false_name_rate=name_rate,
        false_name_rate_excluding_artefacts=name_rate_clean,
        wrongly_named=wrongly,
    )


def _name(ident: Identity | None) -> str:
    return ident.name if ident else "?"


def calibrate(
    encoder,
    threshold: float,
    limit: int = 200,
    langs: Iterable[str] = PORTRAIT_LANGS,
    born_after: int | None = BORN_AFTER,
    on_progress: Callable[[str], None] | None = None,
) -> Calibration:
    """Harvest genuine pairs, pair them against the index, and measure."""
    say = on_progress or (lambda _: None)
    langs = tuple(langs)
    index = IdentityIndex.load(encoder)
    session = make_session()

    # A deterministic prefix, not a random sample: the index is in harvest
    # order, so the head is the most-viewed people, and taking a prefix keeps
    # two runs on the same index comparable.
    sample = [i.qid for i in index.identities[:limit]]
    say(f"resolving {len(sample)} identities across {len(langs)} Wikipedias")
    links = sitelinks(session, sample, born_after=born_after)
    if born_after is not None:
        say(f"  {len(links)} of {len(sample)} were born after {born_after}; "
            "the rest are painted or sculpted, not photographed")
    urls = lead_images(session, links, langs, on_progress=say)

    say(f"encoding {sum(len(v) for v in urls.values())} portraits")
    by_qid = encode_portraits(encoder, urls, on_progress=say)
    return measure(encoder, index, by_qid, threshold,
                   requested=len(sample), born_after=born_after)
