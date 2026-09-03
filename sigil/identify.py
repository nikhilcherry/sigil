"""Face -> name, from a locally built index of labelled public portraits.

Why this exists: Bluesky has no face index, so the social search can only be
seeded by text. That leaves the pipeline unable to answer the question the task
actually poses - "who is this?" - because you had to know the answer to ask.

This closes that loop. It builds a local index of faces with known names,
harvested from Wikipedia's most-viewed articles (which is where public figures
concentrate) joined to Wikidata for the "is a human" check and to Wikimedia
Commons for the portrait. Matching a probe against that index yields candidate
names, which then seed the live social search.

The index deliberately covers public figures only. It is built from an
encyclopaedia, so being in it is a consequence of public notability rather than
of having been scraped, and the whole thing is reproducible from source by
anyone who runs the builder.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from .concurrency import prefetch
from .config import MODELS_DIR, Config
from .face import decode_image, largest_face
from .search.http import fetch_image, make_session
from .search.matcher import PREFETCH_FACTOR, download_workers

PAGEVIEWS = "https://wikimedia.org/api/rest_v1/metrics/pageviews/top/{wiki}/all-access/{ym}/all-days"
DEFAULT_LANGS = ("en", "hi", "ta", "te", "ml", "bn", "mr", "kn", "es", "fr")
INDEX_VECTORS = MODELS_DIR / "identity-index.npz"
INDEX_META = MODELS_DIR / "identity-index.json"

# The harvest is entirely network-bound - a few hundred small API calls to
# Wikimedia, none of which depend on each other. Run in series it dominated
# the build.
HARVEST_WORKERS = 8
# Portrait downloads, which feed a single-threaded encoder. Which half is
# expensive depends on the encoder, so the count comes from the same place the
# search path gets it: 8 when inference dominates, 16 when a GPU has made the
# network the constraint instead. See sigil/search/matcher.py.

# Article titles that are never people, but do rank highly.
SKIP_PREFIXES = ("Special:", "Wikipedia:", "Main_Page", "Portal:", "File:",
                 "Help:", "Category:", "Talk:", "विशेष:", "சிறப்பு:")


def vectors_digest(vectors: np.ndarray) -> str:
    """Content hash of an index's embedding matrix.

    The index is two files - the vectors and the names - written separately and
    read back together, and vector *i* is only meaningful paired with name *i*.
    Nothing but this hash can tell a matched pair from two halves of different
    builds, because a row count cannot: two builds of the same size pair
    entirely different people.
    """
    from .evidence import sha256_hex

    return sha256_hex(np.ascontiguousarray(vectors.astype(np.float32)).tobytes())


@dataclass
class Identity:
    name: str
    qid: str
    image_url: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "qid": self.qid,
                "image_url": self.image_url, "source": self.source}


@dataclass
class IdentityHit:
    identity: Identity
    similarity: float


# --------------------------------------------------------------------- harvest


def _months(count: int) -> list[str]:
    from datetime import date

    out, d = [], date.today().replace(day=1)
    for _ in range(count):
        d = (d.replace(day=1) - __import__("datetime").timedelta(days=1)).replace(day=1)
        out.append(f"{d.year}/{d.month:02d}")
    return out


def _in_parallel(items, work):
    """Run ``work`` over ``items`` concurrently, yielding results in order.

    Ordered rather than as-completed on purpose: the batches are independent,
    but merging them in a fixed order is what keeps a rebuild reproducible.
    """
    with ThreadPoolExecutor(max_workers=HARVEST_WORKERS) as pool:
        for _item, result in prefetch(pool, items, work, HARVEST_WORKERS * 2):
            yield result


def _titles_for(session, lang: str, ym: str) -> set[str]:
    url = PAGEVIEWS.format(wiki=f"{lang}.wikipedia", ym=ym)
    titles: set[str] = set()
    try:
        r = session.get(url, timeout=30)
        if r.status_code != 200:
            return titles
        for art in r.json()["items"][0]["articles"]:
            t = art["article"]
            if not t.startswith(SKIP_PREFIXES):
                titles.add(t)
    except Exception:  # noqa: BLE001 - a missing month is not fatal
        return titles
    return titles


def popular_titles(session, langs: Iterable[str], months: int = 3) -> dict[str, set[str]]:
    """Most-viewed article titles per wiki - where public figures concentrate.

    One request per (wiki, month). They are independent, so they go out
    concurrently; results are merged in a fixed order so a rebuild with the
    same inputs produces the same index.
    """
    tasks = [(lang, ym) for lang in langs for ym in _months(months)]
    found: dict[str, set[str]] = {}
    for (lang, _ym), titles in zip(
        tasks, _in_parallel(tasks, lambda t: _titles_for(session, *t)), strict=True
    ):
        if titles:
            found.setdefault(lang, set()).update(titles)
    return found


def _chunks(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def resolve_people(session, lang: str, titles: list[str]) -> list[Identity]:
    """Join titles -> Wikidata item -> (is human?, portrait) -> Identity."""
    api = f"https://{lang}.wikipedia.org/w/api.php"
    qid_by_title: dict[str, str] = {}
    thumb_by_title: dict[str, str] = {}

    def fetch_pages(batch: list[str]) -> list[dict]:
        try:
            r = session.get(api, params={
                "action": "query", "format": "json", "formatversion": "2",
                "prop": "pageprops|pageimages", "ppprop": "wikibase_item",
                "piprop": "original", "titles": "|".join(batch),
            }, timeout=40)
            return r.json().get("query", {}).get("pages", [])
        except Exception:  # noqa: BLE001 - one lost batch is not a failed build
            return []

    for pages in _in_parallel(list(_chunks(titles, 40)), fetch_pages):
        for pg in pages:
            qid = (pg.get("pageprops") or {}).get("wikibase_item")
            img = (pg.get("original") or {}).get("source")
            if qid and img:
                qid_by_title[pg["title"]] = qid
                thumb_by_title[pg["title"]] = img

    out: list[Identity] = []
    # Reverse the map once; looking the title up per entity turned this into an
    # O(titles x entities) scan over a few thousand of each.
    title_by_qid = {q: t for t, q in qid_by_title.items()}
    qids = list(qid_by_title.values())
    label_lang = "en"

    def fetch_entities(batch: list[str]) -> dict:
        try:
            r = session.get("https://www.wikidata.org/w/api.php", params={
                "action": "wbgetentities", "format": "json",
                "ids": "|".join(batch), "props": "claims|labels",
                "languages": f"{label_lang}|{lang}",
            }, timeout=40)
            return r.json().get("entities", {})
        except Exception:  # noqa: BLE001 - one lost batch is not a failed build
            return {}

    for entities in _in_parallel(list(_chunks(qids, 40)), fetch_entities):
        for qid, ent in entities.items():
            claims = ent.get("claims", {})
            # P31 (instance of) must include Q5 (human). Without this the index
            # fills up with film posters and album covers, which do contain faces.
            is_human = any(
                (c.get("mainsnak", {}).get("datavalue", {}).get("value", {}) or {}).get("id") == "Q5"
                for c in claims.get("P31", [])
            )
            if not is_human:
                continue
            labels = ent.get("labels", {})
            name = (labels.get(label_lang) or labels.get(lang) or {}).get("value")
            if not name:
                continue
            title = title_by_qid.get(qid)
            if title is None:
                continue
            out.append(Identity(name=name, qid=qid,
                                image_url=thumb_by_title[title],
                                source=f"{lang}.wikipedia"))
    return out


# ----------------------------------------------------------------------- build


def build_index(
    encoder,
    langs: Iterable[str] = DEFAULT_LANGS,
    months: int = 3,
    limit: int | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> int:
    """Harvest, encode and persist the identity index. Returns the face count."""
    say = on_progress or (lambda _: None)
    session = make_session()

    say("collecting popular article titles")
    by_lang = popular_titles(session, langs, months)

    people: dict[str, Identity] = {}
    for lang, titles in by_lang.items():
        found = resolve_people(session, lang, sorted(titles))
        for ident in found:
            people.setdefault(ident.qid, ident)
        say(f"  {lang}: {len(titles)} titles -> {len(found)} people "
            f"({len(people)} unique so far)")

    identities = list(people.values())
    if limit:
        identities = identities[:limit]
    say(f"encoding {len(identities)} portraits")

    # Fetching Wikipedia's `original` lead images - often several thousand
    # pixels - and handing them to a detector that immediately fits them into
    # 640x640 looks like obvious waste. Measured over 26 real portraits against
    # the API's own 640px thumbnails, it is not:
    #
    #   bytes      22.1 MB -> 5.6 MB      four times smaller
    #   download     21.5s -> 20.7s       unchanged; this is latency, not bandwidth
    #   encoding      8.1s -> 8.9s        unchanged; the detector normalises anyway
    #   faces found  26/26 -> 26/26       no recall lost
    #   embeddings   agreement 0.9886 mean, 0.9323 worst, 10 of 26 below 0.99
    #
    # So there is no time to win and a real cost to pay: the index would hold
    # vectors measurably different from the ones a full-resolution probe
    # produces, across a 0.45 threshold. The hour this build takes is inference,
    # and inference does not care how many pixels arrived.

    vectors: list[np.ndarray] = []
    kept: list[Identity] = []

    def grab(ident: Identity) -> bytes | None:
        return fetch_image(session, ident.image_url, 30.0)

    # A bounded window, not pool.map: map submits every task up front, so on a
    # few thousand portraits it queues every download at once and holds their
    # bytes in memory ahead of an encoder that is thousands of images behind.
    partial = False
    try:
        workers = download_workers(encoder, Config())
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for done, (ident, blob) in enumerate(
                prefetch(pool, identities, grab, workers * PREFETCH_FACTOR), 1
            ):
                if done % 100 == 0:
                    say(f"  {done}/{len(identities)} · {len(kept)} usable faces")
                if not blob:
                    continue
                img = decode_image(blob)
                if img is None:
                    continue
                face = largest_face(encoder.detect_and_encode(img))
                if face is None:
                    continue
                vectors.append(face.embedding.astype(np.float32))
                kept.append(ident)
    except KeyboardInterrupt:
        # Encoding a full index is the better part of an hour. Throwing that
        # away on a Ctrl-C would be the wrong answer to "this is taking too
        # long" - a smaller index is a working index.
        partial = True
        say(f"  interrupted · keeping the {len(kept)} portraits already encoded")

    if not vectors:
        raise RuntimeError("no faces could be encoded - is the network reachable?")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    matrix = np.vstack(vectors)

    # Both files via a temporary and a rename, and the vectors first: the meta
    # is the commit record, and it names the vectors it belongs to. A build
    # interrupted between the two writes then leaves an older meta whose hash
    # does not match the newer vectors, which `load` refuses - rather than a
    # pair of files from different builds that it would read as one index.
    tmp_vectors = INDEX_VECTORS.with_name(INDEX_VECTORS.name + ".tmp.npz")
    np.savez_compressed(tmp_vectors, vectors=matrix)
    tmp_vectors.replace(INDEX_VECTORS)

    tmp_meta = INDEX_META.with_name(INDEX_META.name + ".tmp")
    tmp_meta.write_text(json.dumps({
        "backend": encoder.name,
        "model": encoder.model,
        "count": len(kept),
        "vectors_sha256": vectors_digest(matrix),
        "langs": list(langs),
        "months": months,
        "partial": partial,
        "identities": [i.to_dict() for i in kept],
    }, ensure_ascii=False, indent=1))
    tmp_meta.replace(INDEX_META)
    say(f"index written: {len(kept)} faces ({encoder.name})"
        + (" · partial, rerun to complete it" if partial else ""))
    return len(kept)


# ------------------------------------------------------------------ query side


class IdentityIndex:
    def __init__(self, vectors: np.ndarray, identities: list[Identity], backend: str,
                 partial: bool = False) -> None:
        self.vectors = vectors
        self.identities = identities
        self.backend = backend
        # True when the build was interrupted, so "no match" may just mean
        # "not harvested yet" rather than "not a public figure".
        self.partial = partial

    @classmethod
    def load(cls, encoder=None) -> IdentityIndex:
        if not (INDEX_VECTORS.exists() and INDEX_META.exists()):
            raise FileNotFoundError(
                "no identity index yet - build one with: sigil index build"
            )
        meta = json.loads(INDEX_META.read_text())
        if encoder is not None and meta["backend"] != encoder.name:
            # Embeddings from different models are not comparable at all; a
            # silent mismatch would return confident nonsense.
            raise RuntimeError(
                f"identity index was built with '{meta['backend']}' but the active "
                f"backend is '{encoder.name}' - rebuild it with: sigil index build"
            )
        vectors = np.load(INDEX_VECTORS)["vectors"]
        identities = [Identity(**d) for d in meta["identities"]]

        # The two files are written separately, so they can disagree: a build
        # interrupted between the writes, or one file restored from a different
        # one. Vector i is only meaningful paired with identity i, so a
        # mismatch does not degrade the answer - it attaches the wrong person's
        # name to a face, with full confidence. That is the worst output this
        # tool can produce, and it is the same reason the backend check above
        # exists.
        if len(vectors) != len(identities):
            raise RuntimeError(
                f"identity index is inconsistent: {len(vectors)} vectors against "
                f"{len(identities)} names. The two files are from different "
                "builds - rebuild with: sigil index build"
            )
        expected = meta.get("vectors_sha256")
        if expected and expected != vectors_digest(vectors):
            raise RuntimeError(
                "identity index is inconsistent: the vectors do not match the "
                "hash recorded when the names were written, so the two files "
                "are from different builds and every name would be attached to "
                "the wrong face. Rebuild with: sigil index build"
            )
        return cls(vectors, identities, meta["backend"], bool(meta.get("partial", False)))

    def __len__(self) -> int:
        return len(self.identities)

    def query(self, embedding: np.ndarray, top: int = 5) -> list[IdentityHit]:
        """Rank every indexed face against the probe. One matrix multiply."""
        v = np.asarray(embedding, dtype=np.float32).ravel()
        v = v / (np.linalg.norm(v) or 1.0)
        sims = self.vectors @ v
        order = np.argsort(-sims)[:top]
        return [IdentityHit(self.identities[i], float(sims[i])) for i in order]


def identify(encoder, face, top: int = 5) -> list[IdentityHit]:
    return IdentityIndex.load(encoder).query(face.embedding, top=top)
