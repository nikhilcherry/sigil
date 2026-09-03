"""Turns a stream of candidate images into a ranked, face-verified match.

The provider only proposes; nothing is a "match" until the same encoder that
read the probe also reads the candidate and the two vectors clear the
threshold. That is the whole point - the search step is a retrieval heuristic,
and the face model is the judge.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..concurrency import prefetch
from ..config import Config
from ..evidence import sha256_hex
from ..face import Face, cosine, decode_image
from ..provenance import IDENTITY, claim_for, fingerprint, photo_similarity
from .base import Candidate
from .http import fetch_image, make_session

DOWNLOAD_WORKERS = 8
# How many downloads may be in flight (or finished and waiting) ahead of the
# encoder. Larger than the worker count on purpose: it is the read-ahead buffer
# that keeps the encoder from ever waiting on the network.
PREFETCH = DOWNLOAD_WORKERS * 3


@dataclass
class ScoredCandidate:
    candidate: Candidate
    similarity: float
    image_sha256: str
    faces_in_image: int
    matched_bbox: list[int]
    # How close the whole picture is to the probe's, judged without the face
    # model. See sigil/provenance.py.
    photo_similarity: float = 0.0
    # Position by raw face similarity across every candidate scored, not just
    # the ones kept for the report. The anchored match is routinely outside the
    # top 20, so its real rank has to travel with it.
    rank: int = 0

    @property
    def claim(self) -> str:
        """"identity" for a different photograph, "provenance" for this one again."""
        return claim_for(self.photo_similarity)


@dataclass
class MatchResult:
    best: ScoredCandidate | None
    ranked: list[ScoredCandidate] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    images_examined: int = 0
    images_with_faces: int = 0
    faces_examined: int = 0
    inference_reused: int = 0

    @property
    def found(self) -> bool:
        return self.best is not None


def _dedup(it: Iterable[Candidate]) -> Iterator[Candidate]:
    seen: set[str] = set()
    for c in it:
        if c.image_url in seen:
            continue
        seen.add(c.image_url)
        yield c


def interleave(
    streams: list[tuple[str, Iterator[Candidate]]],
    on_error: Callable[[str, Exception], None] | None = None,
) -> Iterator[Candidate]:
    """Merge named candidate streams round-robin, dropping each as it runs dry.

    Chaining them instead - which is what this replaces - silently starved
    every arm after the first. ``max_images`` truncates the merged stream, and
    Bluesky alone yields more candidates than the default cap, so a configured
    Google Vision arm was listed as a provider, reported in the trace, and
    never advanced once: its generator body never ran, so it never even called
    the API. An arm that cannot contribute is worse than an absent one, since
    the run claims coverage it does not have.

    Round-robin also puts each arm's own best guesses early, which is what a
    truncated budget spends itself on: Bluesky orders avatars ahead of feed
    images, and Vision orders page-anchored matches ahead of merely similar
    ones.

    An arm that raises is dropped rather than allowed to end the run, and
    ``on_error`` is told which one. Chaining hid the need for this by accident:
    the optional arms sat behind Bluesky, which under the default cap never ran
    out, so their code never executed and could not fail. Round-robin advances
    every arm from the first pass, which is the point - and it means a single
    unlucky response from an optional provider would otherwise take the whole
    search down with it.
    """
    live = list(streams)
    while live:
        for entry in list(live):
            _name, stream = entry
            try:
                yield next(stream)
            except StopIteration:
                live.remove(entry)
            except Exception as exc:  # noqa: BLE001 - an optional arm must not end a run
                live.remove(entry)
                if on_error is not None:
                    on_error(_name, exc)


def score_image(
    encoder, probe: Face, image_bytes: bytes,
    probe_fingerprint: np.ndarray | None = None,
) -> tuple[float, int, list[int], float]:
    """Best face similarity in one image, plus how close the picture itself is.

    Both come from one decode: the image is already in memory, and the whole-
    image fingerprint costs microseconds next to the detector.
    """
    img = decode_image(image_bytes)
    if img is None:
        return -1.0, 0, [], 0.0
    photo = (photo_similarity(probe_fingerprint, fingerprint(img))
             if probe_fingerprint is not None else 0.0)
    faces = encoder.detect_and_encode(img)
    if not faces:
        return -1.0, 0, [], photo
    best_sim, best_box = -1.0, []
    for f in faces:
        sim = cosine(probe.embedding, f.embedding)
        if sim > best_sim:
            best_sim, best_box = sim, f.bbox
    return best_sim, len(faces), best_box, photo


def search_and_match(
    encoder,
    probe: Face,
    providers: list,
    query: str,
    threshold: float,
    cfg: Config,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    probe_image: np.ndarray | None = None,
) -> MatchResult:
    session = make_session()
    probe_fp = None if probe_image is None else fingerprint(probe_image)
    result = MatchResult(best=None)
    scored: list[ScoredCandidate] = []
    # digest -> (similarity, faces_in_image, bbox, photo_similarity). One image
    # is routinely served under several URLs (CDN size variants, cache-busting
    # query strings), and URL dedup cannot see that.
    by_digest: dict[str, tuple[float, int, list[int], float]] = {}

    by_name = {p.name: p for p in providers if hasattr(p, "trace")}

    def arm_failed(name: str, exc: Exception) -> None:
        """Record that an arm died, without recording what it said.

        Only the exception's type is kept. A requests failure carries the URL
        it was fetching, and the Vision arm passes its API key as a query
        parameter - so the message would put a live credential into the
        evidence bundle, which is published and hashed on chain.
        """
        provider = by_name.get(name)
        if provider is not None:
            provider.trace.record(f"{name}.failed",
                                  {"error": type(exc).__name__}, 0)
        if on_event:
            on_event({"type": "provider_failed", "provider": name,
                      "error": type(exc).__name__})

    stream = _dedup(interleave(
        [(p.name, p.candidates(query)) for p in providers], on_error=arm_failed
    ))
    stream = itertools.islice(stream, cfg.max_images)

    def fetch(c: Candidate) -> bytes | None:
        return fetch_image(session, c.image_url, cfg.http_timeout)

    # A cheap pre-filter here does not pay, and it looks like it should.
    # Measured over two real candidate corpora (285 and 244 images), only 22%
    # and 49% of them contain a face at all, so the expensive detector is
    # mostly deciding that there is nothing to decide. Gating on YuNet first -
    # 232 KB, ~16 ms an image against RetinaFace's ~400 ms - and only running
    # the real encoder on images it flags gave 1.24x and 1.17x, because at the
    # only operating point with 100% recall against insightface@640 YuNet
    # still passes about half the face-free images: on open-web material
    # (logos, screenshots, text graphics) it finds faces almost everywhere.
    # Anything faster costs recall, which is the same trade det_size=320 was
    # rejected for. Not worth a second model on the critical path.
    #
    # Network in parallel, inference serially. Inference is the expensive half
    # by a wide margin - a default run spends a couple of minutes in ONNX
    # against a few seconds of download across eight workers - and onnxruntime
    # already saturates the cores from inside, so a parallel outer loop would
    # only fight its intra-op threading. The read-ahead window exists to keep
    # that single encoder fed rather than to make downloading faster.
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        for cand, blob in prefetch(pool, stream, fetch, PREFETCH):
            if not blob:
                continue
            result.images_examined += 1
            # Reuse the score rather than dropping the candidate: identical
            # bytes give an identical verdict, but the *posts* differ, and the
            # post is what gets anchored. Hashing is microseconds; a redundant
            # detect+encode is not.
            #
            # Identical *bytes* is the right granularity, and the looser cache
            # that suggests itself is not. A reverse-image arm returns the same
            # photograph a hundred times over, rescaled and recompressed, so
            # keying this on the whole-image fingerprint instead would collapse
            # them - and it is unsound. Measured over 161 real candidates, even
            # at a fingerprint correlation of 0.9999, where the two pictures are
            # indistinguishable at 32x32, the encoder's verdicts on them differ
            # by up to 0.0613 cosine. That is a sixth of the decision threshold,
            # enough to move a borderline candidate across it, and it is bought
            # for 1.5x. ArcFace is resolution-sensitive at that scale, which is
            # the same fact the det_size note in the insight backend is about.
            digest = sha256_hex(blob)
            cached = by_digest.get(digest)
            if cached is not None:
                result.inference_reused += 1
                sim, n_faces, bbox, photo = cached
            else:
                sim, n_faces, bbox, photo = score_image(
                    encoder, probe, blob, probe_fp
                )
                by_digest[digest] = (sim, n_faces, bbox, photo)
                # Counted only where a comparison actually happened - the
                # report calls this "faces compared", and a reused verdict
                # compared nothing.
                result.faces_examined += n_faces
            if n_faces:
                result.images_with_faces += 1
            if sim >= 0:
                scored.append(
                    ScoredCandidate(
                        candidate=cand,
                        similarity=sim,
                        image_sha256=digest,
                        faces_in_image=n_faces,
                        matched_bbox=bbox,
                        photo_similarity=photo,
                    )
                )
                if on_event:
                    on_event({
                        "type": "candidate",
                        "similarity": round(sim, 4),
                        "handle": cand.author_handle,
                        "display": cand.author_display_name,
                        "image_url": cand.image_url,
                        "post_url": cand.post_url,
                        "via": cand.discovered_via,
                        "faces": n_faces,
                        "hit": sim >= threshold,
                        "photo_similarity": round(photo, 4),
                        "claim": claim_for(photo),
                    })
            # Progress covers every image examined, including the ones with no
            # detectable face. Emitting it only for scored candidates left the
            # live counter reading lower than the count the final report gives,
            # so the UI contradicted itself on screen at the end of a run.
            if on_event:
                on_event({
                    "type": "progress",
                    "examined": result.images_examined,
                    "scored": len(scored),
                    "top": round(max((s.similarity for s in scored), default=0.0), 4),
                })

    # The table stays ranked by raw face similarity, because that is what the
    # model said and hiding it would be the dishonest part.
    scored.sort(key=lambda s: s.similarity, reverse=True)
    for position, candidate in enumerate(scored, 1):
        candidate.rank = position

    result.best = pick_best(scored, threshold)
    result.ranked = scored[:20]
    # The anchored candidate is kept whatever its rank. Preferring a social
    # identity claim over a higher cosine routinely selects something well
    # outside the top 20 - the live AOC avatar sits past 30 behind a wall of
    # republications - and a report that names an anchored match it cannot
    # show contradicts itself on screen.
    if result.best is not None and result.best not in result.ranked:
        result.ranked.append(result.best)

    result.trace = [
        {"provider": p.name, "calls": p.trace.calls} for p in providers if hasattr(p, "trace")
    ]
    return result


def pick_best(
    scored: list[ScoredCandidate], threshold: float
) -> ScoredCandidate | None:
    """Choose what to anchor from the candidates that cleared the threshold.

    Not simply the top of the table, for two reasons that both cut the same way.

    First, what is being looked for is a *social media post*. An open-web page
    that happens to carry the same face is corroboration, not the deliverable,
    so a candidate from a social arm outranks one from an open-web arm.

    Second, a reverse-image arm returns the probe's own photograph republished
    elsewhere, which scores near 1.0 because it *is* the same picture. Ranking
    by cosine alone therefore anchors the weakest evidence available and
    presents it as the strongest; an independent photograph of the same face
    scores lower and proves far more. So an identity claim outranks a
    provenance claim, and cosine only decides between candidates alike on both
    counts.

    Nothing is hidden by this: the table stays ordered by raw similarity, every
    candidate keeps its own label, and when everything that cleared is the
    probe's picture again the best of those is still anchored - a republication
    is a real finding - with the bundle recording it as a provenance claim so
    it cannot be read as the other thing.
    """
    clearing = [s for s in scored if s.similarity >= threshold]
    if not clearing:
        return None
    return max(clearing, key=lambda s: (s.candidate.source_kind == "social",
                                        s.claim == IDENTITY,
                                        s.similarity))
