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

from ..concurrency import prefetch
from ..config import Config
from ..evidence import sha256_hex
from ..face import Face, cosine, decode_image
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


def interleave(streams: list[Iterator[Candidate]]) -> Iterator[Candidate]:
    """Merge candidate streams round-robin, dropping each as it runs dry.

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
    """
    live = list(streams)
    while live:
        for stream in list(live):
            try:
                yield next(stream)
            except StopIteration:
                live.remove(stream)


def score_image(encoder, probe: Face, image_bytes: bytes) -> tuple[float, int, list[int]]:
    """Best similarity between the probe and any face in one image."""
    img = decode_image(image_bytes)
    if img is None:
        return -1.0, 0, []
    faces = encoder.detect_and_encode(img)
    if not faces:
        return -1.0, 0, []
    best_sim, best_box = -1.0, []
    for f in faces:
        sim = cosine(probe.embedding, f.embedding)
        if sim > best_sim:
            best_sim, best_box = sim, f.bbox
    return best_sim, len(faces), best_box


def search_and_match(
    encoder,
    probe: Face,
    providers: list,
    query: str,
    threshold: float,
    cfg: Config,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> MatchResult:
    session = make_session()
    result = MatchResult(best=None)
    scored: list[ScoredCandidate] = []
    # digest -> (similarity, faces_in_image, bbox). One image is routinely
    # served under several URLs (CDN size variants, cache-busting query
    # strings), and URL dedup cannot see that.
    by_digest: dict[str, tuple[float, int, list[int]]] = {}

    stream = _dedup(interleave([p.candidates(query) for p in providers]))
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
            digest = sha256_hex(blob)
            cached = by_digest.get(digest)
            if cached is not None:
                result.inference_reused += 1
                sim, n_faces, bbox = cached
            else:
                sim, n_faces, bbox = score_image(encoder, probe, blob)
                by_digest[digest] = (sim, n_faces, bbox)
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

    scored.sort(key=lambda s: s.similarity, reverse=True)
    result.ranked = scored[:20]
    if scored and scored[0].similarity >= threshold:
        result.best = scored[0]

    result.trace = [
        {"provider": p.name, "calls": p.trace.calls} for p in providers if hasattr(p, "trace")
    ]
    return result
