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

from ..config import Config
from ..evidence import sha256_hex
from ..face import Face, cosine, decode_image
from .base import Candidate
from .http import fetch_image, make_session

DOWNLOAD_WORKERS = 8
BATCH = 16


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

    @property
    def found(self) -> bool:
        return self.best is not None


def _batched(it: Iterator[Candidate], n: int) -> Iterator[list[Candidate]]:
    while chunk := list(itertools.islice(it, n)):
        yield chunk


def _dedup(it: Iterable[Candidate]) -> Iterator[Candidate]:
    seen: set[str] = set()
    for c in it:
        if c.image_url in seen:
            continue
        seen.add(c.image_url)
        yield c


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

    stream = _dedup(
        itertools.chain.from_iterable(p.candidates(query) for p in providers)
    )
    stream = itertools.islice(stream, cfg.max_images)

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        for chunk in _batched(stream, BATCH):
            # Network in parallel, inference serially: the downloads are the
            # slow part, and keeping one thread on the ONNX session avoids
            # fighting onnxruntime's own intra-op threading.
            blobs = list(
                pool.map(lambda c: fetch_image(session, c.image_url, cfg.http_timeout), chunk)
            )
            for cand, blob in zip(chunk, blobs, strict=True):
                if not blob:
                    continue
                result.images_examined += 1
                sim, n_faces, bbox = score_image(encoder, probe, blob)
                result.faces_examined += n_faces
                if n_faces:
                    result.images_with_faces += 1
                if sim < 0:
                    continue
                scored.append(
                    ScoredCandidate(
                        candidate=cand,
                        similarity=sim,
                        image_sha256=sha256_hex(blob),
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
