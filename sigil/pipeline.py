"""The end-to-end pipeline: face scan -> live search -> match -> anchor -> verify."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from .chain import ChainClient, Verification
from .config import Config
from .evidence import Evidence, MatchRef, ProbeRef, sha256_hex, utc_now
from .face import Face, decode_image, largest_face, load_encoder
from .search import BlueskyProvider, SerpApiLensProvider
from .search.matcher import MatchResult, search_and_match


class PipelineError(RuntimeError):
    pass


@dataclass
class PipelineResult:
    probe_face: Face
    probe_ref: ProbeRef
    match: MatchResult
    evidence: Evidence | None = None
    anchor: dict[str, Any] | None = None
    verification: Verification | None = None
    providers_used: list[str] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return self.evidence is not None


def load_probe_bytes(source: str, cfg: Config) -> tuple[bytes, str | None]:
    """Read the probe from a path or a URL. Returns (bytes, public_url_or_None)."""
    if source.startswith(("http://", "https://")):
        r = requests.get(source, timeout=cfg.http_timeout,
                         headers={"User-Agent": "sigil"})
        r.raise_for_status()
        return r.content, source
    p = Path(source)
    if not p.exists():
        raise PipelineError(f"probe image not found: {source}")
    return p.read_bytes(), None


def scan_probe(image_bytes: bytes, cfg: Config) -> tuple[Face, ProbeRef, Any]:
    """Detect and encode the input face. This is step one of the pipeline."""
    encoder = load_encoder(cfg.face_backend)
    img = decode_image(image_bytes)
    if img is None:
        raise PipelineError("probe image could not be decoded as an image")

    faces = encoder.detect_and_encode(img)
    face = largest_face(faces)
    if face is None:
        raise PipelineError(
            "no face detected in the probe image - try a clearer, front-facing photo"
        )

    ref = ProbeRef(
        image_sha256=sha256_hex(image_bytes),
        embedding_sha256=face.embedding_sha256,
        backend=encoder.name,
        model=encoder.model,
        bbox=face.bbox,
        det_score=round(face.det_score, 4),
    )
    return face, ref, encoder


def build_providers(cfg: Config, probe_url: str | None) -> list:
    providers: list = [BlueskyProvider(cfg)]
    if SerpApiLensProvider.available_for(cfg, probe_url):
        providers.append(SerpApiLensProvider(cfg, probe_url))
    return providers


def run_pipeline(
    source: str,
    query: str,
    cfg: Config,
    do_anchor: bool = True,
    on_progress: Callable[[int, int, float], None] | None = None,
) -> PipelineResult:
    image_bytes, probe_url = load_probe_bytes(source, cfg)
    face, probe_ref, encoder = scan_probe(image_bytes, cfg)
    threshold = cfg.threshold_for(encoder.name)

    providers = build_providers(cfg, probe_url)
    match = search_and_match(
        encoder, face, providers, query, threshold, cfg, on_progress=on_progress
    )

    result = PipelineResult(
        probe_face=face,
        probe_ref=probe_ref,
        match=match,
        providers_used=[p.name for p in providers],
    )
    if not match.found:
        return result

    best = match.best
    c = best.candidate
    result.evidence = Evidence(
        probe=probe_ref,
        match=MatchRef(
            platform=c.platform,
            post_url=c.post_url,
            post_uri=c.post_uri,
            author_handle=c.author_handle,
            author_did=c.author_did,
            author_display_name=c.author_display_name,
            text=c.text,
            image_url=c.image_url,
            image_sha256=best.image_sha256,
            created_at=c.created_at,
            discovered_via=c.discovered_via,
        ),
        similarity=best.similarity,
        threshold=threshold,
        searched_at=utc_now(),
        search_trace=match.trace,
    )

    if do_anchor:
        client = ChainClient(cfg)
        result.anchor = client.anchor(result.evidence)
        result.verification = client.verify(
            result.evidence, probe_embedding_sha256=probe_ref.embedding_sha256
        )

    return result
