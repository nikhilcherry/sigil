"""The end-to-end pipeline: face scan -> live search -> match -> anchor -> verify."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
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
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> PipelineResult:
    emit = on_event or (lambda _: None)

    emit({"type": "stage", "stage": "scan", "status": "start"})
    image_bytes, probe_url = load_probe_bytes(source, cfg)
    face, probe_ref, encoder = scan_probe(image_bytes, cfg)
    threshold = cfg.threshold_for(encoder.name)
    emit({
        "type": "probe",
        "backend": probe_ref.backend,
        "model": probe_ref.model,
        "bbox": probe_ref.bbox,
        "det_score": probe_ref.det_score,
        "image_sha256": probe_ref.image_sha256,
        "embedding_sha256": probe_ref.embedding_sha256,
        "threshold": threshold,
        "crop": face_crop_data_uri(image_bytes, probe_ref.bbox),
    })
    emit({"type": "stage", "stage": "scan", "status": "done"})

    emit({"type": "stage", "stage": "search", "status": "start"})
    providers = build_providers(cfg, probe_url)
    emit({"type": "providers", "providers": [p.name for p in providers]})
    match = search_and_match(
        encoder, face, providers, query, threshold, cfg, on_event=on_event
    )
    emit({
        "type": "stage", "stage": "search", "status": "done",
        "examined": match.images_examined,
        "with_faces": match.images_with_faces,
        "faces": match.faces_examined,
        "calls": sum(len(t["calls"]) for t in match.trace),
    })

    result = PipelineResult(
        probe_face=face,
        probe_ref=probe_ref,
        match=match,
        providers_used=[p.name for p in providers],
    )
    if not match.found:
        emit({
            "type": "nomatch",
            "best": round(match.ranked[0].similarity, 4) if match.ranked else 0.0,
            "examined": match.images_examined,
            "threshold": threshold,
        })
        emit({"type": "done", "found": False})
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

    emit({
        "type": "match",
        "similarity": round(best.similarity, 4),
        "threshold": threshold,
        "hash": result.evidence.evidence_hash_hex(),
        "match": asdict(result.evidence.match),
    })

    if not do_anchor:
        emit({"type": "done", "found": True})
        return result

    emit({"type": "stage", "stage": "anchor", "status": "start"})
    client = ChainClient(cfg)
    result.anchor = client.anchor(result.evidence)
    emit({"type": "anchor", **_jsonable(result.anchor)})
    emit({"type": "stage", "stage": "anchor", "status": "done"})

    emit({"type": "stage", "stage": "verify", "status": "start"})
    result.verification = client.verify(
        result.evidence, probe_embedding_sha256=probe_ref.embedding_sha256
    )
    emit({"type": "verification", **verification_payload(result.verification)})
    emit({"type": "stage", "stage": "verify", "status": "done"})
    emit({"type": "done", "found": True})

    return result


def _jsonable(d: dict[str, Any]) -> dict[str, Any]:
    """Receipts carry HexBytes and AttributeDicts; flatten them for transport."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, (bytes, bytearray)):
            out[k] = "0x" + v.hex()
        elif isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        else:
            out[k] = str(v)
    return out


def verification_payload(v) -> dict[str, Any]:
    return {
        "evidence_hash": v.evidence_hash,
        "anchored": v.anchored,
        "similarity_matches": v.similarity_matches,
        "subject_matches": v.subject_matches,
        "probe_matches": v.probe_matches,
        "source_image_intact": v.source_image_intact,
        "on_chain": v.on_chain,
        "notes": v.notes,
        "ok": v.ok,
    }


def face_crop_data_uri(image_bytes: bytes, bbox: list[int], size: int = 168) -> str | None:
    """A small JPEG of just the detected face, for the UI to show what it locked onto."""
    import base64

    import cv2

    img = decode_image(image_bytes)
    if img is None:
        return None
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    pad = int(0.18 * max(x2 - x1, y2 - y1))
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
