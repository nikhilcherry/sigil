"""Face detection and encoding, behind one interface with two interchangeable backends.

Backends are swappable on purpose: ``insightface`` (ArcFace w600k_r50, 512-d) is
the accurate default, and ``opencv`` (YuNet + SFace, 128-d) is a small, always-
installable fallback so the pipeline still runs on a machine that cannot build
the heavier stack. Both emit L2-normalised vectors, so cosine similarity means
the same thing either way - only the decision threshold differs, which is why
thresholds live per-backend in config.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import cv2
import numpy as np


@dataclass
class Face:
    embedding: np.ndarray  # float32, L2-normalised
    bbox: list[int]  # [x1, y1, x2, y2]
    det_score: float

    @property
    def embedding_bytes(self) -> bytes:
        return np.asarray(self.embedding, dtype=np.float32).tobytes()

    @property
    def embedding_sha256(self) -> str:
        return hashlib.sha256(self.embedding_bytes).hexdigest()


@runtime_checkable
class FaceEncoder(Protocol):
    name: str
    model: str

    def detect_and_encode(self, image_bgr: np.ndarray) -> list[Face]: ...


def decode_image(data: bytes) -> np.ndarray | None:
    """Decode arbitrary image bytes to BGR. Returns None on anything unreadable.

    Candidate bytes come off the open internet, so this has to survive empty
    responses, HTML error pages served with an image content-type, and truncated
    downloads. cv2.imdecode raises on some of those rather than returning None.
    """
    if not data:
        return None
    try:
        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:
        return None
    if img is None or img.size == 0:
        return None
    return img


def _l2(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two already-normalised embeddings."""
    return float(np.dot(_l2(a), _l2(b)))


def largest_face(faces: list[Face]) -> Face | None:
    """Pick the dominant face - the subject of a portrait is the biggest one."""
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def load_encoder(backend: str = "auto"):
    """Resolve a backend name to a live encoder.

    ``auto`` prefers insightface and silently degrades to opencv, so a fresh
    clone works before anyone has downloaded a 300 MB model pack.
    """
    if backend in ("auto", "insightface"):
        try:
            from .backends.insight import InsightFaceBackend

            return InsightFaceBackend()
        except Exception as exc:  # noqa: BLE001 - fall through to the fallback
            if backend == "insightface":
                raise RuntimeError(f"insightface backend unavailable: {exc}") from exc

    from .backends.opencv import OpenCVBackend

    return OpenCVBackend()
