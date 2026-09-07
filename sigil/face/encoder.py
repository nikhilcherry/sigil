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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

# How the run decides which detected face it is about, when there is more than
# one. Measured against each other on LFW in scripts/bench_lfw.py.
SUBJECT_POLICIES = ("largest", "centre")

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


def centre_face(faces: list[Face], shape: tuple[int, ...]) -> Face | None:
    """Pick the face nearest the middle of the frame.

    The other reading of "who is this photograph of". Whoever framed the shot
    put the subject in the middle, so on a photograph with bystanders in it
    this disagrees with ``largest_face`` - and measurably beats it when the
    framing is trustworthy. See ``scripts/bench_lfw.py``.
    """
    if not faces:
        return None
    h, w = shape[0], shape[1]
    cx, cy = w / 2.0, h / 2.0
    return min(
        faces,
        key=lambda f: ((f.bbox[0] + f.bbox[2]) / 2.0 - cx) ** 2
        + ((f.bbox[1] + f.bbox[3]) / 2.0 - cy) ** 2,
    )


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def face_at(faces: list[Face], bbox: Sequence[float]) -> Face | None:
    """The detected face that overlaps ``bbox`` most, or None if none does.

    Verification uses this rather than re-running a selection policy. A bundle
    records the box of the face it was about, so re-checking a probe means
    asking whether *that* face still encodes to the recorded digest - a
    question with the same answer whichever policy chose it. Re-running the
    policy instead would make a bundle anchored under one ``--subject`` fail
    to verify under another, which would be a disagreement about framing
    dressed up as a tampering alarm.
    """
    if not faces:
        return None
    target = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    best = max(faces, key=lambda f: _iou(f.bbox, target))
    return best if _iou(best.bbox, target) > 0.5 else None


def select_subject(
    faces: list[Face],
    policy: str = "largest",
    shape: tuple[int, ...] | None = None,
) -> Face | None:
    """Choose which detected face the run is about.

    ``largest`` is the default because a probe someone hands you is usually of
    the person filling the frame. ``centre`` is right when the framing carries
    the intent instead - and it is the convention benchmark sets are built on,
    which is why the two are measured separately rather than one being
    declared correct.
    """
    if policy not in SUBJECT_POLICIES:
        raise ValueError(
            f"unknown subject policy {policy!r}; expected one of {', '.join(SUBJECT_POLICIES)}"
        )
    if policy == "centre":
        if shape is None:
            raise ValueError("the centre policy needs the image shape to find the centre")
        return centre_face(faces, shape)
    return largest_face(faces)


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
