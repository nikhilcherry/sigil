"""OpenCV backend: YuNet detection + SFace 128-d embeddings.

Zero heavy dependencies beyond OpenCV itself, so this is what keeps the pipeline
runnable on a machine where the insightface wheel will not build.
"""

from __future__ import annotations

import contextlib

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("opencv is required") from exc

# OpenCV 5's DNN backend logs a target-support warning to stderr on every model
# load. It is informational and would otherwise interleave with the CLI output.
with contextlib.suppress(AttributeError):
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

from ...config import MODELS_DIR
from ..encoder import Face, _l2

DETECTOR = "face_detection_yunet_2023mar.onnx"
RECOGNIZER = "face_recognition_sface_2021dec.onnx"


class OpenCVBackend:
    name = "opencv"
    model = "yunet+sface"

    def __init__(self, score_threshold: float = 0.7) -> None:
        det_path = MODELS_DIR / DETECTOR
        rec_path = MODELS_DIR / RECOGNIZER
        missing = [p.name for p in (det_path, rec_path) if not p.exists()]
        if missing:
            raise RuntimeError(
                f"missing model files {missing} in {MODELS_DIR}; run scripts/fetch_models.sh"
            )
        self.detector = cv2.FaceDetectorYN.create(
            str(det_path), "", (320, 320), score_threshold
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(rec_path), "")

    def detect_and_encode(self, image_bgr: np.ndarray) -> list[Face]:
        h, w = image_bgr.shape[:2]
        self.detector.setInputSize((w, h))
        _, raw = self.detector.detect(image_bgr)
        if raw is None:
            return []

        out: list[Face] = []
        for row in raw:
            try:
                aligned = self.recognizer.alignCrop(image_bgr, row)
                feat = self.recognizer.feature(aligned)
            except cv2.error:
                continue
            x, y, bw, bh = (int(v) for v in row[:4])
            out.append(
                Face(
                    embedding=_l2(np.asarray(feat, dtype=np.float32)),
                    bbox=[x, y, x + bw, y + bh],
                    det_score=float(row[14]),
                )
            )
        return out
