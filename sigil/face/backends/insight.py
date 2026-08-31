"""InsightFace backend: RetinaFace detection + ArcFace (w600k_r50) 512-d embeddings."""

from __future__ import annotations

import contextlib
import io
import os
import warnings

import numpy as np

from ..encoder import Face


def _quiet_onnxruntime() -> None:
    """Turn down onnxruntime's own logger.

    Its provider banner is written from C++ straight to fd 2, so neither
    redirect_stderr nor a StringIO swap can catch it - the library's own
    severity setting is the only real lever. 3 = warnings and above.
    """
    try:
        import onnxruntime

        onnxruntime.set_default_logger_severity(3)
    except Exception:  # noqa: BLE001 - purely cosmetic
        pass


@contextlib.contextmanager
def _quiet():
    """Muffle insightface's model-loading chatter.

    The library prints provider and model banners straight to stdout on load,
    which would shred the CLI's rendered output. Nothing here is an error path,
    so it is swallowed rather than surfaced.
    """
    warnings.filterwarnings("ignore", category=FutureWarning)
    # onnxruntime writes its provider banner to stderr, insightface writes its
    # model banners to stdout, so both need covering. Scoped tightly to model
    # load and inference calls so it cannot swallow a real error elsewhere.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


class InsightFaceBackend:
    name = "insightface"
    model = "buffalo_l/w600k_r50"

    def __init__(self, det_size: int = 640) -> None:
        from insightface.app import FaceAnalysis

        providers = ["CPUExecutionProvider"]
        if os.getenv("SIGIL_GPU", "").lower() in ("1", "true", "yes"):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        # genderage and landmark models are dead weight here; we only need
        # detection + recognition, and skipping them roughly halves load time.
        _quiet_onnxruntime()
        with _quiet():
            self.app = FaceAnalysis(
                name="buffalo_l",
                providers=providers,
                allowed_modules=["detection", "recognition"],
            )
            self.app.prepare(ctx_id=0 if "CUDAExecutionProvider" in providers else -1,
                             det_size=(det_size, det_size))

    def detect_and_encode(self, image_bgr: np.ndarray) -> list[Face]:
        out: list[Face] = []
        with _quiet():
            detected = self.app.get(image_bgr)
        for f in detected:
            emb = getattr(f, "normed_embedding", None)
            if emb is None:
                continue
            x1, y1, x2, y2 = (int(v) for v in f.bbox)
            out.append(
                Face(
                    embedding=np.asarray(emb, dtype=np.float32),
                    bbox=[x1, y1, x2, y2],
                    det_score=float(getattr(f, "det_score", 0.0)),
                )
            )
        return out
