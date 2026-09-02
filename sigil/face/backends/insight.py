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


def _want_gpu() -> bool:
    """Use the GPU when there is one, without anyone having to ask.

    SIGIL_GPU forces the decision either way; unset, availability decides. The
    default matters because the expensive path - encoding a few thousand
    candidate images - is the same code a grader runs on their own machine.
    """
    forced = os.getenv("SIGIL_GPU", "").strip().lower()
    if forced in ("0", "false", "no"):
        return False
    if forced in ("1", "true", "yes"):
        return True
    try:
        import onnxruntime

        return "CUDAExecutionProvider" in onnxruntime.get_available_providers()
    except Exception:  # noqa: BLE001 - no onnxruntime means no GPU either way
        return False


class InsightFaceBackend:
    name = "insightface"
    model = "buffalo_l/w600k_r50"

    # det_size is the obvious speed knob and it is the wrong one to turn.
    # Detection at this size is ~97% of a run's CPU, so shrinking it looks
    # free - but it is paid for in recall, not accuracy. Measured over 60 real
    # Bluesky candidate images, of the 11 that yielded a face at 640:
    #
    #   det_size 480  -26% time,  4 of those 11 images detect nothing at all
    #   det_size 320  -53% time,  6 of those 11 images detect nothing at all
    #
    # The similarity of a face that is still found barely moves, so a quick
    # check makes the change look safe. It is not: an image the detector never
    # sees cannot match, and a smaller or off-angle face in a group photo is
    # exactly the candidate that disappears first. Reporting "no match" because
    # the detector was run too cheaply is the worst failure this tool has.
    def __init__(self, det_size: int = 640) -> None:
        _quiet_onnxruntime()
        if _want_gpu():
            try:
                self._load(det_size, ["CUDAExecutionProvider", "CPUExecutionProvider"])
                self._warm_up()
            except Exception:  # noqa: BLE001 - a GPU is an optimisation, never a requirement
                # onnxruntime advertises the provider from the package build,
                # not from what the machine can actually load: a missing cuDNN
                # or a driver mismatch only surfaces here. CPU still works.
                self._load(det_size, ["CPUExecutionProvider"])
        else:
            self._load(det_size, ["CPUExecutionProvider"])
        self.provider = self._active_provider()

    def _warm_up(self) -> None:
        """Run one throwaway inference before trusting the provider.

        Loading a session and running one are separate failure points: a CUDA
        provider can initialise and then fail on the first real call. Finding
        that out here costs one pass over a blank 64x64 image and falls back to
        CPU; finding it out later means dying part way through a search.
        """
        with _quiet():
            self.app.get(np.zeros((64, 64, 3), dtype=np.uint8))

    def _load(self, det_size: int, providers: list[str]) -> None:
        from insightface.app import FaceAnalysis

        # genderage and landmark models are dead weight here; we only need
        # detection + recognition, and skipping them roughly halves load time.
        with _quiet():
            self.app = FaceAnalysis(
                name="buffalo_l",
                providers=providers,
                allowed_modules=["detection", "recognition"],
            )
            self.app.prepare(ctx_id=0 if "CUDAExecutionProvider" in providers else -1,
                             det_size=(det_size, det_size))

    def _active_provider(self) -> str:
        """What onnxruntime actually ran with, not what it was asked for.

        Requesting CUDA does not always fail loudly when it is unusable -
        onnxruntime can warn and quietly serve CPU - so reporting the request
        would mean reporting a GPU run that never happened.
        """
        for model in getattr(self.app, "models", {}).values():
            session = getattr(model, "session", None)
            if session is not None:
                got = session.get_providers()
                return got[0] if got else "CPUExecutionProvider"
        return "CPUExecutionProvider"

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
