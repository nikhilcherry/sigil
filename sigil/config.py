"""Runtime configuration, resolved from environment with sane zero-config defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.getenv("SIGIL_MODELS_DIR", PROJECT_ROOT / "models"))
ARTIFACTS_DIR = Path(os.getenv("SIGIL_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))
STATE_PATH = ARTIFACTS_DIR / "chain-state.json"

# Cosine-similarity cut-offs, per recognition model. Below these two faces are
# treated as different people. ArcFace (w600k_r50) and SFace disagree on scale,
# so the threshold has to travel with the backend rather than being global.
DEFAULT_THRESHOLDS = {
    "insightface": 0.38,
    "opencv": 0.363,  # OpenCV Zoo's published SFace cosine threshold
}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


def _env_threshold() -> float | None:
    """Parse SIGIL_THRESHOLD, refusing rather than defaulting on nonsense.

    Every other knob here falls back to its default when it cannot be parsed,
    because a malformed page size is not worth ending a run over. The threshold
    is different: it is the decision boundary between "this is the same person"
    and "this is not". Silently substituting 0.38 for a typo'd 0.5 would accept
    matches the operator meant to reject, and nothing downstream would show it.
    """
    raw = os.getenv("SIGIL_THRESHOLD", "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(
            f"SIGIL_THRESHOLD must be a number, got {raw!r}. It sets the match "
            "decision boundary, so it is not defaulted silently."
        ) from None
    if not -1.0 <= value <= 1.0:
        raise ValueError(
            f"SIGIL_THRESHOLD must be a cosine similarity between -1 and 1, "
            f"got {value}. Above 1 nothing can ever match; below -1 everything does."
        )
    return value


@dataclass
class Config:
    face_backend: str = field(default_factory=lambda: os.getenv("SIGIL_FACE_BACKEND", "auto"))
    threshold: float | None = field(default_factory=_env_threshold)

    # --- search ---
    max_actors: int = field(default_factory=lambda: _env_int("SIGIL_MAX_ACTORS", 25))
    posts_per_actor: int = field(default_factory=lambda: _env_int("SIGIL_POSTS_PER_ACTOR", 20))
    max_images: int = field(default_factory=lambda: _env_int("SIGIL_MAX_IMAGES", 200))
    http_timeout: float = field(default_factory=lambda: _env_float("SIGIL_HTTP_TIMEOUT", 20.0))
    # 0 means "decide from the encoder": the right number depends on whether
    # the run is inference-bound or network-bound. See search/matcher.py.
    download_workers: int = field(
        default_factory=lambda: _env_int("SIGIL_DOWNLOAD_WORKERS", 0)
    )
    bluesky_handle: str | None = field(default_factory=lambda: os.getenv("BLUESKY_HANDLE"))
    bluesky_app_password: str | None = field(
        default_factory=lambda: os.getenv("BLUESKY_APP_PASSWORD")
    )
    serpapi_key: str | None = field(default_factory=lambda: os.getenv("SERPAPI_KEY"))
    google_vision_key: str | None = field(
        default_factory=lambda: os.getenv("GOOGLE_VISION_API_KEY")
    )

    # --- chain ---
    chain_backend: str = field(default_factory=lambda: os.getenv("SIGIL_CHAIN", "local"))
    rpc_url: str | None = field(default_factory=lambda: os.getenv("SIGIL_RPC_URL"))
    private_key: str | None = field(default_factory=lambda: os.getenv("SIGIL_PRIVATE_KEY"))
    contract_address: str | None = field(default_factory=lambda: os.getenv("SIGIL_CONTRACT"))
    explorer_base: str = field(
        default_factory=lambda: os.getenv("SIGIL_EXPLORER", "https://amoy.polygonscan.com")
    )

    # Salt for the subject commitment. Anchoring a raw face embedding on a public
    # chain would publish an irrevocable biometric; a salted commitment proves
    # "same subject" without revealing the face.
    subject_salt: str = field(
        default_factory=lambda: os.getenv("SIGIL_SUBJECT_SALT", "sigil-default-salt")
    )

    def threshold_for(self, backend: str) -> float:
        if self.threshold is not None:
            return self.threshold
        return DEFAULT_THRESHOLDS.get(backend, 0.38)


def ensure_dirs() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
