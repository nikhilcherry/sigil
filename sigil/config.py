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


@dataclass
class Config:
    face_backend: str = field(default_factory=lambda: os.getenv("SIGIL_FACE_BACKEND", "auto"))
    threshold: float | None = field(
        default_factory=lambda: float(os.environ["SIGIL_THRESHOLD"])
        if os.getenv("SIGIL_THRESHOLD")
        else None
    )

    # --- search ---
    max_actors: int = field(default_factory=lambda: _env_int("SIGIL_MAX_ACTORS", 25))
    posts_per_actor: int = field(default_factory=lambda: _env_int("SIGIL_POSTS_PER_ACTOR", 20))
    max_images: int = field(default_factory=lambda: _env_int("SIGIL_MAX_IMAGES", 200))
    http_timeout: float = field(default_factory=lambda: _env_float("SIGIL_HTTP_TIMEOUT", 20.0))
    bluesky_handle: str | None = field(default_factory=lambda: os.getenv("BLUESKY_HANDLE"))
    bluesky_app_password: str | None = field(
        default_factory=lambda: os.getenv("BLUESKY_APP_PASSWORD")
    )
    serpapi_key: str | None = field(default_factory=lambda: os.getenv("SERPAPI_KEY"))

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
