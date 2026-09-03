"""Compile SigilRegistry.sol, caching the artifact so repeat runs skip solc."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import solcx

from ..config import ARTIFACTS_DIR, PROJECT_ROOT

SOLC_VERSION = "0.8.24"
SOURCE = PROJECT_ROOT / "contracts" / "SigilRegistry.sol"
ARTIFACT = ARTIFACTS_DIR / "SigilRegistry.json"


def _ensure_solc() -> None:
    installed = [str(v) for v in solcx.get_installed_solc_versions()]
    if SOLC_VERSION not in installed:
        solcx.install_solc(SOLC_VERSION)
    solcx.set_solc_version(SOLC_VERSION)


def compile_registry(force: bool = False) -> dict[str, Any]:
    """Return {'abi': [...], 'bytecode': '0x...'} for SigilRegistry.

    The cache is keyed on a hash of the source, not on its modification time.
    An mtime is a weak key for exactly the reason this project keeps
    rediscovering: two edits inside one filesystem tick, or a restore that
    preserves the timestamp, leave it unchanged while the content is not - and
    a stale hit here means deploying bytecode that does not correspond to the
    contract in the repository. For a tool whose claim is that anyone can check
    its records against the source, that is the wrong thing to get wrong
    quietly.
    """
    src = SOURCE.read_bytes()
    src_sha256 = hashlib.sha256(src).hexdigest()
    if ARTIFACT.exists() and not force:
        try:
            cached = json.loads(ARTIFACT.read_text())
            # Recompile when the source moved on, so a contract edit can never
            # be silently shadowed by a stale artifact.
            if cached.get("source_sha256") == src_sha256:
                return cached
        except (json.JSONDecodeError, OSError, AttributeError):
            # The artifact is a cache of a deterministic build. An unreadable
            # one means recompile, not stop - the source is the truth here.
            pass

    _ensure_solc()
    compiled = solcx.compile_files(
        [str(SOURCE)],
        output_values=["abi", "bin"],
        solc_version=SOLC_VERSION,
        optimize=True,
        optimize_runs=200,
    )
    key = next(k for k in compiled if k.endswith(":SigilRegistry"))
    out = {
        "abi": compiled[key]["abi"],
        "bytecode": "0x" + compiled[key]["bin"],
        "solc": SOLC_VERSION,
        "source_sha256": src_sha256,
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(out, indent=2))
    return out
