"""The digest must be a property of the bundle, not of the code that wrote it.

Everywhere else the evidence hash comes out of `eth_utils.keccak` through
`sigil.evidence`, so a test that checks it with the same call is only checking
that sigil agrees with itself. `tools/verify_bundle.py` is a second
implementation - its own Keccak-256, its own serialisation, no sigil imports,
no third-party imports - and these tests are the two of them agreeing.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from eth_utils import keccak

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "verify_bundle.py"
sys.path.insert(0, str(TOOL.parent))

import verify_bundle  # noqa: E402

# ------------------------------------------------------- the primitive itself


@pytest.mark.parametrize("data", [
    b"",
    b"abc",
    b"the quick brown fox",
    b"\x00" * 64,
    "unicode: Beyoncé 平仮名 🙂".encode(),
])
def test_the_from_scratch_keccak_agrees_on_known_inputs(data):
    assert verify_bundle.keccak256(data) == keccak(data)


@pytest.mark.parametrize("size", [0, 1, 135, 136, 137, 271, 272, 273, 1000])
def test_it_agrees_across_the_block_boundary(size):
    """135/136/137 is where a padding or rate mistake shows up and nowhere else."""
    data = os.urandom(size)
    assert verify_bundle.keccak256(data) == keccak(data)


def test_it_is_keccak_and_not_sha3():
    """The two differ only in one padding byte, and give unrelated digests.

    Getting this wrong produces a tool that looks right, agrees with the SHA3
    in the standard library, and disagrees with every Ethereum contract.
    """
    import hashlib

    assert verify_bundle.keccak256(b"") != hashlib.sha3_256(b"").digest()
    assert verify_bundle.keccak256(b"").hex().startswith("c5d2460186f7")


# --------------------------------------------------------------- the bundle


def test_the_two_implementations_agree_on_a_real_bundle(evidence, tmp_path):
    path = tmp_path / "evidence.json"
    evidence.write(path)

    assert verify_bundle.evidence_hash(json.loads(path.read_text())) == \
        evidence.evidence_hash_hex()


def test_the_tool_prints_the_same_hash_from_the_command_line(evidence, tmp_path):
    """Run as a subprocess, so nothing this test imported can be helping it."""
    path = tmp_path / "evidence.json"
    evidence.write(path)

    out = subprocess.run([sys.executable, str(TOOL), str(path)],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == evidence.evidence_hash_hex()


def test_the_tool_confirms_a_matching_expectation(evidence, tmp_path):
    path = tmp_path / "evidence.json"
    evidence.write(path)

    out = subprocess.run(
        [sys.executable, str(TOOL), str(path), "--expect", evidence.evidence_hash_hex()],
        capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "MATCH"


def test_the_tool_reports_a_tampered_bundle(evidence, tmp_path):
    path = tmp_path / "evidence.json"
    evidence.write(path)
    anchored = evidence.evidence_hash_hex()

    data = json.loads(path.read_text())
    data["match"]["author_handle"] = "someone.else.bsky.social"
    path.write_text(json.dumps(data))

    out = subprocess.run([sys.executable, str(TOOL), str(path), "--expect", anchored],
                         capture_output=True, text=True)
    assert out.returncode == 1
    assert "MISMATCH" in out.stdout


def test_the_tool_refuses_a_bundle_carrying_fields_the_hash_does_not_cover(
        evidence, tmp_path):
    """The same refusal sigil makes, for the same reason.

    A file with an extra top-level key hashes identically to the file without
    it, so summarising it with one number would attest a subset of what its
    reader is looking at.
    """
    path = tmp_path / "evidence.json"
    data = json.loads(evidence.canonical_json())
    data["note"] = "trust me"
    path.write_text(json.dumps(data))

    out = subprocess.run([sys.executable, str(TOOL), str(path)],
                         capture_output=True, text=True)
    assert out.returncode == 2
    assert "not covered by the hash" in out.stderr


def test_the_tool_needs_nothing_installed():
    """Its whole value is that a third party can run it on a bare interpreter."""
    source = TOOL.read_text()
    imports = [line.strip() for line in source.splitlines()
               if line.startswith(("import ", "from ")) and "__future__" not in line]
    assert imports == ["import argparse", "import json", "import sys"]


def test_running_the_tool_does_not_load_sigil_at_all():
    """Checked by running it, not by reading it - prose mentions sigil freely."""
    probe = (
        "import runpy, sys;"
        f"sys.argv=['verify_bundle','{TOOL}','--expect','0x00'];"
        "  # the path is not a bundle, so it exits early - loading is the point\n"
        f"exec(open(r'{TOOL}').read().split(chr(10)+'if __name__')[0]);"
        "print([m for m in sys.modules if m.split('.')[0] in ('sigil','eth_utils')])"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                         text=True, check=True, cwd=str(ROOT))
    assert out.stdout.strip() == "[]"
