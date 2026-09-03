"""Contract compilation and its artifact cache.

The cache exists so repeat runs skip solc. The property that matters is the
one that makes caching safe: an edit to the contract must never be shadowed by
a stale artifact, because the deployed bytecode would then silently disagree
with the source in the repository.
"""

import json

import pytest

from sigil.chain import compile as compile_mod


class FakeSolcx:
    """Records whether solc was invoked at all."""

    def __init__(self):
        self.compiles = 0

    def get_installed_solc_versions(self):
        return [compile_mod.SOLC_VERSION]

    def install_solc(self, version):  # pragma: no cover - never needed here
        raise AssertionError("should not install solc in a test")

    def set_solc_version(self, version):
        pass

    def compile_files(self, files, **kwargs):
        self.compiles += 1
        return {f"{files[0]}:SigilRegistry": {"abi": [{"name": "anchor"}], "bin": "6080"}}


@pytest.fixture
def wired(tmp_path, monkeypatch):
    source = tmp_path / "SigilRegistry.sol"
    source.write_text("// contract v1")
    artifact = tmp_path / "SigilRegistry.json"
    solcx = FakeSolcx()

    monkeypatch.setattr(compile_mod, "SOURCE", source)
    monkeypatch.setattr(compile_mod, "ARTIFACT", artifact)
    monkeypatch.setattr(compile_mod, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(compile_mod, "solcx", solcx)
    return source, artifact, solcx


def test_a_first_compile_writes_the_artifact(wired):
    source, artifact, solcx = wired

    out = compile_mod.compile_registry()

    assert solcx.compiles == 1
    assert out["bytecode"] == "0x6080"
    assert out["solc"] == compile_mod.SOLC_VERSION
    assert json.loads(artifact.read_text())["bytecode"] == "0x6080"


def test_a_second_call_uses_the_cache_and_never_runs_solc(wired):
    source, artifact, solcx = wired
    compile_mod.compile_registry()

    out = compile_mod.compile_registry()

    assert solcx.compiles == 1, "recompiled despite an up-to-date artifact"
    assert out["bytecode"] == "0x6080"


def test_editing_the_contract_invalidates_the_cache(wired):
    """The property that makes caching safe at all: deployed bytecode must not
    be allowed to silently disagree with the source in the repository."""
    source, artifact, solcx = wired
    compile_mod.compile_registry()

    source.write_text("// contract v2")
    import os
    stat = source.stat()
    os.utime(source, (stat.st_atime, stat.st_mtime + 10))

    compile_mod.compile_registry()

    assert solcx.compiles == 2, "a stale artifact shadowed an edited contract"


def test_force_recompiles_even_with_a_fresh_artifact(wired):
    source, artifact, solcx = wired
    compile_mod.compile_registry()

    compile_mod.compile_registry(force=True)

    assert solcx.compiles == 2


@pytest.mark.parametrize("corrupt", ["{not json", "[]", ""])
def test_a_corrupt_artifact_recompiles_rather_than_crashing(wired, corrupt):
    """The artifact caches a deterministic build, so an unreadable one means
    recompile - the contract source is the truth, and it is right there."""
    source, artifact, solcx = wired
    compile_mod.compile_registry()
    artifact.write_text(corrupt)

    out = compile_mod.compile_registry()

    assert solcx.compiles == 2
    assert out["bytecode"] == "0x6080"
    assert json.loads(artifact.read_text())["bytecode"] == "0x6080"


# ------------------------------------------------- what keys the build cache


def test_an_edit_that_preserves_the_timestamp_still_recompiles(monkeypatch,
                                                               tmp_path):
    """The cache used to key on mtime, which is not the content.

    Two edits inside one filesystem tick, or a restore that preserves the
    timestamp, leave an mtime unchanged while the source is different. A stale
    hit here means deploying bytecode that does not correspond to the contract
    in the repository - and this project's whole claim is that a reader can
    check its records against the source. It is also the exact failure mode
    Python's own mtime-keyed bytecode cache produced during this work.
    """
    import sigil.chain.compile as comp

    source = tmp_path / "SigilRegistry.sol"
    artifact = tmp_path / "SigilRegistry.json"
    monkeypatch.setattr(comp, "SOURCE", source)
    monkeypatch.setattr(comp, "ARTIFACT", artifact)
    monkeypatch.setattr(comp, "ARTIFACTS_DIR", tmp_path)

    calls = []

    def fake_compile(files, **kw):
        calls.append(source.read_bytes())
        return {f"{source}:SigilRegistry": {"abi": [], "bin": "60" + str(len(calls))}}

    monkeypatch.setattr(comp.solcx, "compile_files", fake_compile)
    monkeypatch.setattr(comp, "_ensure_solc", lambda: None)

    source.write_bytes(b"contract SigilRegistry { }")
    first = comp.compile_registry()
    assert len(calls) == 1

    # Same content: a cache hit, no recompile.
    assert comp.compile_registry() == first
    assert len(calls) == 1

    # Different content, timestamp forced back to what it was.
    stat = source.stat()
    source.write_bytes(b"contract SigilRegistry { uint256 x; }")
    import os

    os.utime(source, (stat.st_atime, stat.st_mtime))
    assert source.stat().st_mtime == stat.st_mtime, "the mtime did not stay put"

    second = comp.compile_registry()
    assert len(calls) == 2, "a changed contract reused a stale artifact"
    assert second["bytecode"] != first["bytecode"]


def test_the_artifact_records_the_source_hash_it_was_built_from(monkeypatch,
                                                               tmp_path):
    """So a reader can tell which contract the cached bytecode belongs to."""
    import hashlib
    import json as _json

    import sigil.chain.compile as comp

    source = tmp_path / "SigilRegistry.sol"
    artifact = tmp_path / "SigilRegistry.json"
    monkeypatch.setattr(comp, "SOURCE", source)
    monkeypatch.setattr(comp, "ARTIFACT", artifact)
    monkeypatch.setattr(comp, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(comp, "_ensure_solc", lambda: None)
    monkeypatch.setattr(comp.solcx, "compile_files", lambda files, **kw: {
        f"{source}:SigilRegistry": {"abi": [], "bin": "6001"}})

    body = b"contract SigilRegistry { }"
    source.write_bytes(body)
    comp.compile_registry()

    written = _json.loads(artifact.read_text())
    assert written["source_sha256"] == hashlib.sha256(body).hexdigest()
    assert written["solc"] == comp.SOLC_VERSION
