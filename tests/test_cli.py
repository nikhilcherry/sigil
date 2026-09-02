"""Command-line surface, exercised through click's runner."""

import json

import pytest
from click.testing import CliRunner

from sigil.chain import ChainClient
from sigil.cli import cli
from sigil.evidence import Evidence
from sigil.search.base import Candidate, ProviderTrace
from tests.conftest import EXAMPLE_PROBE


def test_chain_address_reports_without_deploying(cfg, monkeypatch):
    """`chain info` deploys the registry before it can report, which fails on an
    unfunded key - so there has to be a way to find out which address to fund
    without first trying to spend from it."""
    monkeypatch.setenv("SIGIL_CHAIN", "local")
    result = CliRunner().invoke(cli, ["chain", "address"])

    assert result.exit_code == 0, result.output
    assert ChainClient(cfg).address in result.output.replace("\n", "")
    assert ChainClient(cfg).deployed_address() is None, "reporting deployed a contract"


def test_chain_info_does_deploy(cfg, monkeypatch):
    """The contrast that makes the split worth having."""
    monkeypatch.setenv("SIGIL_CHAIN", "local")
    result = CliRunner().invoke(cli, ["chain", "info"])

    assert result.exit_code == 0, result.output
    assert ChainClient(cfg).deployed_address() is not None


def test_backends_reports_the_provider_in_use():
    result = CliRunner().invoke(cli, ["backends"])

    assert result.exit_code == 0, result.output
    assert "runs on" in result.output
    assert "CPU" in result.output or "CUDA" in result.output


# ------------------------------------- the sequence the demo recording follows


class StubProvider:
    name = "stub"

    def __init__(self):
        self.trace = ProviderTrace(provider=self.name)

    def candidates(self, query):
        self.trace.record("stub.search", {"q": query}, 1)
        yield Candidate(
            platform="bluesky", image_url="https://x/same.jpg",
            post_url="https://bsky.app/profile/who.bsky.social/post/1",
            post_uri="at://did:plc:who/app.bsky.feed.post/1",
            author_handle="who.bsky.social", author_did="did:plc:who",
            author_display_name="Who", text="a photo",
            created_at="2026-08-01T00:00:00Z", discovered_via="stub",
        )


@pytest.fixture
def offline(monkeypatch, tmp_path):
    """The whole pipeline, no network, writing into a throwaway directory."""
    import sigil.chain.client as client_mod
    import sigil.pipeline as pipe
    import sigil.search.matcher as matcher

    monkeypatch.setattr(client_mod, "STATE_PATH", tmp_path / "chain-state.json")
    monkeypatch.setattr(
        matcher, "fetch_image",
        lambda s, u, t: EXAMPLE_PROBE.read_bytes() if "same" in u else None,
    )
    monkeypatch.setattr(pipe, "build_providers", lambda cfg, url, blob=None: [StubProvider()])
    monkeypatch.setenv("SIGIL_CHAIN", "local")
    monkeypatch.setenv("SIGIL_SUBJECT_SALT", "test-salt")
    return tmp_path


def test_the_demo_sequence_runs_end_to_end(offline):
    """scan -> run -> verify -> tamper -> verify(fails), as scripts/demo.sh does.

    There are no resubmissions, so the recorded sequence is the deliverable.
    """
    runner = CliRunner()
    evidence = offline / "evidence.json"
    tampered = offline / "evidence.tampered.json"

    scan = runner.invoke(cli, ["scan", str(EXAMPLE_PROBE)])
    assert scan.exit_code == 0, scan.output
    assert "Face encoded" in scan.output

    run = runner.invoke(cli, ["run", str(EXAMPLE_PROBE), "-q", "who",
                              "-o", str(evidence)])
    assert run.exit_code == 0, run.output
    assert "VERIFIED" in run.output
    assert evidence.exists()

    verify = runner.invoke(cli, ["verify", "-e", str(evidence),
                                 "--probe", str(EXAMPLE_PROBE)])
    assert verify.exit_code == 0, verify.output
    assert "VERIFIED" in verify.output

    tamper = runner.invoke(cli, ["tamper", "-e", str(evidence),
                                 "--field", "match.text", "-o", str(tampered)])
    assert tamper.exit_code == 0, tamper.output
    assert tampered.exists()

    # The point of the whole exercise: the altered bundle must be rejected.
    rejected = runner.invoke(cli, ["verify", "-e", str(tampered)])
    assert rejected.exit_code == 1, rejected.output
    assert "NOT VERIFIED" in rejected.output


def test_no_match_exits_two_and_anchors_nothing(offline, monkeypatch):
    """A distinct exit code is what lets the run compose in a script."""
    import sigil.search.matcher as matcher

    monkeypatch.setattr(matcher, "fetch_image", lambda s, u, t: None)

    result = CliRunner().invoke(cli, ["run", str(EXAMPLE_PROBE), "-q", "nobody"])

    assert result.exit_code == 2, result.output
    assert "No match" in result.output


def test_tamper_changes_the_hash_it_reports(offline):
    """The demo's claim is visible in the two hashes it prints side by side."""
    runner = CliRunner()
    evidence = offline / "evidence.json"
    tampered = offline / "evidence.tampered.json"

    runner.invoke(cli, ["run", str(EXAMPLE_PROBE), "-q", "who", "-o", str(evidence)])
    result = runner.invoke(cli, ["tamper", "-e", str(evidence), "-o", str(tampered)])

    before = Evidence.from_dict(json.loads(evidence.read_text())).evidence_hash_hex()
    after = Evidence.from_dict(json.loads(tampered.read_text())).evidence_hash_hex()
    assert before != after
    assert before in result.output.replace("\n", "")


def test_search_stops_before_the_chain(offline):
    """Stages 1-3 only: a bundle, but nothing anchored."""
    from sigil.chain import ChainClient
    from sigil.config import Config

    evidence = offline / "evidence.json"
    result = CliRunner().invoke(cli, ["search", str(EXAMPLE_PROBE), "-q", "who",
                                      "-o", str(evidence)])

    assert result.exit_code == 0, result.output
    assert evidence.exists()
    assert "Anchored on chain" not in result.output
    cfg = Config()
    cfg.chain_backend = "local"
    assert ChainClient(cfg).total_anchored() == 0


def test_a_missing_probe_is_a_clean_error_not_a_traceback(offline):
    result = CliRunner().invoke(cli, ["scan", "/no/such/file.jpg"])

    assert result.exit_code != 0
    assert "not found" in result.output
    assert "Traceback" not in result.output


def test_a_chain_failure_is_a_message_not_a_traceback(monkeypatch, tmp_path):
    """These failures are configuration, not bugs - an unfunded key, an
    unreachable endpoint, a missing key. Someone following the testnet
    instructions should be told what to fix, not shown a stack trace."""
    import sigil.cli as cli_mod

    def explode(cfg):
        raise RuntimeError("0xabc has no funds on chain 80002. Fund it from a faucet")

    monkeypatch.setattr(cli_mod, "ChainClient", explode)

    for argv in (["chain", "info"], ["chain", "address"]):
        result = CliRunner().invoke(cli, argv)
        assert result.exit_code != 0, argv
        assert "Traceback" not in result.output, argv
        assert "Fund it from a faucet" in result.output, argv


def test_a_missing_rpc_key_is_reported_plainly(monkeypatch):
    monkeypatch.setenv("SIGIL_CHAIN", "rpc")
    monkeypatch.setenv("SIGIL_RPC_URL", "https://example.invalid")
    monkeypatch.delenv("SIGIL_PRIVATE_KEY", raising=False)

    result = CliRunner().invoke(cli, ["chain", "address"])

    assert result.exit_code != 0
    assert "SIGIL_PRIVATE_KEY is required" in result.output
    assert "Traceback" not in result.output


def test_chain_reset_removes_state_and_says_so(offline, monkeypatch):
    """demo.sh opens with this, so a recording starts from a known chain."""
    import sigil.config as config_mod

    state = offline / "chain-state.json"
    state.write_text("{}")
    monkeypatch.setattr(config_mod, "STATE_PATH", state)

    result = CliRunner().invoke(cli, ["chain", "reset", "--yes"])

    assert result.exit_code == 0, result.output
    assert not state.exists()
    assert "removed" in result.output


def test_chain_reset_on_a_clean_slate_is_not_an_error(offline, monkeypatch):
    import sigil.config as config_mod

    monkeypatch.setattr(config_mod, "STATE_PATH", offline / "absent.json")

    result = CliRunner().invoke(cli, ["chain", "reset", "--yes"])

    assert result.exit_code == 0, result.output
    assert "no local chain state" in result.output


def test_identify_without_an_index_says_how_to_build_one(offline, monkeypatch):
    """demo.sh runs this before any index exists on a fresh clone."""
    import sigil.identify as idmod

    monkeypatch.setattr(idmod, "INDEX_VECTORS", offline / "none.npz")
    monkeypatch.setattr(idmod, "INDEX_META", offline / "none.json")

    result = CliRunner().invoke(cli, ["identify", str(EXAMPLE_PROBE)])

    assert result.exit_code != 0
    assert "sigil index build" in result.output
    assert "Traceback" not in result.output


def test_identify_names_a_face_from_the_index(offline, monkeypatch):
    """The naming step on its own, as the demo shows it before the full run."""
    import numpy as np

    import sigil.identify as idmod
    from sigil.config import Config
    from sigil.pipeline import scan_probe

    face, _, encoder = scan_probe(EXAMPLE_PROBE.read_bytes(), Config())
    vec = np.asarray(face.embedding, dtype=np.float32).reshape(1, -1)

    vec_path, meta_path = offline / "v.npz", offline / "m.json"
    np.savez_compressed(vec_path, vectors=vec)
    meta_path.write_text(json.dumps({
        "backend": encoder.name, "model": encoder.model, "count": 1,
        "identities": [{"name": "A Known Person", "qid": "Q1",
                        "image_url": "https://x/p.jpg", "source": "en.wikipedia"}],
    }))
    monkeypatch.setattr(idmod, "INDEX_VECTORS", vec_path)
    monkeypatch.setattr(idmod, "INDEX_META", meta_path)

    result = CliRunner().invoke(cli, ["identify", str(EXAMPLE_PROBE)])

    assert result.exit_code == 0, result.output
    assert "A Known Person" in result.output


def test_index_build_passes_its_options_through(monkeypatch):
    """This command runs for the better part of an hour. A wiring bug in it is
    only discovered after that hour has been spent."""
    import sigil.identify as idmod

    seen = {}

    def fake_build(encoder, langs, months, limit, on_progress):
        seen.update(langs=langs, months=months, limit=limit)
        on_progress("harvesting")
        return 42

    monkeypatch.setattr(idmod, "build_index", fake_build)

    result = CliRunner().invoke(cli, ["index", "build", "--langs", "en, fr ,ta",
                                      "--months", "2", "--limit", "500"])

    assert result.exit_code == 0, result.output
    assert seen["langs"] == ("en", "fr", "ta"), "whitespace was not stripped"
    assert seen["months"] == 2
    assert seen["limit"] == 500
    assert "harvesting" in result.output, "progress was not surfaced"
    assert "indexed 42 faces" in result.output


def test_index_build_defaults_to_the_ten_language_spread(monkeypatch):
    """The non-English editions are the reason the index covers non-Western
    figures at all, so the default must not quietly become English-only."""
    import sigil.identify as idmod

    seen = {}

    def fake_build(encoder, langs, months, limit, on_progress):
        seen["langs"] = langs
        return 0

    monkeypatch.setattr(idmod, "build_index", fake_build)

    result = CliRunner().invoke(cli, ["index", "build"])

    assert result.exit_code == 0, result.output
    assert seen["langs"] == idmod.DEFAULT_LANGS
    assert len(seen["langs"]) == 10
    assert "hi" in seen["langs"] and "ta" in seen["langs"]
