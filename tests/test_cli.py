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


# ---------------------------------------------------------------- calibrate


def test_calibrate_show_without_a_measurement_says_how_to_make_one(monkeypatch,
                                                                   tmp_path):
    import sigil.calibrate as cal

    monkeypatch.setattr(cal, "CALIBRATION_PATH", tmp_path / "absent.json")
    result = CliRunner().invoke(cli, ["calibrate", "--show"])
    assert result.exit_code != 0
    assert "sigil calibrate" in result.output


def test_calibrate_show_renders_a_saved_measurement(monkeypatch, tmp_path):
    import numpy as np

    import sigil.calibrate as cal
    from sigil.identify import Identity, IdentityIndex

    class Enc:
        name, model = "fake", "fake-model"

    index = IdentityIndex(
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        [Identity(name=n, qid=f"Q{i}", image_url="https://x", source="en.wikipedia")
         for i, n in enumerate(("A", "B"))],
        "fake",
    )
    by_qid = {"Qa": [np.array([1.0, 0.0]), np.array([0.9, 0.436])]}
    saved = tmp_path / "calibration.json"
    cal.measure(Enc(), index, by_qid, threshold=0.38, requested=1).save(saved)
    monkeypatch.setattr(cal, "CALIBRATION_PATH", saved)

    result = CliRunner().invoke(cli, ["calibrate", "--show"])
    assert result.exit_code == 0, result.output
    assert "Threshold calibration" in result.output
    assert "0.380" in result.output


def test_calibrate_passes_the_era_filter_through_and_zero_disables_it(monkeypatch):
    """`--born-after 0` has to mean "no filter", not "born after year zero"."""
    import sigil.calibrate as cal

    seen = {}

    def fake(encoder, threshold, limit, langs, born_after, on_progress):
        seen.update(threshold=threshold, limit=limit, langs=langs,
                    born_after=born_after)
        raise RuntimeError("stop here - the arguments are what is under test")

    monkeypatch.setattr(cal, "calibrate", fake)
    CliRunner().invoke(cli, ["calibrate", "--limit", "3", "--threshold", "0.5",
                             "--langs", "en,fr", "--born-after", "0"])
    assert seen["born_after"] is None
    assert seen["threshold"] == 0.5 and seen["limit"] == 3
    assert seen["langs"] == ("en", "fr")

    seen.clear()
    CliRunner().invoke(cli, ["calibrate", "--born-after", "1950"])
    assert seen["born_after"] == 1950

    seen.clear()
    CliRunner().invoke(cli, ["calibrate"])
    assert seen["born_after"] == cal.BORN_AFTER


def test_a_mistyped_tamper_field_is_an_error_message_not_a_traceback(evidence,
                                                                     tmp_path):
    """`sigil tamper` is typed by hand during a demo; a typo used to dump a stack."""
    path = tmp_path / "evidence.json"
    path.write_bytes(evidence.canonical_json())

    result = CliRunner().invoke(cli, ["tamper", "-e", str(path),
                                      "--field", "match.nope"])
    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "no field 'match.nope'" in result.output
    assert "match.text" in result.output, "the message should name real fields"


def test_a_tamper_field_that_is_a_list_asks_for_a_value(evidence, tmp_path):
    path = tmp_path / "evidence.json"
    path.write_bytes(evidence.canonical_json())

    result = CliRunner().invoke(cli, ["tamper", "-e", str(path),
                                      "--field", "probe.bbox"])
    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "pass --value" in result.output


def test_calibrate_show_exits_non_zero_when_nothing_is_measured(monkeypatch,
                                                                tmp_path):
    """demo.sh gates on this exit code, so a fresh clone must not fail the run."""
    import sigil.calibrate as cal

    monkeypatch.setattr(cal, "CALIBRATION_PATH", tmp_path / "absent.json")
    assert CliRunner().invoke(cli, ["calibrate", "--show"]).exit_code != 0


# ------------------------------------------------- anchor as its own stage


def test_anchor_records_an_existing_bundle_and_refuses_it_twice(evidence, cfg,
                                                                tmp_path):
    """A documented stage command, and the append-only claim in miniature.

    `sigil run` anchors as part of the pipeline; this is the path someone takes
    when the bundle already exists, and the second call is what makes the
    record tamper-evident rather than merely tamper-resistant.
    """
    path = tmp_path / "evidence.json"
    path.write_bytes(evidence.canonical_json())

    first = CliRunner().invoke(cli, ["anchor", "-e", str(path)])
    assert first.exit_code == 0, first.output
    assert "gas used" in first.output
    assert ChainClient(cfg).lookup(evidence.evidence_hash()) is not None

    second = CliRunner().invoke(cli, ["anchor", "-e", str(path)])
    assert second.exit_code == 0, second.output
    assert "already anchored" in second.output


def test_anchor_on_a_missing_bundle_is_a_message_not_a_traceback(tmp_path):
    result = CliRunner().invoke(cli, ["anchor", "-e", str(tmp_path / "none.json")])
    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "evidence bundle not found" in result.output


# ----------------------------------------------------------- index info


def test_index_info_exits_non_zero_without_an_index(monkeypatch, tmp_path):
    """scripts/demo.sh branches on this exit code.

    A fresh clone has no index, and the demo decides whether to show the
    identify path by asking this command - so the code is a contract, not an
    incidental detail.
    """
    import sigil.identify as ident

    monkeypatch.setattr(ident, "INDEX_VECTORS", tmp_path / "absent.npz")
    monkeypatch.setattr(ident, "INDEX_META", tmp_path / "absent.json")

    result = CliRunner().invoke(cli, ["index", "info"])
    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "sigil index build" in result.output


def test_index_info_reports_what_the_index_holds(monkeypatch, tmp_path):
    import json as _json

    import numpy as np

    import sigil.identify as ident

    vectors = tmp_path / "v.npz"
    meta = tmp_path / "m.json"
    np.savez(vectors, vectors=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    meta.write_text(_json.dumps({
        "backend": "insightface", "model": "buffalo_l/w600k_r50", "count": 2,
        "langs": ["en"], "months": 3,
        "identities": [
            {"name": "A", "qid": "Q1", "image_url": "https://x", "source": "en.wikipedia"},
            {"name": "B", "qid": "Q2", "image_url": "https://y", "source": "en.wikipedia"},
        ],
    }))
    monkeypatch.setattr(ident, "INDEX_VECTORS", vectors)
    monkeypatch.setattr(ident, "INDEX_META", meta)

    result = CliRunner().invoke(cli, ["index", "info"])
    assert result.exit_code == 0, result.output
    assert "2" in result.output
    assert "insightface" in result.output


# --------------------------------------------------------------- serve


def test_serve_binds_to_localhost_by_default(monkeypatch):
    """A safety default, and the last documented command with no test.

    This tool searches for people by face and has no authentication. The
    README promises it does not listen off-box unless asked; that promise is
    one default argument deep, so it gets an assertion rather than trust.
    """
    import sigil.web as web

    seen = {}
    monkeypatch.setattr(web, "serve", lambda **kw: seen.update(kw))

    result = CliRunner().invoke(cli, ["serve", "--no-browser"])
    assert result.exit_code == 0, result.output
    assert seen["host"] == "127.0.0.1"
    assert seen["port"] == 8099
    assert seen["open_browser"] is False


def test_serve_passes_an_explicit_bind_through(monkeypatch):
    """Reachable from off-box is available, but only by asking for it."""
    import sigil.web as web

    seen = {}
    monkeypatch.setattr(web, "serve", lambda **kw: seen.update(kw))

    result = CliRunner().invoke(
        cli, ["serve", "--host", "0.0.0.0", "--port", "9000"]  # noqa: S104
    )
    assert result.exit_code == 0, result.output
    assert seen["host"] == "0.0.0.0"  # noqa: S104
    assert seen["port"] == 9000
    assert seen["open_browser"] is True
