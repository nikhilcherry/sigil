"""Chain behaviour: append-only records, cross-process persistence, and a
verification path that fails loudly rather than quietly."""

import json

import pytest

from sigil.chain import ChainClient
from sigil.evidence import Evidence


@pytest.fixture
def client(cfg):
    return ChainClient(cfg)


def test_deploy_is_idempotent(client):
    first = client.ensure_deployed()
    assert first == client.ensure_deployed()
    assert client.total_anchored() == 0


def test_anchor_then_verify(client, evidence):
    receipt = client.anchor(evidence)
    assert receipt["already_anchored"] is False
    assert receipt["evidence_hash"] == evidence.evidence_hash_hex()

    v = client.verify(evidence)
    assert v.ok
    assert v.on_chain["similarity_bps"] == evidence.similarity_bps()
    assert v.on_chain["submitter"] == client.address


def test_second_anchor_of_same_bundle_is_rejected_not_overwritten(client, evidence):
    """Append-only is the whole guarantee: the first timestamp must survive."""
    first = client.anchor(evidence)
    original_ts = client.verify(evidence).on_chain["anchored_at"]

    second = client.anchor(evidence)
    assert second["already_anchored"] is True
    assert "tx_hash" not in second
    assert client.verify(evidence).on_chain["anchored_at"] == original_ts
    assert client.total_anchored() == 1
    assert first["evidence_hash"] == second["evidence_hash"]


def test_tampered_bundle_does_not_verify(client, evidence):
    client.anchor(evidence)
    d = json.loads(evidence.canonical_json())
    d["match"]["text"] = "something else entirely"
    v = client.verify(Evidence.from_dict(d))
    assert not v.anchored
    assert not v.ok
    assert v.notes


def test_unanchored_bundle_reports_missing_rather_than_raising(client, evidence):
    v = client.verify(evidence)
    assert v.anchored is False
    assert v.ok is False


def test_wrong_probe_fails_even_though_hash_matches(client, evidence):
    """A bundle can be authentic while the probe supplied alongside it is not."""
    client.anchor(evidence)
    v = client.verify(evidence, probe_embedding_sha256="99" * 32)
    assert v.anchored is True
    assert v.probe_matches is False
    assert v.ok is False


def test_salt_change_breaks_the_subject_commitment(cfg, evidence):
    ChainClient(cfg).anchor(evidence)
    cfg.subject_salt = "a-different-salt"
    v = ChainClient(cfg).verify(evidence)
    assert v.anchored is True
    assert v.subject_matches is False
    assert v.ok is False


def test_state_survives_a_new_client(cfg, evidence):
    """`sigil anchor` and `sigil verify` are separate processes; state must persist."""
    writer = ChainClient(cfg)
    writer.anchor(evidence)
    address = writer.deployed_address()

    reader = ChainClient(cfg)
    assert reader.deployed_address() == address
    assert reader.total_anchored() == 1
    assert reader.verify(evidence).ok


def test_optional_checks_that_did_not_run_do_not_count_as_passes(client, evidence):
    client.anchor(evidence)
    v = client.verify(evidence)
    assert v.probe_matches is None
    assert v.source_image_intact is None
    assert v.ok is True


# ------------------------------------------------------- the rpc funding path


class _FakeFn:
    def build_transaction(self, tx):
        return dict(tx)


class _FakeEth:
    """Just enough of w3.eth for _send to get as far as pricing the transaction."""

    chain_id = 80002

    def __init__(self, error):
        self.error = error

    def get_transaction_count(self, address):
        return 0

    def estimate_gas(self, tx):
        raise self.error


def _rpc_client(cfg, monkeypatch, error):
    client = ChainClient(cfg)
    monkeypatch.setattr(client, "backend", "rpc")
    monkeypatch.setattr(client, "account", "0x" + "a" * 40)
    monkeypatch.setattr(client.w3, "eth", _FakeEth(error))
    return client


def test_an_unfunded_key_gets_the_instruction_that_fixes_it(cfg, monkeypatch):
    """Most nodes refuse to price a transaction the sender cannot pay for, so
    the failure lands at gas estimation rather than at send - where a raw web3
    error tells nobody what to do about it."""
    client = _rpc_client(cfg, monkeypatch, ValueError(
        "err: insufficient funds for gas * price + value: have 0 want 12345"
    ))

    with pytest.raises(RuntimeError) as exc:
        client._send(_FakeFn())

    assert "faucet" in str(exc.value)
    assert "sigil chain address" in str(exc.value)


def test_a_gas_requirement_failure_is_also_a_funding_problem(cfg, monkeypatch):
    """Some nodes phrase the same condition as a gas requirement instead."""
    client = _rpc_client(cfg, monkeypatch, ValueError(
        "gas required exceeds allowance (0)"
    ))

    with pytest.raises(RuntimeError, match="faucet"):
        client._send(_FakeFn())


def test_an_unrelated_rpc_error_is_not_disguised_as_a_funding_problem(cfg, monkeypatch):
    """Swallowing every failure into "go find a faucet" would send someone
    chasing the wrong thing."""
    client = _rpc_client(cfg, monkeypatch, ValueError(
        "execution reverted: already anchored"
    ))

    with pytest.raises(ValueError, match="already anchored"):
        client._send(_FakeFn())


# ------------------------------------------------ re-checking the source image


def _anchored_with_image(client, evidence, blob):
    """Anchor a bundle whose recorded image digest matches `blob`."""
    from sigil.evidence import sha256_hex

    evidence.match.image_sha256 = sha256_hex(blob)
    client.anchor(evidence)
    return evidence


def test_recheck_source_passes_when_the_post_image_is_unchanged(client, evidence,
                                                                monkeypatch):
    """The chain proves the bundle did not change; this proves the world did not
    change underneath it."""
    import sigil.search.http as http

    blob = b"the original post image bytes"
    ev = _anchored_with_image(client, evidence, blob)
    monkeypatch.setattr(http, "fetch_image", lambda s, u, t: blob)

    v = client.verify(ev, recheck_source=True)

    assert v.source_image_intact is True
    assert v.ok


def test_recheck_source_fails_when_the_image_bytes_changed(client, evidence,
                                                           monkeypatch):
    """An edited or swapped image must show up rather than passing silently."""
    import sigil.search.http as http

    ev = _anchored_with_image(client, evidence, b"the original post image bytes")
    monkeypatch.setattr(http, "fetch_image", lambda s, u, t: b"different bytes now")

    v = client.verify(ev, recheck_source=True)

    assert v.source_image_intact is False
    assert not v.ok
    assert any("bytes changed" in n for n in v.notes)


def test_recheck_source_fails_when_the_post_is_gone(client, evidence, monkeypatch):
    """A deleted or protected post is a different failure from an edited one,
    and says so."""
    import sigil.search.http as http

    ev = _anchored_with_image(client, evidence, b"the original post image bytes")
    monkeypatch.setattr(http, "fetch_image", lambda s, u, t: None)

    v = client.verify(ev, recheck_source=True)

    assert v.source_image_intact is False
    assert any("no longer retrievable" in n for n in v.notes)


def test_a_check_that_was_not_requested_reads_as_none_not_as_a_pass(client, evidence):
    """An optional check nobody ran must never be reported as having passed."""
    client.anchor(evidence)

    v = client.verify(evidence)

    assert v.source_image_intact is None
    assert v.probe_matches is None
    assert v.ok, "unrequested checks must not drag the verdict down either"


# --------------------------------------------------------------- rpc wiring


def test_rpc_without_an_endpoint_says_which_variable_is_missing(cfg):
    cfg.chain_backend = "rpc"
    cfg.rpc_url = None

    with pytest.raises(RuntimeError, match="SIGIL_RPC_URL"):
        ChainClient(cfg)


def test_rpc_without_a_key_says_which_variable_is_missing(cfg):
    cfg.chain_backend = "rpc"
    cfg.rpc_url = "https://node.example"
    cfg.private_key = None

    with pytest.raises(RuntimeError, match="SIGIL_PRIVATE_KEY"):
        ChainClient(cfg)


def test_an_unreachable_endpoint_names_the_endpoint(cfg, monkeypatch):
    """The usual cause is a dead public RPC, so the message has to say which."""
    import sigil.chain.client as client_mod

    cfg.chain_backend = "rpc"
    cfg.rpc_url = "https://node.example"
    cfg.private_key = "0x" + "11" * 32

    class DeadWeb3:
        def __init__(self, provider):
            self.middleware_onion = type("M", (), {"inject": lambda *a, **k: None})()

        @staticmethod
        def HTTPProvider(url, request_kwargs=None):  # noqa: N802
            return object()

        def is_connected(self):
            return False

    monkeypatch.setattr(client_mod, "Web3", DeadWeb3)

    with pytest.raises(RuntimeError, match="node.example"):
        ChainClient(cfg)
