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
