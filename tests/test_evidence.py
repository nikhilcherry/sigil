"""The evidence bundle is the trust anchor - these guard its two invariants:
serialisation is stable, and any change to content changes the hash."""

import json

import pytest

from sigil.evidence import Evidence, subject_ref


def test_canonical_json_is_key_order_independent(evidence):
    """A bundle reloaded from shuffled JSON must hash identically."""
    original = evidence.canonical_json()
    shuffled = json.loads(original)
    shuffled = {k: shuffled[k] for k in reversed(list(shuffled))}
    reparsed = Evidence.from_dict(shuffled)
    assert reparsed.canonical_json() == original


def test_hash_is_deterministic_across_instances(evidence, probe_ref, match_ref):
    twin = Evidence(
        probe=probe_ref,
        match=match_ref,
        similarity=0.7412,
        threshold=0.38,
        searched_at="2026-08-31T00:00:00Z",
        search_trace=[{"provider": "bluesky", "calls": []}],
    )
    assert twin.evidence_hash() == evidence.evidence_hash()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["match"].__setitem__("text", "a post with a photo!"),
        lambda d: d["match"].__setitem__("post_url", "https://bsky.app/other"),
        lambda d: d["match"].__setitem__("image_sha256", "00" * 32),
        lambda d: d["probe"].__setitem__("image_sha256", "11" * 32),
        lambda d: d.__setitem__("similarity", 0.7413),
        lambda d: d.__setitem__("searched_at", "2026-08-31T00:00:01Z"),
    ],
)
def test_any_mutation_changes_the_hash(evidence, mutate):
    """Tamper-evidence: there is no field a forger can quietly edit."""
    before = evidence.evidence_hash()
    d = json.loads(evidence.canonical_json())
    mutate(d)
    assert Evidence.from_dict(d).evidence_hash() != before


def test_similarity_rounding_absorbs_float_noise(probe_ref, match_ref):
    """Re-running inference can jitter the last bits; the hash must not move."""

    def make(sim):
        return Evidence(probe=probe_ref, match=match_ref, similarity=sim,
                        threshold=0.38, searched_at="2026-08-31T00:00:00Z")

    assert make(0.74120000001).evidence_hash() == make(0.7412).evidence_hash()
    assert make(0.7413).evidence_hash() != make(0.7412).evidence_hash()


def test_similarity_bps_is_clamped_to_uint32(probe_ref, match_ref):
    ev = Evidence(probe=probe_ref, match=match_ref, similarity=-0.5,
                  threshold=0.38, searched_at="2026-08-31T00:00:00Z")
    assert ev.similarity_bps() == 0


def test_subject_ref_is_salt_sensitive(probe_ref):
    a = subject_ref(probe_ref.embedding_sha256, "salt-a")
    b = subject_ref(probe_ref.embedding_sha256, "salt-b")
    assert a != b
    assert subject_ref(probe_ref.embedding_sha256, "salt-a") == a


def test_subject_ref_does_not_leak_the_embedding(probe_ref):
    """The commitment must not contain the digest it commits to."""
    ref = subject_ref(probe_ref.embedding_sha256, "salt")
    assert bytes.fromhex(probe_ref.embedding_sha256) not in ref
    assert len(ref) == 32
