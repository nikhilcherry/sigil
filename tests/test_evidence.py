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


# -------------------------------------------------- the tamper demo's edit


def _bundle_dict(evidence):
    return json.loads(evidence.canonical_json())


def test_a_string_field_gains_a_character(evidence):
    from sigil.evidence import alter_field

    data = _bundle_dict(evidence)
    before, after = alter_field(data, "match.text")
    assert after == before + "!"
    assert data["match"]["text"] == after


def test_a_numeric_field_is_nudged(evidence):
    from sigil.evidence import alter_field

    data = _bundle_dict(evidence)
    before, after = alter_field(data, "similarity")
    assert after == pytest.approx(before + 0.0001)


def test_a_top_level_field_works_not_only_a_nested_one(evidence):
    from sigil.evidence import alter_field

    data = _bundle_dict(evidence)
    _, after = alter_field(data, "searched_at")
    assert data["searched_at"] == after


def test_an_explicit_value_overrides_the_automatic_edit(evidence):
    from sigil.evidence import alter_field

    data = _bundle_dict(evidence)
    before, after = alter_field(data, "match.platform", value="mastodon")
    assert (before, after) == ("bluesky", "mastodon")


def test_a_mistyped_field_names_the_ones_that_exist(evidence):
    """It used to be a KeyError traceback, on a command meant to be typed live."""
    from sigil.evidence import alter_field

    with pytest.raises(ValueError, match="no field 'match.nope'") as exc:
        alter_field(_bundle_dict(evidence), "match.nope")
    assert "match.text" in str(exc.value)


def test_descending_through_a_non_dict_is_refused_not_a_type_error(evidence):
    from sigil.evidence import alter_field

    with pytest.raises(ValueError, match="no field 'similarity.oops'"):
        alter_field(_bundle_dict(evidence), "similarity.oops")


def test_a_field_with_no_obvious_small_edit_asks_for_an_explicit_value(evidence):
    """probe.bbox is a list; adding 0.0001 to it raised a TypeError."""
    from sigil.evidence import alter_field

    with pytest.raises(ValueError, match="pass --value"):
        alter_field(_bundle_dict(evidence), "probe.bbox")
    # ...and giving one works.
    data = _bundle_dict(evidence)
    alter_field(data, "probe.bbox", value=[0, 0, 1, 1])
    assert data["probe"]["bbox"] == [0, 0, 1, 1]


def test_a_boolean_is_not_treated_as_a_number(evidence):
    from sigil.evidence import alter_field

    data = _bundle_dict(evidence)
    data["match"]["verified"] = True
    with pytest.raises(ValueError, match="pass --value"):
        alter_field(data, "match.verified")


def test_every_leaf_of_the_bundle_is_listed(evidence):
    from sigil.evidence import field_paths

    paths = field_paths(_bundle_dict(evidence))
    assert "match.text" in paths and "probe.backend" in paths
    assert "similarity" in paths and "schema" in paths
    assert "match" not in paths, "a container is not an editable leaf"


def test_altering_any_field_changes_the_hash(evidence):
    from sigil.evidence import Evidence, alter_field, field_paths

    data = _bundle_dict(evidence)
    original = evidence.evidence_hash_hex()
    for path in field_paths(data):
        fresh = _bundle_dict(evidence)
        try:
            alter_field(fresh, path)
        except ValueError:
            continue  # a list or bool needs an explicit value; covered above
        assert Evidence.from_dict(fresh).evidence_hash_hex() != original, path
