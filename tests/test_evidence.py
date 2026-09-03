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


def test_a_negative_similarity_is_recorded_as_zero_on_chain(evidence):
    """The registry field is unsigned; the bundle keeps the real number."""
    evidence.similarity = -0.5
    assert evidence.similarity_bps() == 0
    assert evidence.to_dict()["similarity"] == -0.5


def test_two_different_negative_similarities_still_change_the_hash(evidence):
    """They collapse to the same basis points, so the hash has to carry them."""
    from sigil.evidence import Evidence

    a = Evidence.from_dict(evidence.to_dict())
    b = Evidence.from_dict(evidence.to_dict())
    a.similarity, b.similarity = -0.5, -0.9
    assert a.similarity_bps() == b.similarity_bps() == 0
    assert a.evidence_hash_hex() != b.evidence_hash_hex()


def test_similarity_basis_points_span_the_whole_cosine_range(evidence):
    evidence.similarity = 1.0
    assert evidence.similarity_bps() == 10000
    evidence.similarity = 0.7596
    assert evidence.similarity_bps() == 7596


def test_a_typo_in_the_first_path_segment_names_the_fields_too(evidence):
    """A different error path from a typo'd leaf, and a different message.

    `--field noshuch.text` fails while walking to the leaf rather than at it,
    which is the branch a typo in the first segment takes.
    """
    from sigil.evidence import alter_field

    with pytest.raises(ValueError, match="no field 'nosuch'") as exc:
        alter_field(_bundle_dict(evidence), "nosuch.text")
    assert "match.text" in str(exc.value)


def test_a_deep_path_that_runs_out_partway_names_where_it_stopped(evidence):
    from sigil.evidence import alter_field

    with pytest.raises(ValueError, match="no field 'match.nope'"):
        alter_field(_bundle_dict(evidence), "match.nope.deeper")


def test_the_subject_commitment_is_scoped_to_the_provider_not_just_the_face(
    probe_ref,
):
    """What "anyone can recompute it" actually requires.

    The commitment is a function of the embedding digest, and that digest
    differs between CPU and CUDA for one image - the embeddings agree to
    0.9996 and sha256 has no notion of close. So two runs of the same person
    on different providers produce unlinkable commitments, and a third party
    recomputing one needs the provider as well as the photograph.

    Asserted so the scope is a property of the design rather than a surprise:
    if someone later makes the commitment provider-stable, this test is where
    they will come to say so deliberately.
    """
    from sigil.evidence import subject_ref

    on_cpu = "33b13452bd764bcbf76561579c2f1fe52977e8d0c5e20271800230dfcda49e84"
    on_cuda = "9324dd67708b66c8e9db0659c4e57d1217a52a886879b72e8887f85a5" + "0" * 7
    assert on_cpu != on_cuda

    salt = "sigil-default-salt"
    assert subject_ref(on_cpu, salt) != subject_ref(on_cuda, salt)
    # And it is a pure function of (digest, salt), so the same pair always
    # recomputes - which is what makes it checkable at all.
    assert subject_ref(on_cpu, salt) == subject_ref(on_cpu, salt)


# ------------------------------------------------------------ golden vector

# A fully-specified bundle and the exact bytes and hash it must produce. Every
# claim this project makes rests on that hash being reproducible, and nothing
# was pinning it - three schema changes in one afternoon altered every hash the
# project can produce and no test noticed.
#
# This is deliberately brittle. If it fails, one of two things happened: the
# canonical serialisation drifted, which is a bug, or a field was added or
# renamed, which is a schema change - and then SCHEMA in sigil/__init__.py has
# to move too, and this vector is updated in the same commit that moves it.
# Failing loudly is the whole point.
GOLDEN_SCHEMA = "sigil/evidence/v3"
GOLDEN_BYTES = 1140
GOLDEN_HASH = "0xc9a63a787a43bb17c9cc19630a6ad6b9a69a4cc6b2416e9427bcf691e095011f"
GOLDEN_SUBJECT = "0xe6a4852f797e1005246089aba3e37c853555f5adcdcfd8fdd5b767a6dee20276"


def _golden_bundle():
    from sigil.evidence import MatchRef, ProbeRef

    return Evidence(
        probe=ProbeRef(
            image_sha256="ab" * 32, embedding_sha256="cd" * 32,
            backend="insightface", model="buffalo_l/w600k_r50",
            bbox=[10, 20, 110, 140], det_score=0.9312,
            provider="CPUExecutionProvider"),
        match=MatchRef(
            platform="bluesky",
            post_url="https://bsky.app/profile/who.bsky.social/post/3abc",
            post_uri="at://did:plc:example/app.bsky.feed.post/3abc",
            author_handle="who.bsky.social", author_did="did:plc:example",
            author_display_name="Someone", text="a post with a photo",
            image_url="https://cdn.bsky.app/img/feed_fullsize/plain/x/y",
            image_sha256="ef" * 32, created_at="2026-08-01T12:00:00Z",
            discovered_via="app.bsky.actor.searchActors:avatar",
            probe_photo_similarity=0.021300, claim="identity",
            source_kind="social"),
        similarity=0.759554, threshold=0.38,
        searched_at="2026-09-03T00:00:00Z",
        search_trace=[{"provider": "bluesky", "calls": [
            {"endpoint": "app.bsky.actor.searchActors",
             "params": {"q": "AOC", "limit": 25}, "results": 25}]}],
    )


def test_a_known_bundle_hashes_to_a_known_value():
    """The project's central invariant, pinned.

    If this fails: either the canonical serialisation drifted, which is a bug,
    or a field changed, which is a schema change - and then SCHEMA moves in the
    same commit as this vector.
    """
    ev = _golden_bundle()
    assert ev.schema == GOLDEN_SCHEMA
    assert len(ev.canonical_json()) == GOLDEN_BYTES
    assert ev.evidence_hash_hex() == GOLDEN_HASH


def test_the_golden_bundle_survives_a_round_trip_through_json():
    """Reading a bundle back must not change what it hashes to."""
    ev = _golden_bundle()
    back = Evidence.from_dict(json.loads(ev.canonical_json()))
    assert back.canonical_json() == ev.canonical_json()
    assert back.evidence_hash_hex() == GOLDEN_HASH


def test_the_golden_bundle_commits_to_a_known_subject():
    """The value that goes on chain, for the default salt."""
    assert "0x" + subject_ref(
        _golden_bundle().probe.embedding_sha256, "sigil-default-salt"
    ).hex() == GOLDEN_SUBJECT


def test_the_canonical_bytes_are_sorted_compact_and_utf8():
    """The three properties the serialisation pins, checked directly."""
    raw = _golden_bundle().canonical_json()
    assert isinstance(raw, bytes)
    assert b", " not in raw and b": " not in raw, "not compact"
    text = raw.decode("utf-8")
    keys = ("match", "probe", "schema", "searched_at", "similarity")
    positions = [text.index(f'"{k}"') for k in keys]
    assert positions == sorted(positions), "top-level keys are not sorted"


# --------------------------------------------------- writing it out durably


def test_the_bundle_is_written_atomically(evidence, tmp_path):
    """It is the one file here that cannot be regenerated.

    Chain state is written atomically and can be rebuilt by re-running. This
    file cannot: its hash is anchored, and the search that produced it was
    live, so a second run returns different candidates and a different
    timestamp. A bundle truncated by a crash is a permanent orphan - a record
    on chain that nothing can ever verify again.
    """
    path = tmp_path / "evidence.json"
    evidence.write(path)

    assert path.read_bytes() == evidence.canonical_json()
    assert not (tmp_path / "evidence.json.tmp").exists(), "the temp file was left behind"


def test_a_failed_write_leaves_the_previous_bundle_intact(evidence, tmp_path,
                                                          monkeypatch):
    """The property atomicity buys: no half-written file under the real name."""
    path = tmp_path / "evidence.json"
    evidence.write(path)
    original = path.read_bytes()

    def explode(*_a, **_kw):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("sigil.evidence.os.fsync", explode)

    altered = Evidence.from_dict({**evidence.to_dict(), "searched_at": "later"})
    with pytest.raises(OSError):
        altered.write(path)

    assert path.read_bytes() == original, "a failed write damaged the good bundle"


def test_writing_creates_the_directory_if_it_is_missing(evidence, tmp_path):
    path = tmp_path / "nested" / "deeper" / "evidence.json"
    evidence.write(path)
    assert path.exists()


def test_the_written_bytes_are_the_hash_preimage(evidence, tmp_path):
    """The file on disk must be exactly what was hashed, not a re-serialisation."""
    from eth_utils import keccak

    path = tmp_path / "evidence.json"
    evidence.write(path)
    assert "0x" + keccak(path.read_bytes()).hex() == evidence.evidence_hash_hex()
