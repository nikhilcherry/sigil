"""Telling a different photograph of a face from the probe's own photograph.

The distinction matters because the weaker kind of evidence scores higher: a
reverse-image hit is the probe's picture republished, so its face similarity is
near 1.0 for a reason that has nothing to do with recognition. These tests
check both that the signal separates the two cases and that the selection
actually prefers the stronger claim.
"""

import cv2
import numpy as np
import pytest

from sigil.provenance import (
    IDENTITY,
    PROVENANCE,
    PROVENANCE_CUTOFF,
    claim_for,
    fingerprint,
    photo_similarity,
)
from sigil.search.base import Candidate
from sigil.search.matcher import ScoredCandidate, pick_best, score_image


def _picture(seed: int, w: int = 200, h: int = 260) -> np.ndarray:
    """A deterministic, structured image - not noise, so resizing is stable."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    return cv2.resize(base, (w, h), interpolation=cv2.INTER_LINEAR)


# ------------------------------------------------------------------ the signal


def test_a_fingerprint_is_unit_length_so_a_dot_product_is_a_correlation():
    fp = fingerprint(_picture(1))
    assert fp.shape == (32 * 32,)
    assert float(np.linalg.norm(fp)) == pytest.approx(1.0, abs=1e-5)


def test_the_same_picture_correlates_at_one():
    fp = fingerprint(_picture(2))
    assert photo_similarity(fp, fp) == pytest.approx(1.0, abs=1e-5)


def test_a_rescaled_copy_is_still_recognised_as_the_same_picture():
    """Republication resizes; that must not break the signal."""
    original = _picture(3)
    smaller = cv2.resize(original, (64, 83), interpolation=cv2.INTER_AREA)
    sim = photo_similarity(fingerprint(original), fingerprint(smaller))
    assert sim >= PROVENANCE_CUTOFF, sim


def test_a_recompressed_copy_is_still_recognised_as_the_same_picture():
    original = _picture(4)
    ok, buf = cv2.imencode(".jpg", original, [cv2.IMWRITE_JPEG_QUALITY, 40])
    assert ok
    lossy = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    assert photo_similarity(fingerprint(original), fingerprint(lossy)) >= \
        PROVENANCE_CUTOFF


def test_a_brightened_copy_is_still_recognised_as_the_same_picture():
    """Mean-centring and normalising is what buys this."""
    original = _picture(5)
    brighter = cv2.convertScaleAbs(original, alpha=0.7, beta=60)
    assert photo_similarity(fingerprint(original), fingerprint(brighter)) >= \
        PROVENANCE_CUTOFF


def test_an_unrelated_picture_does_not_correlate():
    sim = photo_similarity(fingerprint(_picture(6)), fingerprint(_picture(7)))
    assert abs(sim) < PROVENANCE_CUTOFF


def test_a_flat_image_is_unrelated_to_everything_rather_than_dividing_by_zero():
    flat = np.full((40, 40, 3), 128, dtype=np.uint8)
    fp = fingerprint(flat)
    assert np.isfinite(fp).all()
    assert photo_similarity(fp, fingerprint(_picture(8))) == 0.0
    # Including to itself: no structure means no evidence, not perfect evidence.
    assert photo_similarity(fp, fp) == 0.0


def test_a_greyscale_image_is_accepted_not_just_three_channel():
    colour = _picture(9)
    grey = cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY)
    assert photo_similarity(fingerprint(colour), fingerprint(grey)) == \
        pytest.approx(1.0, abs=1e-5)


def test_similarity_is_clipped_to_the_valid_correlation_range():
    fp = fingerprint(_picture(10))
    assert -1.0 <= photo_similarity(fp, fp * 1.5) <= 1.0


def test_the_claim_follows_the_cutoff():
    assert claim_for(PROVENANCE_CUTOFF) == PROVENANCE
    assert claim_for(PROVENANCE_CUTOFF - 0.01) == IDENTITY
    assert claim_for(0.99) == PROVENANCE
    assert claim_for(0.0) == IDENTITY


# -------------------------------------------------------------------- scoring


class OneFace:
    name = model = "fake"

    def detect_and_encode(self, image_bgr):
        from sigil.face import Face

        return [Face(embedding=np.array([1.0, 0.0], dtype=np.float32),
                     bbox=[0, 0, 4, 4], det_score=0.9)]


def _probe_face():
    from sigil.face import Face

    return Face(embedding=np.array([1.0, 0.0], dtype=np.float32),
                bbox=[0, 0, 4, 4], det_score=0.9)


def test_score_image_reports_the_picture_similarity_alongside_the_face_score():
    picture = _picture(11)
    ok, buf = cv2.imencode(".png", picture)
    assert ok
    sim, n_faces, _bbox, photo = score_image(
        OneFace(), _probe_face(), buf.tobytes(), fingerprint(picture)
    )
    assert sim == pytest.approx(1.0, abs=1e-5) and n_faces == 1
    assert photo == pytest.approx(1.0, abs=1e-5)


def test_score_image_without_a_probe_fingerprint_reports_zero_not_a_crash():
    ok, buf = cv2.imencode(".png", _picture(12))
    assert ok
    assert score_image(OneFace(), _probe_face(), buf.tobytes())[3] == 0.0


def test_an_undecodable_image_reports_no_picture_similarity():
    assert score_image(OneFace(), _probe_face(), b"not an image",
                       fingerprint(_picture(13))) == (-1.0, 0, [], 0.0)


def test_a_faceless_image_still_reports_its_picture_similarity():
    """The picture is the probe's even when the detector finds nothing in it."""
    class NoFace:
        name = model = "fake"

        def detect_and_encode(self, image_bgr):
            return []

    picture = _picture(14)
    ok, buf = cv2.imencode(".png", picture)
    assert ok
    sim, n_faces, _b, photo = score_image(
        NoFace(), _probe_face(), buf.tobytes(), fingerprint(picture)
    )
    assert (sim, n_faces) == (-1.0, 0)
    assert photo == pytest.approx(1.0, abs=1e-5)


# ------------------------------------------------------------------- selection


def _scored(similarity, photo, handle="who", kind="web"):
    return ScoredCandidate(
        candidate=Candidate(
            platform="p", image_url=f"https://{handle}", post_url="https://p",
            post_uri="at://p", author_handle=handle, author_did="",
            author_display_name=handle, text="", created_at="",
            discovered_via="test", source_kind=kind,
        ),
        similarity=similarity, image_sha256=handle * 2,
        faces_in_image=1, matched_bbox=[0, 0, 1, 1], photo_similarity=photo,
    )


def test_a_scored_candidate_knows_which_claim_it_supports():
    assert _scored(0.99, 0.99).claim == PROVENANCE
    assert _scored(0.76, 0.10).claim == IDENTITY


def test_an_independent_photograph_is_anchored_over_a_higher_scoring_copy():
    """The regression this exists for.

    Reverse image search returns the probe's own picture, which scores 0.9952
    because it *is* the same picture. Ranking by cosine alone anchored that and
    presented it as the strongest evidence available, while the account whose
    avatar is a genuinely different photograph of the same face sat below it at
    0.7596 and proved far more.
    """
    copy = _scored(0.9952, 0.9998, "influencewatch")
    avatar = _scored(0.7596, 0.1200, "aoc")
    best = pick_best([copy, avatar], threshold=0.38)
    assert best is avatar
    assert best.claim == IDENTITY


def test_cosine_still_decides_between_two_independent_photographs():
    weaker = _scored(0.55, 0.10, "a")
    stronger = _scored(0.80, 0.12, "b")
    assert pick_best([weaker, stronger], threshold=0.38) is stronger


def test_cosine_still_decides_between_two_copies():
    assert pick_best([_scored(0.97, 0.99, "a"), _scored(0.99, 0.99, "b")],
                     threshold=0.38).candidate.author_handle == "b"


def test_a_copy_is_anchored_when_it_is_the_only_thing_that_cleared():
    """A republication is a real finding; it just is not an identity claim."""
    only = _scored(0.99, 0.995, "reprint")
    best = pick_best([only, _scored(0.20, 0.05, "nobody")], threshold=0.38)
    assert best is only
    assert best.claim == PROVENANCE


def test_nothing_clearing_the_threshold_selects_nothing():
    assert pick_best([_scored(0.30, 0.05), _scored(0.10, 0.02)],
                     threshold=0.38) is None


def test_an_empty_candidate_list_selects_nothing():
    assert pick_best([], threshold=0.38) is None


def test_a_below_threshold_independent_photo_does_not_beat_a_clearing_copy():
    """The preference applies only among candidates that actually cleared."""
    copy = _scored(0.99, 0.995, "reprint")
    near_miss = _scored(0.37, 0.05, "different")
    assert pick_best([copy, near_miss], threshold=0.38) is copy


def test_a_social_post_is_anchored_over_a_better_scoring_open_web_page():
    """The deliverable is a social media post; a web page corroborates it.

    Both of these are independent photographs of the same face, so the
    provenance signal cannot separate them - only where they were found can.
    """
    page = _scored(0.9892, 0.4566, "squarespace-cdn", kind="web")
    post = _scored(0.7596, 0.1200, "aoc", kind="social")
    assert pick_best([page, post], threshold=0.38) is post


def test_an_open_web_page_is_anchored_when_no_social_post_cleared():
    page = _scored(0.9892, 0.4566, "squarespace-cdn", kind="web")
    quiet = _scored(0.2000, 0.0500, "nobody", kind="social")
    assert pick_best([page, quiet], threshold=0.38) is page


def test_within_social_candidates_an_independent_photo_still_wins():
    """The two preferences compose; the source kind does not override the claim."""
    repost = _scored(0.99, 0.995, "reposter", kind="social")
    avatar = _scored(0.76, 0.12, "aoc", kind="social")
    assert pick_best([repost, avatar], threshold=0.38) is avatar
