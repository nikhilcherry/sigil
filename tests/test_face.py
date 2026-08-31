"""Face-encoder behaviour. Backend-parameterised, so both paths are held to the
same contract; a backend that cannot load on this machine is skipped, not failed."""

import numpy as np
import pytest

from sigil.face import Face, cosine, decode_image, largest_face, load_encoder
from tests.conftest import EXAMPLE_CONTROL, EXAMPLE_PROBE

BACKENDS = ["insightface", "opencv"]


@pytest.fixture(params=BACKENDS)
def encoder(request):
    try:
        enc = load_encoder(request.param)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"{request.param} unavailable: {exc}")
    if enc.name != request.param:
        pytest.skip(f"{request.param} fell back to {enc.name}")
    return enc


@pytest.fixture
def probe_image():
    if not EXAMPLE_PROBE.exists():
        pytest.skip("example probe image not present")
    return EXAMPLE_PROBE.read_bytes()


@pytest.fixture
def control_image():
    """A photo of a different person, for the cross-identity control."""
    if not EXAMPLE_CONTROL.exists():
        pytest.skip("control image not present")
    return EXAMPLE_CONTROL.read_bytes()


def test_decode_rejects_non_images():
    assert decode_image(b"not an image at all") is None
    assert decode_image(b"") is None


def test_cosine_is_bounded_and_self_similar():
    v = np.array([0.3, -0.4, 0.5], dtype=np.float32)
    assert cosine(v, v) == pytest.approx(1.0, abs=1e-6)
    assert cosine(v, -v) == pytest.approx(-1.0, abs=1e-6)
    assert -1.0 <= cosine(v, np.array([1.0, 0.0, 0.0], dtype=np.float32)) <= 1.0


def test_largest_face_picks_the_dominant_subject():
    small = Face(np.ones(4, dtype=np.float32), [0, 0, 10, 10], 0.99)
    big = Face(np.ones(4, dtype=np.float32), [0, 0, 100, 100], 0.50)
    assert largest_face([small, big]) is big
    assert largest_face([]) is None


def test_embedding_digest_is_stable_for_identical_vectors():
    a = Face(np.array([1, 2, 3], dtype=np.float32), [0, 0, 1, 1], 0.9)
    b = Face(np.array([1, 2, 3], dtype=np.float32), [0, 0, 1, 1], 0.1)
    assert a.embedding_sha256 == b.embedding_sha256


def test_encoder_finds_no_face_in_a_blank_image(encoder):
    assert encoder.detect_and_encode(np.zeros((480, 480, 3), dtype=np.uint8)) == []


def test_encoder_extracts_a_normalised_embedding(encoder, probe_image):
    faces = encoder.detect_and_encode(decode_image(probe_image))
    assert faces, "expected at least one face in the example probe"
    face = largest_face(faces)
    assert np.linalg.norm(face.embedding) == pytest.approx(1.0, abs=1e-3)
    assert face.bbox[2] > face.bbox[0] and face.bbox[3] > face.bbox[1]


def test_encoding_is_deterministic(encoder, probe_image):
    """Same bytes in, same digest out - otherwise no hash could ever verify."""
    img = decode_image(probe_image)
    first = largest_face(encoder.detect_and_encode(img))
    second = largest_face(encoder.detect_and_encode(img))
    assert first.embedding_sha256 == second.embedding_sha256


def test_two_different_people_score_below_the_threshold(
    encoder, probe_image, control_image
):
    """The separation every threshold in this project rests on.

    Two unrelated faces must land far below the decision boundary. If this ever
    narrows, the thresholds in config.py are no longer defensible.
    """
    from sigil.config import DEFAULT_THRESHOLDS

    a = largest_face(encoder.detect_and_encode(decode_image(probe_image)))
    b = largest_face(encoder.detect_and_encode(decode_image(control_image)))
    assert a is not None and b is not None

    cross = cosine(a.embedding, b.embedding)
    threshold = DEFAULT_THRESHOLDS[encoder.name]
    assert cross < threshold, f"{encoder.name}: unrelated faces scored {cross:.4f}"
    assert cosine(a.embedding, a.embedding) - cross > 0.5


def test_a_random_vector_is_near_orthogonal_to_a_real_face(encoder, probe_image):
    """Guards against a degenerate embedding space where everything matches."""
    rng = np.random.default_rng(0)
    face = largest_face(encoder.detect_and_encode(decode_image(probe_image)))
    noise = rng.normal(size=face.embedding.shape).astype(np.float32)
    assert abs(cosine(face.embedding, noise)) < 0.3
