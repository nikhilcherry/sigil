"""Configuration resolution.

Every knob here comes from the environment, which means every knob can arrive
malformed. The rule is that a bad value falls back to the default rather than
taking the run down - but it has to fall back to the *documented* default, not
to whatever survives.
"""

import pytest

from sigil.config import DEFAULT_THRESHOLDS, Config


def test_defaults_hold_with_nothing_set(monkeypatch):
    for var in ("SIGIL_MAX_ACTORS", "SIGIL_MAX_IMAGES", "SIGIL_POSTS_PER_ACTOR",
                "SIGIL_HTTP_TIMEOUT", "SIGIL_THRESHOLD", "SIGIL_CHAIN"):
        monkeypatch.delenv(var, raising=False)

    cfg = Config()

    assert cfg.max_actors == 25
    assert cfg.posts_per_actor == 20
    assert cfg.max_images == 200
    assert cfg.http_timeout == 20.0
    assert cfg.threshold is None
    assert cfg.chain_backend == "local"


@pytest.mark.parametrize("value", ["", "  ", "lots", "12.5", "1e4", "-"])
def test_a_malformed_integer_falls_back_to_the_default(monkeypatch, value):
    """A typo in an env var must not crash a run, and must not quietly become
    some other number either."""
    monkeypatch.setenv("SIGIL_MAX_IMAGES", value)

    assert Config().max_images == 200


@pytest.mark.parametrize("value", ["", "soon", "abc"])
def test_a_malformed_float_falls_back_to_the_default(monkeypatch, value):
    monkeypatch.setenv("SIGIL_HTTP_TIMEOUT", value)

    assert Config().http_timeout == 20.0


def test_valid_overrides_are_honoured(monkeypatch):
    monkeypatch.setenv("SIGIL_MAX_IMAGES", "40")
    monkeypatch.setenv("SIGIL_HTTP_TIMEOUT", "5.5")

    cfg = Config()

    assert cfg.max_images == 40
    assert cfg.http_timeout == 5.5


def test_the_threshold_travels_with_the_backend():
    """ArcFace and SFace put same-person pairs at different scales, so a single
    global threshold would be wrong for one of them."""
    cfg = Config()
    cfg.threshold = None

    assert cfg.threshold_for("insightface") == DEFAULT_THRESHOLDS["insightface"]
    assert cfg.threshold_for("opencv") == DEFAULT_THRESHOLDS["opencv"]
    assert cfg.threshold_for("insightface") != cfg.threshold_for("opencv")


def test_an_unknown_backend_gets_the_conservative_default():
    cfg = Config()
    cfg.threshold = None

    assert cfg.threshold_for("something-new") == 0.38


def test_an_explicit_threshold_overrides_every_backend(monkeypatch):
    monkeypatch.setenv("SIGIL_THRESHOLD", "0.9")

    cfg = Config()

    assert cfg.threshold == 0.9
    assert cfg.threshold_for("insightface") == 0.9
    assert cfg.threshold_for("opencv") == 0.9


# --------------------------------------------- the threshold is not a soft knob


@pytest.mark.parametrize("value", ["abc", "0.4.5", "point four", "--"])
def test_a_malformed_threshold_is_refused_not_defaulted(monkeypatch, value):
    """Unlike every other knob here, this one must not fall back quietly.

    It is the boundary between "same person" and "not". Substituting 0.38 for a
    typo'd 0.5 would accept matches the operator meant to reject, and nothing
    downstream would reveal it - the run would look entirely normal.
    """
    monkeypatch.setenv("SIGIL_THRESHOLD", value)

    with pytest.raises(ValueError, match="SIGIL_THRESHOLD"):
        Config()


@pytest.mark.parametrize("value", ["5", "1.5", "-2", "100"])
def test_a_threshold_outside_cosine_range_is_refused(monkeypatch, value):
    """Above 1 nothing can ever match, below -1 everything does. Either way the
    run does something confusing rather than something wrong-looking."""
    monkeypatch.setenv("SIGIL_THRESHOLD", value)

    with pytest.raises(ValueError, match="between -1 and 1"):
        Config()


@pytest.mark.parametrize("value", ["0", "1", "-1", "0.45", " 0.5 "])
def test_a_threshold_inside_cosine_range_is_accepted(monkeypatch, value):
    monkeypatch.setenv("SIGIL_THRESHOLD", value)

    assert Config().threshold == float(value)


def test_a_blank_threshold_means_use_the_backend_default(monkeypatch):
    """An empty variable is how a .env file says "I did not set this"."""
    monkeypatch.setenv("SIGIL_THRESHOLD", "   ")

    assert Config().threshold is None
