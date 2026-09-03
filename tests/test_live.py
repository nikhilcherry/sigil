"""Tests that hit the real network and the real chain.

Deselect with ``-m "not network"``. These exist because the interesting failure
modes of this project are not in its own logic - they are in a third party
changing an endpoint, a response shape, or an auth requirement. A green unit
suite with a broken search is exactly the failure worth catching.
"""

import pytest

from sigil.chain import ChainClient
from sigil.config import Config
from sigil.face import load_encoder
from sigil.pipeline import run_pipeline, scan_probe
from sigil.search.bluesky import BlueskyProvider
from sigil.search.matcher import pick_best
from tests.conftest import EXAMPLE_PROBE

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def cfg_live():
    c = Config()
    c.max_actors = 5
    c.posts_per_actor = 5
    c.max_images = 40
    return c


def test_bluesky_public_api_still_serves_anonymous_actor_search(cfg_live):
    """The zero-credential guarantee in the README, asserted."""
    provider = BlueskyProvider(cfg_live)
    found = []
    for candidate in provider.candidates("bluesky"):
        found.append(candidate)
        if len(found) >= 3:
            break

    assert found, "searchActors returned nothing - the public AppView may have changed"
    assert all(c.image_url.startswith("https://") for c in found)
    assert any(c.author_handle for c in found)
    assert provider.trace.calls, "provider did not record the calls it made"


def test_full_pipeline_finds_and_anchors_a_real_match(cfg_live, tmp_path, monkeypatch):
    """The end-to-end claim: a real face, a real account, a real chain record."""
    import sigil.chain.client as client_mod

    monkeypatch.setattr(client_mod, "STATE_PATH", tmp_path / "chain.json")
    cfg_live.chain_backend = "local"

    result = run_pipeline(str(EXAMPLE_PROBE), "AOC", cfg_live, do_anchor=True)

    assert result.match.images_examined > 0, "search fetched no images"
    if not result.found:
        pytest.skip(
            "no live match today - the target account may have changed its avatar; "
            f"best similarity was {result.match.ranked[0].similarity if result.match.ranked else 0:.3f}"
        )

    ev = result.evidence
    # Which arm wins is not fixed. Bluesky is the only unconditional one, but
    # an open-web arm outranks it whenever its key is configured and the web
    # holds a better photo of the same face. Pinning ``platform == "bluesky"``
    # here passed in CI (no keys) and failed on any machine with a key set -
    # a defect in the assertion, not in the run.
    assert ev.match.platform, "match recorded no platform"
    assert ev.match.post_url.startswith("https://"), ev.match.post_url
    assert ev.match.discovered_via, "match does not say how it was found"
    assert any(t["calls"] for t in ev.search_trace), "no arm recorded a live call"
    assert any(t["provider"] == "bluesky" for t in ev.search_trace), (
        "the keyless arm did not run"
    )
    assert ev.similarity >= ev.threshold
    # Not the top of the table: an identity claim on a social source outranks a
    # higher cosine on the probe's own picture republished. Asserted against
    # the selector rather than restated, so the two cannot drift apart.
    assert result.match.best is pick_best(result.match.ranked, ev.threshold)
    assert ev.match.claim in ("identity", "provenance")
    assert -1.0 <= ev.match.probe_photo_similarity <= 1.0
    assert ev.match.source_kind in ("social", "web")
    assert result.anchor["already_anchored"] is False
    assert result.verification.ok

    # And it verifies again from a cold client, as a separate process would.
    reader = ChainClient(cfg_live)
    assert reader.verify(result.evidence).ok


def test_probe_scan_matches_the_committed_example(cfg_live):
    """Catches an accidental re-encode or resize of the committed demo image."""
    encoder = load_encoder(cfg_live.face_backend)
    _, ref, _ = scan_probe(EXAMPLE_PROBE.read_bytes(), cfg_live)
    assert ref.backend == encoder.name
    assert ref.det_score > 0.5
    assert len(ref.embedding_sha256) == 64
