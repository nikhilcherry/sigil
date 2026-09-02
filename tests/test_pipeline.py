"""The end-to-end claim, offline.

The live suite proves the real APIs still answer; this proves the orchestration
itself - scan, identify, search, match, anchor, re-verify - without depending on
a third party being up.
"""

import numpy as np
import pytest

from sigil.chain import ChainClient
from sigil.pipeline import PipelineError, run_pipeline
from sigil.search.base import Candidate, ProviderTrace
from tests.conftest import EXAMPLE_CONTROL, EXAMPLE_PROBE


class StubProvider:
    name = "stub"

    def __init__(self, urls):
        self.urls = urls
        self.trace = ProviderTrace(provider=self.name)
        self.queries = []

    def candidates(self, query):
        self.queries.append(query)
        self.trace.record("stub.search", {"q": query}, len(self.urls))
        for url in self.urls:
            yield Candidate(
                platform="bluesky", image_url=url,
                post_url="https://bsky.app/profile/who.bsky.social/post/1",
                post_uri="at://did:plc:who/app.bsky.feed.post/1",
                author_handle="who.bsky.social", author_did="did:plc:who",
                author_display_name="Who", text="a photo",
                created_at="2026-08-01T00:00:00Z", discovered_via="stub",
            )


@pytest.fixture
def wired(monkeypatch):
    """Serve real image bytes for candidate URLs, with no network."""
    import sigil.pipeline as pipe
    import sigil.search.matcher as matcher

    bodies = {
        "https://x/same.jpg": EXAMPLE_PROBE.read_bytes(),
        "https://x/other.jpg": EXAMPLE_CONTROL.read_bytes(),
        "https://x/broken.jpg": b"not an image",
    }
    monkeypatch.setattr(matcher, "fetch_image", lambda s, u, t: bodies.get(u))

    def install(urls):
        provider = StubProvider(urls)
        monkeypatch.setattr(pipe, "build_providers", lambda cfg, url: [provider])
        return provider

    return install


def test_a_matching_face_is_anchored_and_verifies_again_from_a_cold_client(wired, cfg):
    """The whole claim in one test: the record is read back out of chain state
    by a client that never saw the run, not replayed from the file."""
    wired(["https://x/other.jpg", "https://x/same.jpg"])

    result = run_pipeline(str(EXAMPLE_PROBE), "someone", cfg, do_anchor=True)

    assert result.found
    assert result.evidence.similarity > result.evidence.threshold
    assert result.anchor["already_anchored"] is False
    assert result.verification.ok
    assert ChainClient(cfg).verify(result.evidence).ok


def test_a_face_that_matches_nothing_anchors_nothing(wired, cfg):
    """The correct outcome for no match is an empty chain, not a weak record."""
    # Only the other person's photo is on offer, so the probe's face has
    # nothing to match against.
    wired(["https://x/other.jpg"])

    result = run_pipeline(str(EXAMPLE_PROBE), "someone", cfg, do_anchor=True)

    assert not result.found
    assert result.evidence is None
    assert result.anchor is None
    assert ChainClient(cfg).total_anchored() == 0


def test_undecodable_candidates_are_skipped_not_fatal(wired, cfg):
    wired(["https://x/broken.jpg", "https://x/same.jpg"])

    result = run_pipeline(str(EXAMPLE_PROBE), "someone", cfg, do_anchor=False)

    assert result.found
    assert result.match.images_examined == 2


def test_the_evidence_records_what_was_actually_searched(wired, cfg):
    """The trace is the difference between "it searched" and "it says it did"."""
    provider = wired(["https://x/same.jpg"])

    result = run_pipeline(str(EXAMPLE_PROBE), "a query", cfg, do_anchor=False)

    assert provider.queries == ["a query"]
    assert result.evidence.search_trace == [
        {"provider": "stub", "calls": [
            {"endpoint": "stub.search", "params": {"q": "a query"}, "results": 1}
        ]}
    ]


def test_no_query_and_no_index_refuses_rather_than_guessing(wired, cfg, monkeypatch, tmp_path):
    """Without a name to search for, the honest answer is to stop."""
    import sigil.identify as idmod

    monkeypatch.setattr(idmod, "INDEX_VECTORS", tmp_path / "none.npz")
    monkeypatch.setattr(idmod, "INDEX_META", tmp_path / "none.json")
    wired(["https://x/same.jpg"])

    with pytest.raises(PipelineError, match="index"):
        run_pipeline(str(EXAMPLE_PROBE), "", cfg, do_anchor=False)


def test_a_named_face_seeds_the_search_with_its_own_name(wired, cfg, monkeypatch, tmp_path):
    """The loop the whole identity index exists to close: no query in, a real
    name out, and that name is what gets searched."""
    import json

    import sigil.identify as idmod
    from sigil.pipeline import scan_probe

    face, _, encoder = scan_probe(EXAMPLE_PROBE.read_bytes(), cfg)
    vec = np.asarray(face.embedding, dtype=np.float32).reshape(1, -1)

    vec_path, meta_path = tmp_path / "v.npz", tmp_path / "m.json"
    np.savez_compressed(vec_path, vectors=vec)
    meta_path.write_text(json.dumps({
        "backend": encoder.name, "model": encoder.model, "count": 1,
        "identities": [{"name": "A Known Person", "qid": "Q1",
                        "image_url": "https://x/portrait.jpg",
                        "source": "en.wikipedia"}],
    }))
    monkeypatch.setattr(idmod, "INDEX_VECTORS", vec_path)
    monkeypatch.setattr(idmod, "INDEX_META", meta_path)

    provider = wired(["https://x/same.jpg"])
    result = run_pipeline(str(EXAMPLE_PROBE), "", cfg, do_anchor=False)

    assert provider.queries == ["A Known Person"]
    assert result.found

