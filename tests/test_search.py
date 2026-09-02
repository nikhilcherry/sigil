"""Search-layer behaviour, exercised without touching the network."""

import numpy as np
import pytest

from sigil.config import Config
from sigil.face import Face
from sigil.search.base import Candidate, ProviderTrace
from sigil.search.bluesky import BlueskyProvider, at_uri_to_web_url
from sigil.search.matcher import _dedup, search_and_match
from sigil.search.serpapi import SerpApiLensProvider


def _face(vec):
    v = np.asarray(vec, dtype=np.float32)
    return Face(embedding=v / np.linalg.norm(v), bbox=[0, 0, 10, 10], det_score=0.9)


def _candidate(url, handle="a.bsky.social"):
    return Candidate(
        platform="bluesky", image_url=url, post_url="https://bsky.app/x",
        post_uri="at://x", author_handle=handle, author_did="did:plc:x",
        author_display_name="A", text="", created_at="", discovered_via="test",
    )


class FakeProvider:
    """Yields fixed candidates and records what it was asked, like the real ones."""

    name = "fake"

    def __init__(self, candidates):
        self._candidates = candidates
        self.trace = ProviderTrace(provider=self.name)

    def candidates(self, query):
        self.trace.record("fake.search", {"q": query}, len(self._candidates))
        yield from self._candidates


class FakeEncoder:
    """Maps an image's bytes to a preset embedding, so matching is deterministic."""

    name = "fake"
    model = "fake"

    def __init__(self, by_bytes):
        self.by_bytes = by_bytes

    def detect_and_encode(self, image_bgr):
        return []


def test_at_uri_converts_to_a_browsable_url():
    url = at_uri_to_web_url("at://did:plc:abc/app.bsky.feed.post/3xyz", "who.bsky.social")
    assert url == "https://bsky.app/profile/who.bsky.social/post/3xyz"


def test_images_are_extracted_from_a_post_embed(bluesky_post_fixture):
    urls = BlueskyProvider._images_from_post(bluesky_post_fixture)
    assert urls == ["https://cdn/full.jpg"]


def test_images_are_extracted_from_record_with_media():
    post = {"embed": {"media": {"images": [{"fullsize": "https://cdn/m.jpg"}]}}}
    assert BlueskyProvider._images_from_post(post) == ["https://cdn/m.jpg"]


def test_post_without_images_yields_nothing():
    assert BlueskyProvider._images_from_post({"embed": {}}) == []
    assert BlueskyProvider._images_from_post({}) == []


def test_dedup_keeps_first_occurrence_of_each_image():
    cands = [_candidate("a"), _candidate("b"), _candidate("a"), _candidate("c")]
    assert [c.image_url for c in _dedup(cands)] == ["a", "b", "c"]


def test_serpapi_is_skipped_without_a_public_probe_url():
    cfg = Config()
    cfg.serpapi_key = "k"
    assert SerpApiLensProvider.available_for(cfg, None) is False
    assert SerpApiLensProvider.available_for(cfg, "/local/path.jpg") is False
    assert SerpApiLensProvider.available_for(cfg, "https://example.com/a.jpg") is True


def test_serpapi_is_skipped_without_a_key():
    cfg = Config()
    cfg.serpapi_key = None
    assert SerpApiLensProvider.available_for(cfg, "https://example.com/a.jpg") is False


def test_search_returns_no_match_when_nothing_clears_the_threshold(monkeypatch):
    """A weak best-candidate must produce no evidence at all, not a weak record."""
    import sigil.search.matcher as m

    monkeypatch.setattr(m, "fetch_image", lambda s, u, t: b"bytes-" + u.encode())
    monkeypatch.setattr(m, "score_image", lambda e, p, b: (0.11, 1, [0, 0, 5, 5]))

    cfg = Config()
    provider = FakeProvider([_candidate("https://a"), _candidate("https://b")])
    result = search_and_match(
        FakeEncoder({}), _face([1, 0, 0]), [provider], "q", 0.38, cfg
    )
    assert result.found is False
    assert result.best is None
    assert result.ranked and result.ranked[0].similarity == pytest.approx(0.11)


def test_search_picks_the_highest_scoring_candidate(monkeypatch):
    import sigil.search.matcher as m

    scores = {"https://a": 0.20, "https://b": 0.81, "https://c": 0.55}
    monkeypatch.setattr(m, "fetch_image", lambda s, u, t: u.encode())
    monkeypatch.setattr(
        m, "score_image", lambda e, p, blob: (scores[blob.decode()], 1, [0, 0, 5, 5])
    )

    provider = FakeProvider([_candidate(u) for u in scores])
    result = search_and_match(
        FakeEncoder({}), _face([1, 0, 0]), [provider], "q", 0.38, Config()
    )
    assert result.found
    assert result.best.candidate.image_url == "https://b"
    assert [s.similarity for s in result.ranked] == sorted(
        scores.values(), reverse=True
    )


def test_undownloadable_images_are_skipped_not_fatal(monkeypatch):
    import sigil.search.matcher as m

    monkeypatch.setattr(m, "fetch_image", lambda s, u, t: None if "bad" in u else b"ok")
    monkeypatch.setattr(m, "score_image", lambda e, p, b: (0.9, 1, [0, 0, 5, 5]))

    provider = FakeProvider([_candidate("https://bad"), _candidate("https://good")])
    result = search_and_match(
        FakeEncoder({}), _face([1, 0, 0]), [provider], "q", 0.38, Config()
    )
    assert result.images_examined == 1
    assert result.found


def test_max_images_caps_the_work(monkeypatch):
    import sigil.search.matcher as m

    monkeypatch.setattr(m, "fetch_image", lambda s, u, t: b"x")
    monkeypatch.setattr(m, "score_image", lambda e, p, b: (0.1, 1, []))

    cfg = Config()
    cfg.max_images = 5
    provider = FakeProvider([_candidate(f"https://{i}") for i in range(50)])
    result = search_and_match(
        FakeEncoder({}), _face([1, 0, 0]), [provider], "q", 0.38, cfg
    )
    assert result.images_examined == 5


def test_the_trace_records_what_was_actually_queried(monkeypatch):
    import sigil.search.matcher as m

    monkeypatch.setattr(m, "fetch_image", lambda s, u, t: b"x")
    monkeypatch.setattr(m, "score_image", lambda e, p, b: (0.9, 1, []))

    provider = FakeProvider([_candidate("https://a")])
    result = search_and_match(
        FakeEncoder({}), _face([1, 0, 0]), [provider], "who", 0.38, Config()
    )
    assert result.trace == [
        {"provider": "fake", "calls": [{"endpoint": "fake.search",
                                        "params": {"q": "who"}, "results": 1}]}
    ]


def test_identical_bytes_reuse_the_score_without_dropping_the_post(monkeypatch):
    """One image served under several URLs must not cost several inferences.

    It must also not collapse the candidates: the verdict is identical, but the
    *posts* differ, and the post is what ends up anchored.
    """
    import sigil.search.matcher as m

    calls = []

    def score(encoder, probe, blob):
        calls.append(blob)
        return 0.9, 1, [0, 0, 5, 5]

    monkeypatch.setattr(m, "fetch_image", lambda s, u, t: b"same-bytes")
    monkeypatch.setattr(m, "score_image", score)

    urls = ["https://cdn/a?w=1", "https://cdn/a?w=2", "https://cdn/a?w=3"]
    provider = FakeProvider([_candidate(u, handle=f"h{i}") for i, u in enumerate(urls)])
    result = search_and_match(
        FakeEncoder({}), _face([1, 0, 0]), [provider], "q", 0.38, Config()
    )

    assert len(calls) == 1, "the same bytes were encoded more than once"
    assert result.inference_reused == 2
    assert result.images_examined == 3
    assert [s.candidate.author_handle for s in result.ranked] == ["h0", "h1", "h2"]


def test_downloads_keep_running_while_the_encoder_works(monkeypatch):
    """Network and inference must overlap, not alternate.

    A batch-synchronous loop leaves every download worker idle for the whole
    inference phase. The property that distinguishes the two designs is not
    wall-clock time - it is whether a fetch ever *starts* while a score is in
    flight.
    """
    import threading
    import time

    import sigil.search.matcher as m

    scoring = threading.Event()
    overlapped = threading.Event()

    def fetch(session, url, timeout):
        if scoring.is_set():
            overlapped.set()
        time.sleep(0.01)
        return url.encode()

    def score(encoder, probe, blob):
        scoring.set()
        time.sleep(0.02)
        scoring.clear()
        return 0.1, 1, []

    monkeypatch.setattr(m, "fetch_image", fetch)
    monkeypatch.setattr(m, "score_image", score)

    cfg = Config()
    cfg.max_images = m.PREFETCH * 2
    provider = FakeProvider([_candidate(f"https://{i}") for i in range(cfg.max_images)])
    search_and_match(FakeEncoder({}), _face([1, 0, 0]), [provider], "q", 0.38, cfg)

    assert overlapped.is_set(), "no download overlapped with inference"


def test_prefetch_preserves_candidate_order(monkeypatch):
    """Downloads finish out of order; results must not.

    Ranking sorts by similarity, but ties break on arrival order, so a run that
    reordered its candidates would not be reproducible.
    """
    import random
    import time

    import sigil.search.matcher as m

    def fetch(session, url, timeout):
        time.sleep(random.uniform(0, 0.01))
        return url.encode()

    monkeypatch.setattr(m, "fetch_image", fetch)
    monkeypatch.setattr(m, "score_image", lambda e, p, b: (0.5, 1, []))

    cfg = Config()
    cfg.max_images = 40
    urls = [f"https://{i}" for i in range(cfg.max_images)]
    provider = FakeProvider([_candidate(u) for u in urls])
    result = search_and_match(
        FakeEncoder({}), _face([1, 0, 0]), [provider], "q", 0.38, cfg
    )

    assert [s.candidate.image_url for s in result.ranked] == urls[:20]
