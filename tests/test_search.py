"""Search-layer behaviour, exercised without touching the network."""

import itertools

import numpy as np
import pytest

from sigil.config import Config
from sigil.face import Face
from sigil.search.base import Candidate, ProviderTrace
from sigil.search.bluesky import BlueskyProvider, at_uri_to_web_url
from sigil.search.matcher import _dedup, interleave, search_and_match
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


def test_interleave_takes_from_every_stream_in_turn():
    got = list(interleave([iter("abc"), iter("xy"), iter("1")]))
    assert got == ["a", "x", "1", "b", "y", "c"]


def test_interleave_is_lazy_and_does_not_drain_a_stream_to_reach_the_next():
    """A chained stream only advances arm 2 after arm 1 is exhausted."""
    pulled = []

    def counted(tag, n):
        for i in range(n):
            pulled.append(tag)
            yield f"{tag}{i}"

    merged = interleave([counted("a", 100), counted("b", 100)])
    first_four = [next(merged) for _ in range(4)]
    assert first_four == ["a0", "b0", "a1", "b1"]
    # One read-ahead per stream is inherent to round-robin; a hundred is not.
    assert pulled.count("b") <= 3


def test_a_prolific_arm_cannot_starve_the_others_out_of_the_budget():
    """The regression this exists for.

    Bluesky routinely yields more candidates than ``max_images``. Chained, that
    meant a configured Vision arm was never advanced even once - reported as a
    provider, credited in the trace, and never actually asked.
    """
    prolific = (f"a{i}" for i in range(500))
    quiet = iter(["b0", "b1"])
    budget = list(itertools.islice(interleave([prolific, quiet]), 20))
    assert "b0" in budget and "b1" in budget


def test_interleave_over_no_streams_is_empty_not_a_hang():
    assert list(interleave([])) == []


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
    monkeypatch.setattr(m, "score_image", lambda e, p, b, fp=None: (0.11, 1, [0, 0, 5, 5], 0.0))

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
        m, "score_image", lambda e, p, blob, fp=None: (scores[blob.decode()], 1, [0, 0, 5, 5], 0.0)
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
    monkeypatch.setattr(m, "score_image", lambda e, p, b, fp=None: (0.9, 1, [0, 0, 5, 5], 0.0))

    provider = FakeProvider([_candidate("https://bad"), _candidate("https://good")])
    result = search_and_match(
        FakeEncoder({}), _face([1, 0, 0]), [provider], "q", 0.38, Config()
    )
    assert result.images_examined == 1
    assert result.found


def test_max_images_caps_the_work(monkeypatch):
    import sigil.search.matcher as m

    monkeypatch.setattr(m, "fetch_image", lambda s, u, t: b"x")
    monkeypatch.setattr(m, "score_image", lambda e, p, b, fp=None: (0.1, 1, [], 0.0))

    cfg = Config()
    cfg.max_images = 5
    provider = FakeProvider([_candidate(f"https://{i}") for i in range(50)])
    result = search_and_match(
        FakeEncoder({}), _face([1, 0, 0]), [provider], "q", 0.38, cfg
    )
    assert result.images_examined == 5


def test_a_second_arm_is_consulted_even_when_the_first_overflows_the_budget(
    monkeypatch,
):
    """End to end through the matcher, not just the merge helper.

    A prolific first arm used to consume ``max_images`` entirely, so a second
    provider's generator was never advanced - meaning it never issued its API
    call and contributed nothing, while still being named as a provider used.
    """
    import sigil.search.matcher as m

    monkeypatch.setattr(m, "fetch_image", lambda s, u, t: b"x")
    monkeypatch.setattr(m, "score_image", lambda e, p, b, fp=None: (0.1, 1, [], 0.0))

    cfg = Config()
    cfg.max_images = 10
    prolific = FakeProvider([_candidate(f"https://a{i}") for i in range(500)])
    quiet = FakeProvider([_candidate("https://b0"), _candidate("https://b1")])
    result = search_and_match(
        FakeEncoder({}), _face([1, 0, 0]), [prolific, quiet], "q", 0.38, cfg
    )

    assert quiet.trace.calls, "the second arm was never asked for candidates"
    seen = {s.candidate.image_url for s in result.ranked}
    assert {"https://b0", "https://b1"} <= seen
    assert result.images_examined == 10


def test_the_trace_records_what_was_actually_queried(monkeypatch):
    import sigil.search.matcher as m

    monkeypatch.setattr(m, "fetch_image", lambda s, u, t: b"x")
    monkeypatch.setattr(m, "score_image", lambda e, p, b, fp=None: (0.9, 1, [], 0.0))

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

    def score(encoder, probe, blob, fp=None):
        calls.append(blob)
        return 0.9, 1, [0, 0, 5, 5], 0.0

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

    def score(encoder, probe, blob, fp=None):
        scoring.set()
        time.sleep(0.02)
        scoring.clear()
        return 0.1, 1, [], 0.0

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
    monkeypatch.setattr(m, "score_image", lambda e, p, b, fp=None: (0.5, 1, [], 0.0))

    cfg = Config()
    cfg.max_images = 40
    urls = [f"https://{i}" for i in range(cfg.max_images)]
    provider = FakeProvider([_candidate(u) for u in urls])
    result = search_and_match(
        FakeEncoder({}), _face([1, 0, 0]), [provider], "q", 0.38, cfg
    )

    assert [s.candidate.image_url for s in result.ranked] == urls[:20]


class FakeBskySession:
    """Stands in for the AppView, and records how many calls overlap."""

    def __init__(self, actors, delay=0.02):
        import threading

        self.actors = actors
        self.delay = delay
        self.lock = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0

    def get(self, url, params=None, headers=None, timeout=None):
        import time

        with self.lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            time.sleep(self.delay)
            if "searchActors" in url:
                body = {"actors": self.actors}
            else:
                who = params["actor"]
                body = {"feed": [{"post": {
                    "uri": f"at://{who}/app.bsky.feed.post/1",
                    "author": {"handle": who, "did": f"did:{who}"},
                    "record": {"text": "hi", "createdAt": "2026-01-01"},
                    "embed": {"images": [{"fullsize": f"https://img/{who}.jpg"}]},
                }}]}
        finally:
            with self.lock:
                self.in_flight -= 1

        class R:
            status_code = 200

            @staticmethod
            def json():
                return body

        return R()


def _bsky_provider(actors, delay=0.02):
    p = BlueskyProvider(Config())
    p.session = FakeBskySession(actors, delay)
    return p


def _actors(n):
    return [{"handle": f"a{i}.bsky.social", "did": f"did:{i}",
             "displayName": f"A{i}", "avatar": f"https://av/{i}.jpg"} for i in range(n)]


def test_author_feeds_are_fetched_concurrently():
    """One round trip per matching account, and they do not depend on each other.

    Serially this is the single largest wait in a run.
    """
    provider = _bsky_provider(_actors(12))
    list(provider.candidates("q"))

    assert provider.session.max_in_flight > 1


def test_concurrent_feeds_do_not_reorder_the_candidate_stream():
    """Order must match a serial run exactly: avatar then posts, actor by actor.

    Ranking breaks ties on arrival order, so a reordered stream would make a
    run unreproducible.
    """
    provider = _bsky_provider(_actors(8))
    urls = [c.image_url for c in provider.candidates("q")]

    expected = []
    for i in range(8):
        expected.append(f"https://av/{i}.jpg")
        expected.append(f"https://img/a{i}.bsky.social.jpg")
    assert urls == expected


def test_the_trace_stays_in_actor_order_despite_concurrency():
    """The trace is audit evidence; it is written on the consuming thread so it
    reads the same as a serial run rather than in completion order."""
    provider = _bsky_provider(_actors(6))
    list(provider.candidates("q"))

    endpoints = [c["endpoint"] for c in provider.trace.calls]
    assert endpoints[0] == "app.bsky.actor.searchActors"
    assert endpoints[1:] == ["app.bsky.feed.getAuthorFeed"] * 6
    assert [c["params"]["actor"] for c in provider.trace.calls[1:]] == [
        f"a{i}.bsky.social" for i in range(6)
    ]


def test_a_failed_feed_is_recorded_once_not_twice():
    """The trace is the difference between "it searched" and "it claims it
    searched" - a failure that appears twice overstates what was asked."""
    provider = _bsky_provider(_actors(2))
    real_get = provider.session.get

    def failing(url, params=None, headers=None, timeout=None):
        if "getAuthorFeed" in url:
            class R:
                status_code = 500

                @staticmethod
                def json():
                    return {}
            return R()
        return real_get(url, params, headers, timeout)

    provider.session.get = failing
    list(provider.candidates("q"))

    feed_calls = [c for c in provider.trace.calls
                  if c["endpoint"] == "app.bsky.feed.getAuthorFeed"]
    assert len(feed_calls) == 2
    assert all(c["results"] == 0 for c in feed_calls)


class FakeAuthSession:
    """A session that can also answer createSession, like the real bsky.social."""

    def __init__(self, token="jwt-123", auth_status=200):
        self.token = token
        self.auth_status = auth_status
        self.seen = []

    def post(self, url, json=None, timeout=None):
        self.seen.append(("POST", url, json))
        status, token = self.auth_status, self.token

        class R:
            status_code = status

            @staticmethod
            def json():
                return {"accessJwt": token}

        return R()

    def get(self, url, params=None, headers=None, timeout=None):
        self.seen.append(("GET", url, headers))
        if "searchActors" in url:
            body = {"actors": []}
        else:
            body = {"posts": [{
                "uri": "at://did:plc:z/app.bsky.feed.post/9",
                "author": {"handle": "z.bsky.social", "did": "did:plc:z"},
                "record": {"text": "hello", "createdAt": "2026-01-01"},
                "embed": {"images": [{"fullsize": "https://img/z.jpg"}]},
            }]}

        class R:
            status_code = 200

            @staticmethod
            def json():
                return body

        return R()


def _authed_provider(monkeypatch, **kw):
    monkeypatch.setenv("BLUESKY_HANDLE", "me.bsky.social")
    monkeypatch.setenv("BLUESKY_APP_PASSWORD", "app-pw")
    session = FakeAuthSession(**kw)
    monkeypatch.setattr("sigil.search.bluesky.make_session", lambda: session)
    return BlueskyProvider(Config()), session


def test_an_app_password_unlocks_post_search(monkeypatch):
    """searchPosts is the one endpoint that needs a session. With credentials it
    is used; the README's claim that it widens the net depends on this."""
    provider, session = _authed_provider(monkeypatch)

    assert provider.authenticated
    urls = [c.image_url for c in provider.candidates("q")]

    assert urls == ["https://img/z.jpg"]
    endpoints = [c["endpoint"] for c in provider.trace.calls]
    assert "app.bsky.feed.searchPosts" in endpoints


def test_post_search_candidates_say_where_they_came_from(monkeypatch):
    """The trace and the candidate must agree on provenance."""
    provider, _ = _authed_provider(monkeypatch)

    found = list(provider.candidates("q"))
    assert all(c.discovered_via == "app.bsky.feed.searchPosts" for c in found)
    assert found[0].post_url == "https://bsky.app/profile/z.bsky.social/post/9"


def test_post_search_is_authenticated_against_the_auth_host(monkeypatch):
    """An anonymous call to searchPosts 403s, so it must carry the bearer token
    and go to bsky.social rather than the public AppView."""
    provider, session = _authed_provider(monkeypatch)
    list(provider.candidates("q"))

    posts_call = [c for c in session.seen
                  if c[0] == "GET" and "searchPosts" in c[1]][0]
    assert posts_call[1].startswith("https://bsky.social/xrpc")
    assert posts_call[2]["Authorization"] == "Bearer jwt-123"


def test_failed_auth_degrades_to_anonymous_rather_than_failing(monkeypatch):
    """Credentials are strictly an upgrade - a bad app password must not stop a
    run that would have worked without one."""
    provider, _ = _authed_provider(monkeypatch, auth_status=401)

    assert not provider.authenticated
    list(provider.candidates("q"))
    endpoints = [c["endpoint"] for c in provider.trace.calls]
    assert "app.bsky.feed.searchPosts" not in endpoints


def test_no_credentials_means_no_post_search(monkeypatch):
    monkeypatch.delenv("BLUESKY_HANDLE", raising=False)
    monkeypatch.delenv("BLUESKY_APP_PASSWORD", raising=False)
    session = FakeAuthSession()
    monkeypatch.setattr("sigil.search.bluesky.make_session", lambda: session)

    provider = BlueskyProvider(Config())

    assert not provider.authenticated
    assert not any(c[0] == "POST" for c in session.seen), "authenticated uninvited"


class FakeSerpSession:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status
        self.params = None

    def get(self, url, params=None, timeout=None):
        self.params = params
        payload, status = self.payload, self.status

        class R:
            status_code = status

            @staticmethod
            def json():
                if isinstance(payload, Exception):
                    raise payload
                return payload

        return R()


def _serp(monkeypatch, payload, status=200):
    cfg = Config()
    cfg.serpapi_key = "secret-key"
    session = FakeSerpSession(payload, status)
    monkeypatch.setattr("sigil.search.serpapi.make_session", lambda: session)
    return SerpApiLensProvider(cfg, "https://example.com/probe.jpg"), session


def test_lens_visual_matches_become_candidates(monkeypatch):
    provider, _ = _serp(monkeypatch, {"visual_matches": [
        {"image": "https://img/1.jpg", "link": "https://site.example/post",
         "source": "Site", "title": "a caption"},
    ]})

    found = list(provider.candidates("ignored"))

    assert len(found) == 1
    assert found[0].image_url == "https://img/1.jpg"
    assert found[0].post_url == "https://site.example/post"
    assert found[0].discovered_via == "serpapi:google_lens:visual_matches"


def test_lens_matches_missing_an_image_or_link_are_dropped(monkeypatch):
    """A candidate with no image cannot be face-verified, and one with no link
    cannot be cited as evidence."""
    provider, _ = _serp(monkeypatch, {"visual_matches": [
        {"link": "https://site.example/a"},
        {"image": "https://img/b.jpg"},
        {"image": "https://img/c.jpg", "link": "https://site.example/c"},
    ]})

    assert [c.image_url for c in provider.candidates("q")] == ["https://img/c.jpg"]


def test_the_api_key_never_reaches_the_evidence_trace(monkeypatch):
    """The trace is written into the evidence bundle, which is published and
    hashed on chain - a key in there is a key leaked permanently."""
    provider, session = _serp(monkeypatch, {"visual_matches": []})

    list(provider.candidates("q"))

    assert session.params["api_key"] == "secret-key", "the key must still be sent"
    assert "secret-key" not in str(provider.trace.calls)
    assert provider.trace.calls[0]["params"] == {"url": "https://example.com/probe.jpg"}


def test_a_failed_lens_call_yields_nothing_rather_than_raising(monkeypatch):
    """The open-web arm is optional; its outage must not end the run."""
    provider, _ = _serp(monkeypatch, {"visual_matches": [{"image": "i", "link": "l"}]},
                        status=500)

    assert list(provider.candidates("q")) == []
    assert provider.trace.calls[0]["results"] == 0


def test_malformed_lens_json_yields_nothing_rather_than_raising(monkeypatch):
    provider, _ = _serp(monkeypatch, ValueError("not json"))

    assert list(provider.candidates("q")) == []


def test_reused_verdicts_are_not_counted_as_faces_compared(monkeypatch):
    """The report calls this "faces compared". A reused verdict compared
    nothing, and this project's traces are meant to be literally true."""
    import sigil.search.matcher as m

    monkeypatch.setattr(m, "fetch_image", lambda s, u, t: b"same-bytes")
    monkeypatch.setattr(m, "score_image", lambda e, p, b, fp=None: (0.9, 2, [0, 0, 5, 5], 0.0))

    provider = FakeProvider([_candidate(f"https://cdn/a?w={i}") for i in range(4)])
    result = search_and_match(
        FakeEncoder({}), _face([1, 0, 0]), [provider], "q", 0.38, Config()
    )

    assert result.images_examined == 4
    assert result.inference_reused == 3
    assert result.faces_examined == 2, "counted faces it never compared"
    assert result.images_with_faces == 4


def test_progress_counts_every_image_not_only_the_ones_with_faces(monkeypatch):
    """The live counter and the final report must agree.

    Emitting progress only for scored candidates left the UI showing a smaller
    "fetched" number than the summary printed at the end of the same run -
    two numbers contradicting each other on one screen.
    """
    import sigil.search.matcher as m

    # Only the middle image has a detectable face.
    faces = {"https://b": (0.9, 1, [0, 0, 5, 5], 0.0)}
    monkeypatch.setattr(m, "fetch_image", lambda s, u, t: u.encode())
    monkeypatch.setattr(m, "score_image",
                        lambda e, p, blob, fp=None: faces.get(blob.decode(), (-1.0, 0, [], 0.0)))

    events = []
    provider = FakeProvider([_candidate(u) for u in
                             ("https://a", "https://b", "https://c")])
    result = search_and_match(
        FakeEncoder({}), _face([1, 0, 0]), [provider], "q", 0.38, Config(),
        on_event=events.append,
    )

    progress = [e for e in events if e["type"] == "progress"]
    assert [p["examined"] for p in progress] == [1, 2, 3]
    assert progress[-1]["examined"] == result.images_examined
    # Only the one real face becomes a candidate event.
    assert len([e for e in events if e["type"] == "candidate"]) == 1
    assert progress[-1]["scored"] == 1
