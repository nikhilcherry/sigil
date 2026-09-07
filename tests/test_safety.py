"""The candidate safety screen: what it refuses, and what it must not refuse."""

import pytest

from sigil.search.base import Candidate
from sigil.search.matcher import MatchResult, screen
from sigil.search.safety import classify, is_safe, registrable_host


def _candidate(post_url="https://bsky.app/x", image_url="https://cdn.example.com/a.jpg",
               handle="a.bsky.social", display_name="A"):
    return Candidate(
        platform="bluesky", image_url=image_url, post_url=post_url,
        post_uri="at://x", author_handle=handle, author_did="did:plc:x",
        author_display_name=display_name, text="", created_at="",
        discovered_via="test",
    )


# --------------------------------------------------------------- what is refused


@pytest.mark.parametrize("url", [
    "https://pornhub.com/view?v=1",
    "https://www.xvideos.com/video1",
    "https://cdn.coomer.su/data/x.jpg",
    "https://media.bunkr.si/a/b.jpg",
])
def test_blocked_hosts_are_refused_including_subdomains(url):
    verdict = classify(post_url=url)
    assert not verdict
    assert verdict.category == "host"


def test_a_blocked_host_on_the_image_url_alone_is_enough():
    """The page can be innocuous while the image it embeds is not."""
    verdict = classify(post_url="https://example.com/article",
                       image_url="https://thothub.tv/i/1.jpg")
    assert not verdict and verdict.category == "host"


def test_adult_subreddits_are_refused_without_blocking_reddit():
    assert not classify(post_url="https://www.reddit.com/r/gonewild/comments/x/")
    assert classify(post_url="https://www.reddit.com/r/india/comments/x/")


def test_tokens_in_the_url_path_are_refused():
    verdict = classify(post_url="https://someblog.example/leaked-nudes/anna")
    assert not verdict and verdict.category == "url"


def test_tokens_in_a_handle_or_display_name_are_refused():
    assert classify(handle="nsfw-archive.bsky.social").category == "handle"
    assert classify(display_name="Deepfake Central").category == "display_name"


# ------------------------------------------------- what must NOT be refused
#
# This half is the reason the matching is tokenised rather than a substring
# scan. Every string below contains a blocked token as a substring, and every
# one of them is an ordinary place name, surname or English word. A substring
# implementation passes the block tests above and silently discards these,
# which is invisible unless it is asserted.


@pytest.mark.parametrize("url", [
    "https://www.sussex.ac.uk/profiles/12345",          # sex
    "https://example.com/passport/renewal",             # ass
    "https://news.example/2019/middlesex-council",      # sex
    "https://example.com/class-of-2020/photos",         # ass
    "https://example.com/leaksville-nc/history",        # leaks
    "https://example.com/nudged-into-office",           # nude
    "https://scunthorpe.example/parish",                # the canonical case
    "https://example.com/analysis/quarterly",           # anal
])
def test_ordinary_urls_are_not_refused(url):
    assert classify(post_url=url), url


@pytest.mark.parametrize("name", [
    "Cassidy Hutchinson",     # ass
    "Essex County Council",   # sex
    "Anand Nudurupati",       # nude
    "Bassam Al-Assad",        # ass, twice
])
def test_ordinary_names_are_not_refused(name):
    assert classify(display_name=name, handle=name.lower().replace(" ", "")), name


def test_an_unlisted_host_whose_domain_contains_a_token_is_not_refused():
    """`sexton` is a surname; the domain is not tokenised for exactly this."""
    assert classify(post_url="https://sextonpublishing.com/about")


def test_post_text_is_never_a_reason_to_refuse():
    """A word in a post does not make its source unsafe - see the module docstring.

    Refusing here would discard a true match and report it as "not found",
    which is the worst available failure: it looks identical to the search
    simply not having worked.
    """
    c = _candidate()
    c.text = "sexy new album art, nudes of the sculpture at the Met"
    assert is_safe(c)


def test_malformed_urls_do_not_raise():
    assert registrable_host("not a url") == ""
    assert classify(post_url="http://[oops", image_url="")


# ------------------------------------------------------------------ the screen


def test_screen_counts_by_category_and_yields_the_rest():
    result = MatchResult(best=None)
    kept = list(screen(
        [_candidate(post_url="https://pornhub.com/a"),
         _candidate(post_url="https://bsky.app/ok"),
         _candidate(handle="nudes.bsky.social")],
        result,
    ))
    assert [c.post_url for c in kept] == ["https://bsky.app/ok"]
    assert result.blocked_unsafe == 2
    assert result.blocked_by_category == {"host": 1, "handle": 1}


def test_screen_events_carry_the_category_but_never_the_url():
    """A blocked source echoed to the terminal has defeated the point of blocking it."""
    events = []
    result = MatchResult(best=None)
    list(screen([_candidate(post_url="https://pornhub.com/a")], result,
                on_event=events.append))
    assert events == [{"type": "blocked", "category": "host", "total": 1}]
    assert "pornhub" not in repr(events)


def test_allow_unsafe_passes_everything_through_uncounted():
    result = MatchResult(best=None)
    kept = list(screen([_candidate(post_url="https://pornhub.com/a")], result,
                       allow_unsafe=True))
    assert len(kept) == 1
    assert result.blocked_unsafe == 0


# ------------------------------------------------- the screen inside a real search


class _Provider:
    """A search arm yielding fixed candidates, like the real ones."""

    name = "fake"
    kind = "web"

    def __init__(self, candidates):
        from sigil.search.base import ProviderTrace

        self._candidates = candidates
        self.trace = ProviderTrace(provider=self.name)

    def candidates(self, query):
        self.trace.record("fake.search", {"q": query}, len(self._candidates))
        yield from self._candidates


def _search(providers, cfg, monkeypatch, **kw):
    import numpy as np

    import sigil.search.matcher as m
    from sigil.face import Face
    from sigil.search.matcher import search_and_match

    monkeypatch.setattr(m, "fetch_image", lambda s, u, t: b"bytes-" + u.encode())
    monkeypatch.setattr(m, "score_image", lambda e, p, b, fp=None: (0.9, 1, [], 0.0))
    probe = Face(embedding=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
                 bbox=[0, 0, 10, 10], det_score=0.9)
    return search_and_match(object(), probe, providers, "q", 0.38, cfg, **kw)


def test_refused_candidates_do_not_consume_the_image_budget(monkeypatch):
    """Screening after the cap would spend the whole run on things it refused.

    With max_images=2 and two blocked sources ahead of two usable ones, a
    screen placed after the truncation examines nothing and reports "no match",
    while the arm that had the answer never advanced.
    """
    from sigil.config import Config

    cfg = Config()
    cfg.max_images = 2
    provider = _Provider([
        _candidate(post_url="https://pornhub.com/1", image_url="https://x/1.jpg"),
        _candidate(post_url="https://coomer.su/2", image_url="https://x/2.jpg"),
        _candidate(post_url="https://bsky.app/3", image_url="https://x/3.jpg"),
        _candidate(post_url="https://bsky.app/4", image_url="https://x/4.jpg"),
    ])
    result = _search([provider], cfg, monkeypatch)

    assert result.images_examined == 2
    assert result.blocked_unsafe == 2
    assert result.found
    assert result.best.candidate.post_url.startswith("https://bsky.app/")


def test_the_screen_is_recorded_in_the_trace_without_naming_a_host(monkeypatch):
    """The trace is hashed into the bundle, so it carries counts and no hosts."""
    import json

    from sigil.config import Config

    provider = _Provider([
        _candidate(post_url="https://pornhub.com/1", image_url="https://x/1.jpg"),
        _candidate(post_url="https://bsky.app/2", image_url="https://x/2.jpg"),
    ])
    result = _search([provider], Config(), monkeypatch)

    entry = [t for t in result.trace if t["provider"] == "safety"]
    assert entry == [{"provider": "safety", "calls": [
        {"endpoint": "screen", "params": {"categories": {"host": 1}}, "results": 1}
    ]}]
    assert "pornhub" not in json.dumps(result.trace)


def test_no_safety_entry_in_the_trace_when_nothing_was_refused(monkeypatch):
    from sigil.config import Config

    provider = _Provider([_candidate(post_url="https://bsky.app/2",
                                     image_url="https://x/2.jpg")])
    result = _search([provider], Config(), monkeypatch)
    assert all(t["provider"] != "safety" for t in result.trace)
