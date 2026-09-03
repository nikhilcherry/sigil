import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sigil.config import Config  # noqa: E402
from sigil.evidence import Evidence, MatchRef, ProbeRef  # noqa: E402

EXAMPLE_PROBE = ROOT / "examples" / "probe-aoc.jpg"
EXAMPLE_CONTROL = ROOT / "examples" / "control-buttigieg.jpg"


@pytest.fixture
def probe_ref():
    return ProbeRef(
        image_sha256="ab" * 32,
        embedding_sha256="cd" * 32,
        backend="insightface",
        model="buffalo_l/w600k_r50",
        bbox=[10, 20, 110, 140],
        det_score=0.9312,
    )


@pytest.fixture
def match_ref():
    return MatchRef(
        platform="bluesky",
        post_url="https://bsky.app/profile/someone.bsky.social/post/3abc",
        post_uri="at://did:plc:example/app.bsky.feed.post/3abc",
        author_handle="someone.bsky.social",
        author_did="did:plc:example",
        author_display_name="Someone",
        text="a post with a photo",
        image_url="https://cdn.bsky.app/img/feed_fullsize/plain/x/y",
        image_sha256="ef" * 32,
        created_at="2026-08-01T12:00:00Z",
        discovered_via="app.bsky.actor.searchActors:avatar",
    )


@pytest.fixture
def evidence(probe_ref, match_ref):
    return Evidence(
        probe=probe_ref,
        match=match_ref,
        similarity=0.7412,
        threshold=0.38,
        searched_at="2026-08-31T00:00:00Z",
        search_trace=[{"provider": "bluesky", "calls": []}],
    )


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """A Config wired to a throwaway chain state file, so tests never share state."""
    import sigil.chain.client as client_mod

    monkeypatch.setattr(client_mod, "STATE_PATH", tmp_path / "chain-state.json")
    c = Config()
    c.chain_backend = "local"
    c.subject_salt = "test-salt"
    return c


@pytest.fixture
def bluesky_post_fixture():
    return json.loads(
        """
        {
          "uri": "at://did:plc:example/app.bsky.feed.post/3xyz",
          "author": {"did": "did:plc:example", "handle": "a.bsky.social",
                     "displayName": "A"},
          "record": {"text": "hello", "createdAt": "2026-08-01T00:00:00Z"},
          "embed": {"images": [{"fullsize": "https://cdn/full.jpg",
                                "thumb": "https://cdn/thumb.jpg"}]}
        }
        """
    )


def pytest_collection_modifyitems(session, config, items):
    """Record how many offline tests this run collected.

    The README states the number, and it has gone stale more than once. Rather
    than remember to update it, `tests/test_readme.py` compares it against
    this - the same reason that file already checks the README's links.
    """
    config.offline_test_count = sum(
        1 for item in items if "network" not in item.keywords
    )
    config.collected_modules = len({item.module.__name__ for item in items})
