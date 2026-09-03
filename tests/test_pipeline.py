"""The end-to-end claim, offline.

The live suite proves the real APIs still answer; this proves the orchestration
itself - scan, identify, search, match, anchor, re-verify - without depending on
a third party being up.
"""

import numpy as np
import pytest

from sigil.chain import ChainClient
from sigil.config import Config
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
        monkeypatch.setattr(pipe, "build_providers", lambda cfg, url, blob=None: [provider])
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



def test_the_probe_is_encoded_a_second_time_to_verify_rather_than_echoed(
    wired, cfg, monkeypatch
):
    """The row used to compare the bundle's own digest to itself.

    ``result.evidence.probe`` *is* the ProbeRef the pipeline built, so handing
    its digest to the check that exists to test a photograph against it made
    "probe re-encodes" a tautology which passed on every run.
    """
    import sigil.pipeline as pipe

    wired(["https://x/same.jpg"])
    real_scan = pipe.scan_probe
    seen = []

    def counted(image_bytes, config):
        out = real_scan(image_bytes, config)
        seen.append(out[1].embedding_sha256)
        return out

    monkeypatch.setattr(pipe, "scan_probe", counted)
    result = run_pipeline(str(EXAMPLE_PROBE), "who", cfg, do_anchor=True)

    assert len(seen) == 2, f"the probe was encoded {len(seen)} time(s), not twice"
    assert seen[0] == seen[1], "encoding one probe twice gave two different answers"
    assert result.verification.probe_matches is True


# ------------------------------------------------ where the probe comes from


def test_a_probe_can_be_an_https_url_not_only_a_path(monkeypatch):
    """A documented capability of `sigil run`, and the reason probe_url exists.

    The Lens arm matches on a URL rather than an upload, so it can only run at
    all when the probe arrived as one. That makes this path load-bearing rather
    than a convenience.
    """
    import sigil.pipeline as pipe

    blob = EXAMPLE_PROBE.read_bytes()

    class Response:
        content = blob
        raise_for_status = staticmethod(lambda: None)

    got = {}

    def fake_get(url, timeout=None, headers=None):
        got["url"] = url
        got["headers"] = headers
        return Response()

    monkeypatch.setattr(pipe.requests, "get", fake_get)
    data, url = pipe.load_probe_bytes("https://example.com/face.jpg", Config())

    assert data == blob
    # The URL is handed back so the Lens arm can be offered the same address.
    assert url == "https://example.com/face.jpg"
    assert got["url"] == "https://example.com/face.jpg"
    assert got["headers"] == {"User-Agent": "sigil"}


def test_a_local_path_reports_no_public_url(monkeypatch):
    """The contrast that makes the returned URL meaningful."""
    import sigil.pipeline as pipe

    data, url = pipe.load_probe_bytes(str(EXAMPLE_PROBE), Config())
    assert data == EXAMPLE_PROBE.read_bytes()
    assert url is None


def test_a_missing_probe_path_says_so_rather_than_raising_oserror():
    import sigil.pipeline as pipe

    with pytest.raises(PipelineError, match="probe image not found"):
        pipe.load_probe_bytes("/no/such/face.jpg", Config())


def test_a_url_probe_is_what_unlocks_the_lens_arm(monkeypatch):
    """Threading probe_url through the pipeline has exactly one purpose."""
    import sigil.pipeline as pipe

    cfg = Config()
    cfg.serpapi_key = "k"

    assert [p.name for p in pipe.build_providers(cfg, None, b"bytes")] == ["bluesky"]
    with_url = pipe.build_providers(cfg, "https://example.com/face.jpg", b"bytes")
    assert [p.name for p in with_url] == ["bluesky", "serpapi-google-lens"]


def test_a_probe_with_no_detectable_face_is_refused_with_advice():
    """The most common way a run fails before it starts."""
    import cv2
    import numpy as np

    import sigil.pipeline as pipe

    ok, buf = cv2.imencode(".png", np.full((80, 80, 3), 200, dtype=np.uint8))
    assert ok
    with pytest.raises(PipelineError, match="no face detected"):
        pipe.scan_probe(buf.tobytes(), Config())


def test_an_undecodable_probe_is_refused_before_the_encoder():
    import sigil.pipeline as pipe

    with pytest.raises(PipelineError, match="could not be decoded"):
        pipe.scan_probe(b"this is not an image", Config())
