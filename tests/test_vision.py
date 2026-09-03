"""Google Cloud Vision web detection: the open-web arm that needs no hosting.

Google only proposes here - it says "this image appears on these pages". Whether
the face in any of them is the probe's face is decided by the encoder, exactly
as with every other arm.
"""

import pytest

from sigil.config import Config
from sigil.search.vision import GoogleVisionProvider


class FakeVisionSession:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status
        self.calls = []

    def post(self, url, params=None, json=None, timeout=None):
        self.calls.append((url, params, json))
        payload, status = self.payload, self.status

        class R:
            status_code = status

            @staticmethod
            def json():
                if isinstance(payload, Exception):
                    raise payload
                return payload

        return R()


def _web(**kw):
    return {"responses": [{"webDetection": kw}]}


def _provider(monkeypatch, payload, status=200, key="vision-secret"):
    cfg = Config()
    cfg.google_vision_key = key
    session = FakeVisionSession(payload, status)
    monkeypatch.setattr("sigil.search.vision.make_session", lambda: session)
    return GoogleVisionProvider(cfg, b"probe-bytes"), session


# ------------------------------------------------------------- availability


def test_it_needs_a_key_and_an_image(monkeypatch):
    cfg = Config()
    cfg.google_vision_key = None
    assert GoogleVisionProvider.available_for(cfg, b"bytes") is False

    cfg.google_vision_key = "k"
    assert GoogleVisionProvider.available_for(cfg, None) is False
    assert GoogleVisionProvider.available_for(cfg, b"") is False
    assert GoogleVisionProvider.available_for(cfg, b"bytes") is True


def test_it_needs_no_public_url_unlike_the_lens_arm(monkeypatch):
    """The whole reason this arm exists: Lens matches on a URL, so the probe has
    to be hosted before it can be searched. This one takes the bytes."""
    provider, session = _provider(monkeypatch, _web())

    list(provider.candidates("ignored"))

    sent = session.calls[0][2]["requests"][0]
    assert "content" in sent["image"], "the probe was not sent inline"
    assert "source" not in sent["image"]
    assert sent["features"][0]["type"] == "WEB_DETECTION"


# ------------------------------------------------------------------ parsing


def test_page_matches_become_candidates_anchored_to_their_page(monkeypatch):
    provider, _ = _provider(monkeypatch, _web(pagesWithMatchingImages=[{
        "url": "https://news.example/story",
        "pageTitle": "A story about someone",
        "fullMatchingImages": [{"url": "https://cdn.example/a.jpg"}],
        "partialMatchingImages": [{"url": "https://cdn.example/b.jpg"}],
    }]))

    found = list(provider.candidates(""))

    assert [c.image_url for c in found] == ["https://cdn.example/a.jpg",
                                            "https://cdn.example/b.jpg"]
    assert all(c.post_url == "https://news.example/story" for c in found)
    assert all(c.platform == "news.example" for c in found)
    assert found[0].text == "A story about someone"
    assert found[0].discovered_via == "vision:pagesWithMatchingImages"


def test_loose_matches_cite_themselves_when_no_page_is_known(monkeypatch):
    """An image with no page context must not be attributed to a page it may
    not be on - the evidence has to point at something real."""
    provider, _ = _provider(monkeypatch, _web(
        fullMatchingImages=[{"url": "https://cdn.example/full.jpg"}],
        visuallySimilarImages=[{"url": "https://cdn.example/similar.jpg"}],
    ))

    found = list(provider.candidates(""))

    assert [c.discovered_via for c in found] == [
        "vision:fullMatchingImages", "vision:visuallySimilarImages"]
    assert all(c.post_url == c.image_url for c in found)


def test_page_matches_come_before_loose_ones(monkeypatch):
    """Ordering is a ranking hint: an image on an identified page is a stronger
    citation than a bare image URL."""
    provider, _ = _provider(monkeypatch, _web(
        visuallySimilarImages=[{"url": "https://cdn.example/weak.jpg"}],
        pagesWithMatchingImages=[{
            "url": "https://news.example/story",
            "fullMatchingImages": [{"url": "https://cdn.example/strong.jpg"}],
        }],
    ))

    found = [c.image_url for c in provider.candidates("")]

    assert found == ["https://cdn.example/strong.jpg", "https://cdn.example/weak.jpg"]


def test_entries_without_a_url_are_dropped(monkeypatch):
    provider, _ = _provider(monkeypatch, _web(
        pagesWithMatchingImages=[{"url": "https://news.example/x",
                                  "fullMatchingImages": [{}, {"url": "https://ok/1.jpg"}]}],
        fullMatchingImages=[{"score": 0.9}],
    ))

    assert [c.image_url for c in provider.candidates("")] == ["https://ok/1.jpg"]


# -------------------------------------------------------------------- safety


def test_the_api_key_never_reaches_the_evidence_trace(monkeypatch):
    """The trace is written into the bundle, which is published and hashed on
    chain. A key in there is a key leaked permanently."""
    provider, session = _provider(monkeypatch, _web(
        fullMatchingImages=[{"url": "https://cdn.example/a.jpg"}]))

    list(provider.candidates(""))

    assert session.calls[0][1]["key"] == "vision-secret", "the key must still be sent"
    assert "vision-secret" not in str(provider.trace.calls)


def test_the_trace_counts_what_came_back(monkeypatch):
    provider, _ = _provider(monkeypatch, _web(
        pagesWithMatchingImages=[{"url": "https://p/1"}],
        fullMatchingImages=[{"url": "https://i/1.jpg"}, {"url": "https://i/2.jpg"}],
    ))

    list(provider.candidates(""))

    assert provider.trace.calls[0]["endpoint"] == "vision.webDetection"
    assert provider.trace.calls[0]["results"] == 3


@pytest.mark.parametrize("payload,status", [
    ({}, 500),
    (ValueError("not json"), 200),
    ({"responses": []}, 200),
    ({"responses": [{}]}, 200),
    ({"responses": [{"error": {"message": "API key not valid"}}]}, 200),
])
def test_a_failed_or_empty_call_yields_nothing_rather_than_raising(
        monkeypatch, payload, status):
    """This arm is optional. Its outage, or a bad key, must not end the run -
    the zero-credential Bluesky arm still has to finish."""
    provider, _ = _provider(monkeypatch, payload, status=status)

    assert list(provider.candidates("")) == []


def test_it_joins_the_pipeline_only_when_a_key_is_present(monkeypatch):
    from sigil.pipeline import build_providers

    cfg = Config()
    cfg.google_vision_key = None
    assert [p.name for p in build_providers(cfg, None, b"bytes")] == ["bluesky"]

    cfg.google_vision_key = "k"
    names = [p.name for p in build_providers(cfg, None, b"bytes")]
    assert names == ["bluesky", "google-vision-web"], names


def test_the_probe_bytes_never_reach_the_evidence_trace(monkeypatch):
    """The bundle deliberately holds no image; the trace must not smuggle one.

    This arm is the only one that sends the probe itself to a third party, as
    base64 in the request body, so it is the only one that could.
    """
    provider, session = _provider(monkeypatch, _web())

    list(provider.candidates("ignored"))

    _url, _params, body = session.calls[0]
    assert body["requests"][0]["image"]["content"], "the probe must still be sent"
    recorded = str(provider.trace.calls)
    assert "content" not in recorded
    assert "probe-bytes" not in recorded
