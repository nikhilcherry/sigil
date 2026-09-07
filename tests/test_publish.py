"""Publishing the probe for the Lens arm: when it happens, and when it must not."""

import pytest

from sigil.config import Config
from sigil.search import publish


class _Response:
    def __init__(self, status_code=200, text=""):
        self.status_code = status_code
        self.text = text


class _Session:
    def __init__(self, response=None, boom=False):
        self.response = response
        self.boom = boom
        self.posts = []

    def post(self, url, **kw):
        self.posts.append((url, kw))
        if self.boom:
            raise RuntimeError("network")
        return self.response


def _cfg(**kw):
    cfg = Config()
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


# ------------------------------------------------------------------ when


def test_publishing_is_off_unless_asked_for():
    """It puts a photograph of somebody's face on a public URL.

    That is a disclosure about the subject, so it is not something a key in
    the environment should be able to turn on by itself.
    """
    assert publish.enabled_for(_cfg(serpapi_key="k")) is False


def test_publishing_is_skipped_when_nothing_would_use_the_url():
    """Lens is the only consumer; without its key the upload buys nothing."""
    assert publish.enabled_for(_cfg(publish_probe=True, serpapi_key=None)) is False


def test_publishing_happens_when_both_are_present():
    assert publish.enabled_for(_cfg(publish_probe=True, serpapi_key="k")) is True


# ------------------------------------------------------------------ how


def test_the_temporary_host_and_a_short_expiry_are_used(monkeypatch):
    """The permanent host is the obvious choice and leaves a face online forever."""
    session = _Session(_Response(200, "https://litter.example/abc.jpg\n"))
    monkeypatch.setattr(publish, "make_session", lambda: session)

    url = publish.publish_probe(b"jpeg-bytes", _cfg())

    assert url == "https://litter.example/abc.jpg"
    endpoint, kw = session.posts[0]
    assert endpoint == publish.ENDPOINT
    assert "litterbox" in endpoint
    assert kw["data"]["time"] == publish.EXPIRY == "1h"


@pytest.mark.parametrize("response", [
    _Response(500, "error"),
    _Response(200, "No file selected"),      # the API reports refusals as 200
    _Response(200, "   "),
])
def test_a_refusal_is_none_rather_than_a_url(monkeypatch, response):
    monkeypatch.setattr(publish, "make_session", lambda: _Session(response))
    assert publish.publish_probe(b"x", _cfg()) is None


def test_a_dead_host_costs_coverage_and_not_the_run(monkeypatch):
    monkeypatch.setattr(publish, "make_session", lambda: _Session(boom=True))
    assert publish.publish_probe(b"x", _cfg()) is None


def test_no_bytes_means_no_request(monkeypatch):
    session = _Session(_Response(200, "https://x/y"))
    monkeypatch.setattr(publish, "make_session", lambda: session)
    assert publish.publish_probe(b"", _cfg()) is None
    assert session.posts == []


# ------------------------------------------------------- inside the pipeline


def test_the_published_url_is_never_written_into_the_bundle(monkeypatch, cfg,
                                                            tmp_path):
    """It is a copy this tool made, not something the subject published.

    A bundle citing it would be citing sigil's own upload as evidence about
    somebody else.
    """
    import sigil.pipeline as pipeline
    from tests.conftest import EXAMPLE_PROBE

    cfg.publish_probe = True
    cfg.serpapi_key = "k"
    published = "https://litter.example/probe.jpg"
    monkeypatch.setattr(pipeline.publish, "publish_probe",
                        lambda b, c: published)
    monkeypatch.setattr(pipeline, "build_providers", lambda *a, **k: [])

    events = []
    result = pipeline.run_pipeline(str(EXAMPLE_PROBE), "someone", cfg,
                                   do_anchor=False, on_event=events.append)

    assert any(e["type"] == "published" and e["url"] == published for e in events)
    assert result.evidence is None


def test_an_https_probe_is_not_re_uploaded(monkeypatch, cfg):
    """It is already at a public URL; a second copy is a second disclosure."""
    import sigil.pipeline as pipeline
    from tests.conftest import EXAMPLE_PROBE

    cfg.publish_probe = True
    cfg.serpapi_key = "k"
    calls = []
    monkeypatch.setattr(pipeline.publish, "publish_probe",
                        lambda b, c: calls.append(b) or "https://x/y")
    monkeypatch.setattr(pipeline, "load_probe_bytes",
                        lambda s, c: (EXAMPLE_PROBE.read_bytes(), "https://given/probe.jpg"))
    monkeypatch.setattr(pipeline, "build_providers", lambda *a, **k: [])

    pipeline.run_pipeline("https://given/probe.jpg", "someone", cfg, do_anchor=False)
    assert calls == []
