"""Web-layer behaviour: routing, input validation, and the tamper endpoint.

The pipeline itself is covered elsewhere; what matters here is that the server
rejects bad input with a message instead of a traceback, and that a browser can
never be handed a half-finished answer.
"""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from sigil.web import server as web


@pytest.fixture
def app(tmp_path, monkeypatch, cfg):
    # Depends on `cfg` so the chain-state path has exactly one owner. The server
    # builds its own Config per request, so it must resolve to the same chain
    # the test writes to - two fixtures each patching STATE_PATH would work only
    # by resolution order.
    monkeypatch.setattr(web, "EVIDENCE_PATH", tmp_path / "evidence.json")
    monkeypatch.setattr(web, "TAMPERED_PATH", tmp_path / "evidence.tampered.json")

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def get(url):
    with urllib.request.urlopen(url, timeout=20) as r:
        return r.status, r.read()


def post(url, payload=None):
    body = json.dumps(payload).encode() if payload is not None else b""
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_index_is_served(app):
    status, body = get(app + "/")
    assert status == 200
    assert b"<title>sigil" in body
    # The page must be self-contained: no build step and no third-party origin.
    assert b"cdn." not in body and b"https://fonts." not in body


def test_unknown_route_is_a_json_404(app):
    assert post(app + "/api/nope")[0] == 404


def test_run_requires_a_query(app):
    status, body = post(app + "/api/run", {"image_b64": "aGk=", "query": "  "})
    assert status == 400
    assert "query" in body["error"]


def test_run_requires_an_image(app):
    status, body = post(app + "/api/run", {"query": "someone", "image_b64": ""})
    assert status == 400
    assert "image" in body["error"]


def test_stream_for_an_unknown_job_is_404(app):
    status, body = None, None
    try:
        get(app + "/api/stream?job=deadbeef")
    except urllib.error.HTTPError as e:
        status, body = e.code, json.loads(e.read())
    assert status == 404
    assert body["error"] == "unknown job"


def test_tamper_without_a_bundle_explains_itself(app):
    status, body = post(app + "/api/tamper", {"field": "match.text"})
    assert status == 400
    assert "run the pipeline first" in body["error"]


def test_tamper_changes_the_hash_and_breaks_verification(app, evidence, cfg):
    """The demo's payload: one edited field, a different hash, a failed check."""
    from sigil.chain import ChainClient

    ChainClient(cfg).anchor(evidence)
    web.EVIDENCE_PATH.write_bytes(evidence.canonical_json())

    status, body = post(app + "/api/tamper", {"field": "match.text"})
    assert status == 200
    assert body["original_hash"] == evidence.evidence_hash_hex()
    assert body["tampered_hash"] != body["original_hash"]
    assert body["verification"]["anchored"] is False
    assert body["verification"]["ok"] is False


def test_verify_endpoint_confirms_an_anchored_bundle(app, evidence, cfg):
    from sigil.chain import ChainClient

    ChainClient(cfg).anchor(evidence)
    web.EVIDENCE_PATH.write_bytes(evidence.canonical_json())

    status, body = post(app + "/api/verify")
    assert status == 200
    assert body["anchored"] is True
    assert body["probe_matches"] is True
