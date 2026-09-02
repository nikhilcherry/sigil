"""Web-layer behaviour: routing, input validation, and the tamper endpoint.

The pipeline itself is covered elsewhere; what matters here is that the server
rejects bad input with a message instead of a traceback, and that a browser can
never be handed a half-finished answer.
"""

import json
import threading
import time
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


def test_run_accepts_an_empty_query(app):
    """An empty query is not an error - it asks the pipeline to identify the face."""
    status, body = post(app + "/api/run", {"image_b64": "aGk=", "query": "  "})
    assert status == 200
    assert "job" in body


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


def _read_stream(url, limit=10, deadline=90):
    """Collect SSE frames until the end event, `limit` frames, or the deadline.

    The deadline is the point. A stream that never terminates is precisely the
    failure being guarded against, and the server sends a keepalive comment
    every 30s - so without a wall-clock bound a regression would hang the suite
    instead of failing it.
    """
    frames = []
    started = time.monotonic()
    with urllib.request.urlopen(url, timeout=30) as r:
        assert r.headers["Content-Type"] == "text/event-stream"
        for raw in r:
            if time.monotonic() - started > deadline:
                raise AssertionError(
                    f"stream did not end within {deadline}s; got {len(frames)} frames"
                )
            line = raw.decode().rstrip("\n")
            if not line:
                continue
            frames.append(line)
            if line.startswith("event: end") or len(frames) >= limit:
                break
    return frames


def test_the_event_stream_delivers_events_then_ends(app):
    """The browser drives the whole UI off this stream; if it never terminates,
    the page sits on a spinner after a finished run."""
    job = web.Job()
    web.JOBS[job.id] = job
    job.emit({"type": "stage", "stage": "scan", "status": "start"})
    job.emit({"type": "probe", "backend": "insightface"})
    job.finish()

    frames = _read_stream(f"{app}/api/stream?job={job.id}")

    payloads = [json.loads(f[len("data: "):]) for f in frames if f.startswith("data: ")]
    assert payloads[0]["type"] == "stage"
    assert payloads[1]["backend"] == "insightface"
    assert any(f.startswith("event: end") for f in frames)


def test_a_finished_job_is_forgotten_once_streamed(app):
    """JOBS would otherwise grow for the life of the process, holding every
    event of every run in memory."""
    job = web.Job()
    web.JOBS[job.id] = job
    job.finish()

    _read_stream(f"{app}/api/stream?job={job.id}")

    # The handler pops the job on its own thread, which can lag the client
    # finishing its read - so wait for it rather than sampling once.
    deadline = time.monotonic() + 10
    while job.id in web.JOBS and time.monotonic() < deadline:
        time.sleep(0.01)
    assert job.id not in web.JOBS


def test_the_stream_closes_the_connection_rather_than_keeping_it_alive(app):
    """An event stream has no Content-Length, so a keep-alive connection leaves
    the client unable to tell "finished" from "still thinking"."""
    job = web.Job()
    web.JOBS[job.id] = job
    job.finish()

    with urllib.request.urlopen(f"{app}/api/stream?job={job.id}", timeout=30) as r:
        assert r.headers["Connection"] == "close"
        assert r.headers.get("Content-Length") is None


def test_a_bad_option_ends_the_stream_with_an_error_not_a_hang(app, monkeypatch):
    """The page drives everything off this stream, so anything that escapes the
    job function does not show as an error - it leaves the UI on a spinner
    forever. Config() itself can raise, since a malformed threshold is refused
    rather than defaulted."""
    import base64

    from tests.conftest import EXAMPLE_PROBE

    status, body = post(f"{app}/api/run", {
        "image_b64": base64.b64encode(EXAMPLE_PROBE.read_bytes()).decode(),
        "query": "someone",
        "threshold": "not-a-number",
    })
    assert status == 200, body

    frames = _read_stream(f"{app}/api/stream?job={body['job']}", limit=40)

    assert any(f.startswith("event: end") for f in frames), "the stream never ended"
    errors = [json.loads(f[len("data: "):]) for f in frames if f.startswith("data: ")]
    assert any(e.get("type") == "error" for e in errors), errors


def test_a_bad_max_images_is_also_reported_rather_than_hanging(app):
    import base64

    from tests.conftest import EXAMPLE_PROBE

    status, body = post(f"{app}/api/run", {
        "image_b64": base64.b64encode(EXAMPLE_PROBE.read_bytes()).decode(),
        "query": "someone",
        "max_images": "twenty",
    })
    assert status == 200, body

    frames = _read_stream(f"{app}/api/stream?job={body['job']}", limit=40)

    assert any(f.startswith("event: end") for f in frames), "the stream never ended"


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_a_viewer_who_navigates_away_does_not_leak_the_job(app):
    """Closing the tab mid-stream raises out of the write. The job and every
    event still queued on it must not be retained for the life of the process.

    The handler ends its thread with SystemExit, which threading.excepthook
    ignores by design - so nothing is printed in production. Only pytest's
    thread-exception hook notices, hence the filter.
    """
    job = web.Job()
    web.JOBS[job.id] = job
    for i in range(2000):
        job.emit({"type": "progress", "examined": i})

    # Read a little, then hang up without waiting for the end event.
    r = urllib.request.urlopen(f"{app}/api/stream?job={job.id}", timeout=30)
    for _ in range(3):
        r.readline()
    r.close()

    deadline = time.monotonic() + 15
    while job.id in web.JOBS and time.monotonic() < deadline:
        time.sleep(0.05)
    assert job.id not in web.JOBS, "the abandoned job was never dropped"
