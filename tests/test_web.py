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


def test_verify_endpoint_confirms_an_anchored_bundle(app, evidence, cfg,
                                                     monkeypatch):
    from sigil.chain import ChainClient

    monkeypatch.setattr(web, "LAST_PROBE", None)
    ChainClient(cfg).anchor(evidence)
    web.EVIDENCE_PATH.write_bytes(evidence.canonical_json())

    status, body = post(app + "/api/verify")
    assert status == 200
    assert body["anchored"] is True
    # With no probe image on disk the re-encode cannot be attempted, and an
    # unrun check reads as not-run. This used to say True: the endpoint passed
    # the bundle's own embedding digest back into the check that exists to
    # test a photograph against it, so the row read PASS on every run while
    # proving nothing at all.
    assert body["probe_matches"] is None


def _real_probe_evidence(match_ref):
    """An evidence bundle whose ProbeRef really came from the committed image."""
    from sigil.config import Config
    from sigil.evidence import Evidence
    from sigil.pipeline import scan_probe
    from tests.conftest import EXAMPLE_PROBE

    blob = EXAMPLE_PROBE.read_bytes()
    _, ref, _ = scan_probe(blob, Config())
    return blob, Evidence(probe=ref, match=match_ref, similarity=0.7596,
                          threshold=0.38, searched_at="2026-09-03T00:00:00Z")


def test_the_verify_endpoint_re_encodes_the_probe_rather_than_echoing_it(
    app, cfg, match_ref, monkeypatch
):
    """The real thing: a probe on disk is scanned, not taken on the bundle's word."""
    from sigil.chain import ChainClient
    from tests.conftest import EXAMPLE_PROBE

    blob, ev = _real_probe_evidence(match_ref)
    ChainClient(cfg).anchor(ev)
    web.EVIDENCE_PATH.write_bytes(ev.canonical_json())
    monkeypatch.setattr(web, "LAST_PROBE", EXAMPLE_PROBE)

    body = post(app + "/api/verify")[1]
    assert body["probe_matches"] is True, body.get("notes")
    assert blob  # the bundle really is built from those bytes


def test_a_probe_from_a_different_run_is_not_used_to_verify_this_bundle(
    app, cfg, evidence, monkeypatch
):
    """A run that finds nothing leaves the previous bundle in place.

    Verifying that bundle against the newer upload would fail for a reason
    having nothing to do with either of them, so the probe is matched by
    digest and skipped when it does not belong.
    """
    from sigil.chain import ChainClient
    from tests.conftest import EXAMPLE_PROBE

    ChainClient(cfg).anchor(evidence)          # synthetic ProbeRef digests
    web.EVIDENCE_PATH.write_bytes(evidence.canonical_json())
    monkeypatch.setattr(web, "LAST_PROBE", EXAMPLE_PROBE)

    body = post(app + "/api/verify")[1]
    assert body["probe_matches"] is None
    assert body["claim_reproduces"] is None



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


class _FakeHTTPD:
    """Stands in for ThreadingHTTPServer so serve() can return."""

    instances = []

    def __init__(self, address, handler):
        self.address = address
        self.handler = handler
        self.served = False
        self.closed = False
        _FakeHTTPD.instances.append(self)

    def serve_forever(self):
        self.served = True

    def server_close(self):
        self.closed = True


@pytest.fixture
def fake_httpd(monkeypatch):
    _FakeHTTPD.instances = []
    monkeypatch.setattr(web, "ThreadingHTTPServer", _FakeHTTPD)
    return _FakeHTTPD


def test_serve_binds_where_it_was_told(fake_httpd, monkeypatch):
    """It binds to localhost by default on purpose - this tool searches for
    people by face and has no authentication."""
    opened = []
    monkeypatch.setattr("webbrowser.open", opened.append)

    web.serve(host="127.0.0.1", port=9123, open_browser=False)

    httpd = fake_httpd.instances[0]
    assert httpd.address == ("127.0.0.1", 9123)
    assert httpd.served and httpd.closed, "the socket was not closed on the way out"
    assert opened == []


def test_serve_closes_the_socket_even_on_interrupt(fake_httpd, monkeypatch):
    """Ctrl-C is the documented way to stop it, so it is the path that must not
    leak the listening socket."""
    monkeypatch.setattr("webbrowser.open", lambda url: None)

    def interrupted(self):
        raise KeyboardInterrupt

    monkeypatch.setattr(_FakeHTTPD, "serve_forever", interrupted)

    web.serve(host="127.0.0.1", port=9124, open_browser=False)

    assert fake_httpd.instances[0].closed


def test_serve_opens_a_browser_only_when_asked(fake_httpd, monkeypatch):
    opened = []
    monkeypatch.setattr("webbrowser.open", opened.append)
    # The real one fires on a Timer; run it inline so the test stays sequential.
    monkeypatch.setattr(web.threading, "Timer",
                        lambda delay, fn: type("T", (), {"start": staticmethod(fn)})())

    web.serve(host="127.0.0.1", port=9125, open_browser=True)

    assert opened == ["http://127.0.0.1:9125"]


@pytest.mark.parametrize("mime", ["../../etc/passwd", "x/../../y", "",
                                  "image/jpeg\x00.sh", "text/html", "image/png"])
def test_a_hostile_mime_cannot_steer_where_the_probe_is_written(app, tmp_path,
                                                                monkeypatch, mime):
    """The probe filename is derived from a client-supplied MIME type, so it has
    to stay inside the artifacts directory whatever the client claims.

    The pipeline is stubbed out: what is under test is the path derivation in
    _start_run, and running a real search six times to check a filename would
    be minutes of network and inference for nothing.
    """
    import base64

    from tests.conftest import EXAMPLE_PROBE

    probes = tmp_path / "probes"
    probes.mkdir()
    monkeypatch.setattr(web, "ARTIFACTS_DIR", probes)
    # A no-match result: the job runs to completion and writes no bundle, so
    # whatever lands in the directory is the probe and nothing else.
    monkeypatch.setattr(web, "run_pipeline",
                        lambda *a, **k: type("R", (), {"evidence": None})())

    status, body = post(f"{app}/api/run", {
        "image_b64": base64.b64encode(EXAMPLE_PROBE.read_bytes()).decode(),
        "query": "someone",
        "mime": mime,
    })
    assert status == 200, body

    _read_stream(f"{app}/api/stream?job={body['job']}", limit=20)

    written = [p for p in probes.rglob("*") if p.is_file()]
    assert written, "the probe was not written at all"
    for path in written:
        assert path.resolve().is_relative_to(probes.resolve()), path
        assert path.parent.resolve() == probes.resolve(), f"escaped into {path.parent}"
        assert path.name.startswith("probe"), path.name


def test_every_check_the_ui_renders_is_a_key_the_server_actually_sends():
    """A row the payload never fills reads as NOT RUN forever, silently.

    The two lists are in different files and different languages, so nothing
    else would notice one of them gaining a field the other did not.
    """
    import re
    from pathlib import Path

    from sigil.chain.client import Verification
    from sigil.pipeline import verification_payload

    html = (Path(__file__).resolve().parent.parent
            / "sigil" / "web" / "index.html").read_text()
    start = html.index("const CHECKS")
    # To the "];" that closes the array - not the first "]", which closes the
    # first pair and would leave this test asserting one key and passing
    # whatever the rest of the list said.
    block = html[start:html.index("];", start)]
    rendered = set(re.findall(r'\["(\w+)"', block))
    assert len(rendered) >= 5, f"CHECKS list parsed as only {rendered}"

    sent = set(verification_payload(Verification(evidence_hash="0x", anchored=True)))
    missing = rendered - sent
    assert not missing, f"the UI renders checks the server never sends: {missing}"


def test_a_probe_that_no_longer_yields_a_face_fails_the_check_not_skips_it(
    app, cfg, match_ref, monkeypatch, tmp_path
):
    """A check that cannot be completed must not read as one that passed.

    The stored probe belongs to the bundle - the digests match - so the check
    was requested. It just cannot be answered, and "cannot answer" is a
    failure here rather than a not-run.
    """
    from sigil.chain import ChainClient
    from sigil.pipeline import PipelineError
    from tests.conftest import EXAMPLE_PROBE

    _blob, ev = _real_probe_evidence(match_ref)
    ChainClient(cfg).anchor(ev)
    web.EVIDENCE_PATH.write_bytes(ev.canonical_json())
    monkeypatch.setattr(web, "LAST_PROBE", EXAMPLE_PROBE)

    def no_face(image_bytes, config):
        raise PipelineError("no face detected in the probe image")

    monkeypatch.setattr("sigil.pipeline.scan_probe", no_face)

    body = post(app + "/api/verify")[1]
    assert body["probe_matches"] is False
    assert body["ok"] is False


def get_json(url):
    """GET returning (status, parsed body), including for error responses.

    The plain `get` helper hands back raw bytes and lets an HTTPError escape,
    which is what most tests here want; these want the error body.
    """
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_an_unknown_route_is_a_json_404_not_a_traceback(app):
    assert get_json(app + "/api/nope") == (404, {"error": "not found"})
    assert post(app + "/api/nope", {"x": 1}) == (404, {"error": "not found"})


def test_the_evidence_endpoint_serves_the_stored_bundle(app, evidence):
    web.EVIDENCE_PATH.write_bytes(evidence.canonical_json())
    status, body = get_json(app + "/api/evidence")
    assert status == 200
    assert body["match"]["platform"] == evidence.match.platform


def test_the_evidence_endpoint_errors_cleanly_when_there_is_none(app):
    if web.EVIDENCE_PATH.exists():
        web.EVIDENCE_PATH.unlink()
    status, body = get_json(app + "/api/evidence")
    assert status == 400
    assert "error" in body and "Traceback" not in body["error"]


# --------------------------------------------- untrusted text in the page

# Fields whose value is written by someone other than the operator: a Bluesky
# display name is set by its owner, an identity-index label comes from
# Wikidata, a post URL comes from whichever arm found it. Each of these was
# interpolated straight into innerHTML, so a display name of
# `<img src=x onerror=...>` executed in a page served from localhost with
# /api/run, /api/tamper and /api/evidence one fetch away. Confirmed executable
# in a real browser before the fix, and confirmed inert after it.
UNTRUSTED_IN_PAGE = (
    "author_display_name",
    "author_handle",
    "h.name",
    "h.source",
    "x.handle",
    "post_url",
    "e.provider",
    "e.error",
    "e.tx_hash",
)


def _interpolations(html: str) -> list[str]:
    """Every `${...}` in the page, non-greedy so nested braces do not swallow."""
    import re

    return re.findall(r"\$\{([^{}]*)\}", html)


def test_every_untrusted_field_is_escaped_before_it_reaches_the_page():
    """A browser test cannot run here, so this guards the shape instead.

    It is a lint rather than a proof: it checks that no interpolation mentions
    an attacker-controlled field without passing it through esc(). That is the
    property that was violated, and the one a future edit is most likely to
    violate again.
    """
    from pathlib import Path

    html = (Path(__file__).resolve().parent.parent
            / "sigil" / "web" / "index.html").read_text()
    assert "function esc(" in html, "the escaping helper is gone"

    offenders = []
    for expr in _interpolations(html):
        for field in UNTRUSTED_IN_PAGE:
            if field in expr and "esc(" not in expr:
                offenders.append(expr.strip())
    assert not offenders, f"unescaped untrusted fields in the page: {offenders}"


# Every interpolation that reaches innerHTML without esc() or safeUrl(),
# reviewed one by one. The allowlist above names fields known to be hostile;
# this names the exceptions instead, which is the direction that survives
# somebody rendering a field nobody thought to add to a list. That is not
# hypothetical - `r.after` is the tail of `match.text`, which is a Bluesky
# account's own description, and it was interpolated raw for exactly that
# reason: no list mentioned it.
REVIEWED_SAFE_INTERPOLATIONS = {
    # Numeric formatting. Produces digits, a dot and possibly a minus sign.
    "e.best.toFixed(4)",
    "e.threshold.toFixed(3)",
    "h.similarity.toFixed(4)",
    "x.similarity.toFixed(3)",
    "e.examined",
    # Ternaries over string literals this file owns end to end.
    'copy ? "copy" : ""',
    'h.accepted ? "ok" : ""',
    'mine ? "anchored" : ""',
    'x.hit ? "hit" : ""',
    'mine ? "\u25c0 anchored" : (copy ? "same photo" : "different photo")',
    "cls",
    "txt",
    # The second element of each CHECKS pair, a literal in this same file.
    "label",
    # keccak hex this code computed, and one character sliced out of it.
    "a",
    "b",
    "ch",
    # Escaped per item already; the join is the markup, deliberately.
    'v.notes.map(esc).join("<br>")',
}


def _html_interpolations(html: str) -> list[str]:
    """Interpolations inside template literals that build markup."""
    import re

    out = []
    for lit in re.findall(r"`(?:[^`\\]|\\.)*`", html, re.S):
        if "<" not in lit:
            continue
        for expr in re.findall(r"\$\{(.*?)\}", lit, re.S):
            out.append(" ".join(expr.split()))
    return out


def test_no_interpolation_reaches_the_page_unescaped_by_default():
    """Deny by default, so a new field is escaped or explicitly justified.

    The earlier lint asked "is this one of the fields we know to be hostile?",
    which is only ever as good as the list. This asks the opposite question -
    "has this been escaped, or written down as safe and why?" - so the burden
    falls on whoever adds the interpolation rather than on this file having
    predicted its name.

    It caught a real one: the tamper preview rendered `${r.field}` and
    `${r.after}` raw, and `after` is the last 60 characters of `match.text` -
    a Bluesky account's own description, set by its owner. The demo's Tamper
    button posts `field: "match.text"` with no input from the operator, so one
    click on the page rendered a third party's text as HTML, in the page that
    has /api/run and /api/tamper one fetch away.
    """
    from pathlib import Path

    html = (Path(__file__).resolve().parent.parent
            / "sigil" / "web" / "index.html").read_text()

    offenders = sorted({
        expr for expr in _html_interpolations(html)
        if "esc(" not in expr
        and "safeUrl(" not in expr
        and expr not in REVIEWED_SAFE_INTERPOLATIONS
    })
    assert not offenders, (
        "interpolation reaches innerHTML unescaped and is not reviewed: "
        f"{offenders}. Wrap it in esc(), or add it to "
        "REVIEWED_SAFE_INTERPOLATIONS with the reason it cannot carry markup."
    )


def test_anything_becoming_an_href_goes_through_the_url_allowlist():
    """`javascript:` in an href is the same hole with an extra click."""
    import re
    from pathlib import Path

    html = (Path(__file__).resolve().parent.parent
            / "sigil" / "web" / "index.html").read_text()
    assert "function safeUrl(" in html, "the URL allowlist is gone"

    hrefs = re.findall(r'href="(\$\{[^"]*\})"', html)
    assert hrefs, "no interpolated hrefs found - did the page change shape?"
    for h in hrefs:
        assert "safeUrl(" in h, f"href built without the allowlist: {h}"


def test_the_url_allowlist_only_admits_http_and_https():
    """Asserted against the helper's own source, since JS cannot run here."""
    from pathlib import Path

    html = (Path(__file__).resolve().parent.parent
            / "sigil" / "web" / "index.html").read_text()
    body = html[html.index("function safeUrl("):]
    body = body[:body.index("\n}")]
    assert "https?" in body and 'return' in body
    assert '"#"' in body, "a rejected URL must become inert, not pass through"


# ------------------------------------------------- runs nobody listened to


def test_a_run_nobody_streamed_is_not_retained_forever(app, monkeypatch):
    """`_stream`'s finally cannot fire if no client ever connects.

    A client that POSTs /api/run and never opens the stream leaves the job,
    and every event the pipeline queued on it, in the table for the life of
    the process. The abandoned-mid-stream case was already handled; this is
    the never-connected one.
    """
    import base64

    from tests.conftest import EXAMPLE_PROBE

    monkeypatch.setattr(web, "_run_job", lambda job, *a, **kw: job.finish())

    body = {"image_b64": base64.b64encode(EXAMPLE_PROBE.read_bytes()).decode(),
            "query": "q"}
    ids = []
    for _ in range(3):
        status, payload = post(app + "/api/run", body)
        assert status == 200, payload
        ids.append(payload["job"])
        # The stub finishes synchronously, so the job is done and undrained.

    # Each new run reaps the finished ones, so the table never accumulates.
    assert len(web.JOBS) <= 1, f"retained {len(web.JOBS)} finished jobs"
    assert len(set(ids)) == 3, "job ids should be distinct"


def test_reaping_leaves_a_run_that_is_still_going(monkeypatch):
    """An unfinished run is someone's live stream; it must survive the sweep."""
    monkeypatch.setattr(web, "JOBS", {})

    finished, running = web.Job(), web.Job()
    finished.finish()
    web.JOBS[finished.id] = finished
    web.JOBS[running.id] = running

    assert web._reap_finished_jobs() == 1
    assert list(web.JOBS) == [running.id]


def test_reaping_an_empty_table_is_a_no_op(monkeypatch):
    monkeypatch.setattr(web, "JOBS", {})
    assert web._reap_finished_jobs() == 0
