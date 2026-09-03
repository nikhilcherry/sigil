"""A local web UI for the pipeline, on the standard library alone.

No framework, no build step, no CDN - the page is one file served from disk and
the live pipeline events reach it over Server-Sent Events. The image arrives as
base64 inside a JSON body rather than as multipart, because ``cgi.FieldStorage``
was removed in Python 3.13 and hand-rolling a multipart parser to save one
base64 encode is a bad trade.

Binds to 127.0.0.1 by default. This tool searches for people by face; it has no
authentication and no business being reachable from off-box.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import queue
import threading
import traceback
import uuid
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from ..chain import ChainClient
from ..config import ARTIFACTS_DIR, Config, ensure_dirs
from ..evidence import Evidence, alter_field
from ..pipeline import PipelineError, run_pipeline, verification_payload

HERE = Path(__file__).parent
INDEX = HERE / "index.html"
EVIDENCE_PATH = ARTIFACTS_DIR / "evidence.json"
TAMPERED_PATH = ARTIFACTS_DIR / "evidence.tampered.json"

MAX_BODY = 20 * 1024 * 1024
SENTINEL = object()


class Job:
    """One pipeline run, streaming its events to whoever is listening."""

    def __init__(self) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.events: queue.Queue = queue.Queue()
        self.done = threading.Event()

    def emit(self, event: dict[str, Any]) -> None:
        self.events.put(event)

    def finish(self) -> None:
        self.events.put(SENTINEL)
        self.done.set()


JOBS: dict[str, Job] = {}

# Where the last uploaded probe was written. Re-verification needs the actual
# image, not the bundle's description of it, and the browser only sends it once.
LAST_PROBE: Path | None = None


def _run_job(job: Job, probe_path: Path, query: str, opts: dict[str, Any]) -> None:
    # Everything is inside the try, including building the config. The browser
    # drives the whole UI off this job's event stream and the stream only ends
    # when finish() runs in the finally - so anything that escapes this
    # function does not surface as an error, it hangs the page on a spinner.
    # Config() itself can raise: a malformed SIGIL_THRESHOLD is refused rather
    # than defaulted, and the options below come straight from the request.
    try:
        cfg = Config()
        for key in ("face_backend", "chain_backend"):
            if opts.get(key):
                setattr(cfg, key, opts[key])
        if opts.get("max_images"):
            cfg.max_images = int(opts["max_images"])
        if opts.get("threshold"):
            cfg.threshold = float(opts["threshold"])

        result = run_pipeline(str(probe_path), query, cfg, do_anchor=True,
                              on_event=job.emit)
        if result.evidence is not None:
            EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
            EVIDENCE_PATH.write_bytes(result.evidence.canonical_json())
    except PipelineError as exc:
        job.emit({"type": "error", "message": str(exc)})
    except Exception as exc:  # noqa: BLE001 - surface it in the UI, never hang the stream
        job.emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        traceback.print_exc()
    finally:
        job.finish()


def _tamper(field: str) -> dict[str, Any]:
    """Alter one field of the stored bundle and re-verify it, for the demo."""
    if not EVIDENCE_PATH.exists():
        raise FileNotFoundError("no evidence bundle yet - run the pipeline first")

    original = Evidence.from_dict(json.loads(EVIDENCE_PATH.read_text()))
    data = json.loads(original.canonical_json())
    before, after = alter_field(data, field)
    tampered = Evidence.from_dict(data)
    TAMPERED_PATH.write_bytes(tampered.canonical_json())

    client = ChainClient(Config())
    return {
        "field": field,
        "before": before if not isinstance(before, str) else before[-60:],
        "after": after if not isinstance(after, str) else after[-60:],
        "original_hash": original.evidence_hash_hex(),
        "tampered_hash": tampered.evidence_hash_hex(),
        "verification": verification_payload(client.verify(tampered)),
    }


def _probe_for(evidence: Evidence) -> bytes | None:
    """The bytes of the probe this bundle was built from, if they are still here.

    Matched by digest rather than assumed. The last upload is not necessarily
    the one behind the stored bundle - a second run that finds no match leaves
    the previous bundle in place - and re-verifying one probe against another
    bundle would fail for a reason that has nothing to do with either.
    """
    from ..evidence import sha256_hex

    if LAST_PROBE is None or not LAST_PROBE.exists():
        return None
    blob = LAST_PROBE.read_bytes()
    return blob if sha256_hex(blob) == evidence.probe.image_sha256 else None


def _reverify() -> dict[str, Any]:
    from ..config import Config as _Config
    from ..pipeline import scan_probe

    ev = Evidence.from_dict(json.loads(EVIDENCE_PATH.read_text()))
    cfg = _Config()
    client = ChainClient(cfg)

    # Re-encode the probe from its pixels. Passing ev.probe.embedding_sha256
    # here - which is what this used to do - compares the bundle to itself, so
    # the "probe re-encodes" row read PASS on every run while proving nothing.
    probe_bytes = _probe_for(ev)
    digest = None
    if probe_bytes is not None:
        try:
            _, ref, _ = scan_probe(probe_bytes, cfg)
            digest = ref.embedding_sha256
        except PipelineError:
            # The face that was found once is not found now: report that as a
            # failed check rather than as no check at all.
            digest = ""

    payload = verification_payload(
        client.verify(ev, probe_embedding_sha256=digest, recheck_source=True,
                      probe_image_bytes=probe_bytes)
    )
    payload["evidence"] = {"match": asdict(ev.match), "similarity": ev.similarity}
    return payload


def _chain_state() -> dict[str, Any]:
    cfg = Config()
    client = ChainClient(cfg)
    return {
        "backend": cfg.chain_backend,
        "chain_id": client.chain_id,
        "contract": client.ensure_deployed(),
        "submitter": client.address,
        "total": client.total_anchored(),
        "has_evidence": EVIDENCE_PATH.exists(),
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "sigil"

    def log_message(self, fmt: str, *args) -> None:  # quieter than the default
        if self.path.startswith("/api/stream"):
            return
        print(f"  {self.command} {self.path}")

    # -- helpers ---------------------------------------------------------

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict[str, Any]) -> None:
        self._send(code, json.dumps(payload).encode(), "application/json")

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_BODY:
            raise ValueError(f"request body exceeds {MAX_BODY} bytes")
        return json.loads(self.rfile.read(length) or b"{}")

    # -- routes ----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/state":
            self._guard(lambda: self._json(200, _chain_state()))
        elif path == "/api/stream":
            self._stream()
        elif path == "/api/evidence":
            self._guard(
                lambda: self._send(200, EVIDENCE_PATH.read_bytes(), "application/json")
            )
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        if path == "/api/run":
            self._guard(self._start_run)
        elif path == "/api/tamper":
            self._guard(lambda: self._json(200, _tamper(self._body().get("field", "match.text"))))
        elif path == "/api/verify":
            self._guard(lambda: self._json(200, _reverify()))
        else:
            self._json(404, {"error": "not found"})

    def _guard(self, fn) -> None:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            self._json(400, {"error": f"{type(exc).__name__}: {exc}"})

    def _start_run(self) -> None:
        body = self._body()
        # An empty query is legitimate: it means "identify the face first".
        query = (body.get("query") or "").strip()

        raw = body.get("image_b64") or ""
        if "," in raw[:64]:  # strip a data: URI prefix if the browser sent one
            raw = raw.split(",", 1)[1]
        blob = base64.b64decode(raw)
        if not blob:
            raise ValueError("no image supplied")

        ensure_dirs()
        suffix = mimetypes.guess_extension(body.get("mime", "image/jpeg")) or ".jpg"
        probe_path = ARTIFACTS_DIR / f"probe{suffix}"
        probe_path.write_bytes(blob)
        global LAST_PROBE
        LAST_PROBE = probe_path

        job = Job()
        JOBS[job.id] = job
        threading.Thread(
            target=_run_job, args=(job, probe_path, query, body), daemon=True
        ).start()
        self._json(200, {"job": job.id})

    def _stream(self) -> None:
        job_id = self.path.partition("?job=")[2]
        job = JOBS.get(job_id)
        if job is None:
            self._json(404, {"error": "unknown job"})
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        # An event stream has no Content-Length, so on a keep-alive connection
        # the client cannot tell the difference between "finished" and "still
        # thinking" and hangs after the last event. Close it explicitly.
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()

        try:
            while True:
                try:
                    event = job.events.get(timeout=30)
                except queue.Empty:
                    # A comment frame keeps proxies and impatient clients from
                    # dropping a stream during a long search.
                    self._write_raw(b": keepalive\n\n")
                    continue
                if event is SENTINEL:
                    self._write_raw(b"event: end\ndata: {}\n\n")
                    break
                self._write_raw(f"data: {json.dumps(event)}\n\n".encode())
        finally:
            # In a finally because a viewer navigating away mid-stream raises
            # out of _write_raw. Without this the job, and every event still
            # queued on it, would be retained for the life of the process.
            JOBS.pop(job_id, None)

    def _write_raw(self, data: bytes) -> None:
        try:
            self.wfile.write(data)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            raise SystemExit from None  # client navigated away; end this thread


def serve(host: str = "127.0.0.1", port: int = 8099, open_browser: bool = True) -> None:
    ensure_dirs()
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}"
    print(f"sigil ui  ->  {url}")
    print("   (local only; this tool searches for people by face)")
    if open_browser:
        import webbrowser

        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
