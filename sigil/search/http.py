"""One shared HTTP session with the guards every fetch in this project needs."""

from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .. import __version__

USER_AGENT = f"sigil/{__version__} (+https://github.com/nikhilcherry/sigil)"
MAX_IMAGE_BYTES = 12 * 1024 * 1024


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"),
        respect_retry_after_header=True,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=16))
    s.mount("http://", HTTPAdapter(max_retries=retry, pool_maxsize=16))
    return s


def fetch_image(session: requests.Session, url: str, timeout: float) -> bytes | None:
    """Download an image, refusing anything that is not one or is absurdly large.

    Streaming with a hard byte cap matters here: candidate URLs come from a
    third-party feed, so an unbounded read is a trivial way to hang the run.
    """
    try:
        with session.get(url, timeout=timeout, stream=True) as r:
            if r.status_code != 200:
                return None
            ctype = r.headers.get("Content-Type", "")
            if not ctype.startswith("image/"):
                return None
            declared = r.headers.get("Content-Length")
            if declared and int(declared) > MAX_IMAGE_BYTES:
                return None
            buf = bytearray()
            for chunk in r.iter_content(64 * 1024):
                buf.extend(chunk)
                if len(buf) > MAX_IMAGE_BYTES:
                    return None
            return bytes(buf)
    except (requests.RequestException, ValueError):
        return None
