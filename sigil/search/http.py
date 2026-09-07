"""One shared HTTP session with the guards every fetch in this project needs."""

from __future__ import annotations

import ipaddress
import os
import socket
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .. import __version__

DEFAULT_USER_AGENT = f"sigil/{__version__} (+https://github.com/nikhilcherry/sigil)"

# Overridable, and deliberately not overridden by default. Some CDNs serve
# images only to something that looks like a browser, and the fix people reach
# for is to send Chrome's UA string - which is a decision about whether to lie
# to a host that has said what it wants, not a technical detail, so it is the
# operator's to make rather than this file's. What ships identifies itself, and
# SIGIL_USER_AGENT is there for someone who has decided otherwise.
USER_AGENT = os.getenv("SIGIL_USER_AGENT", "").strip() or DEFAULT_USER_AGENT

# Sent on every request. Not cosmetic: a client offering no Accept at all is
# unusual enough that some image CDNs answer it with a 406 or an HTML error
# page, and an HTML error page fails the Content-Type check in fetch_image and
# is then indistinguishable from "that candidate had no image". These headers
# say truthfully what this client wants and will read.
DEFAULT_HEADERS = {
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

MAX_IMAGE_BYTES = 12 * 1024 * 1024


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    s.headers.update(DEFAULT_HEADERS)
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


def is_public_http_url(url: str) -> bool:
    """Is this an address a publicly posted image could actually live at?

    Candidate URLs are chosen by third parties - an AppView response, whatever
    Google's web index returned - and then fetched by this process. Without a
    check, "download every candidate" means "issue a GET wherever a search
    result points", which includes `http://127.0.0.1:8099/api/evidence` (this
    tool's own web UI), a cloud metadata endpoint, or anything else on the
    machine's private networks.

    The reason to refuse is not only defensive. The evidence bundle asserts
    that the match is a publicly posted image; an address on the loopback or a
    private range cannot be one, so a candidate there is wrong on the bundle's
    own terms rather than merely risky.

    Known limit, stated rather than papered over: the name is resolved here and
    again by the request, so a host that answers differently between the two
    lookups defeats this. Closing that needs the connection pinned to the
    address that was checked, which is a larger change than this file.
    """
    parts = urlparse(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return False
    try:
        infos = socket.getaddrinfo(parts.hostname, None)
    except (socket.gaierror, UnicodeError, ValueError):
        return False
    if not infos:
        return False
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            return False
        if not ip.is_global or ip.is_multicast:
            return False
    return True


def fetch_image(session: requests.Session, url: str, timeout: float) -> bytes | None:
    """Download an image, refusing anything that is not one or is absurdly large.

    Streaming with a hard byte cap matters here: candidate URLs come from a
    third-party feed, so an unbounded read is a trivial way to hang the run.
    """
    if not is_public_http_url(url):
        return None
    try:
        with session.get(url, timeout=timeout, stream=True) as r:
            if r.status_code != 200:
                return None
            # Redirects are followed, so where the response actually came from
            # has to be checked too - an https CDN URL that 302s to loopback
            # would otherwise walk straight past the check above.
            if r.url != url and not is_public_http_url(r.url):
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
