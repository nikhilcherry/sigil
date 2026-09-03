"""The guards on every fetch.

Candidate URLs come off a third-party feed, so this is the boundary where a
hostile or merely broken response has to be refused rather than trusted.
"""

import pytest
import requests

from sigil.search.http import MAX_IMAGE_BYTES, USER_AGENT, fetch_image, make_session


@pytest.fixture(autouse=True)
def _fake_hosts_resolve(monkeypatch):
    """Let this file's invented hostnames resolve, without faking the guard.

    The download guard refuses a host that does not resolve, which is right in
    a real run and would otherwise make every test here pass for the wrong
    reason - a content-type test returning None because the *name* was refused
    has stopped testing content types.

    An address literal is passed through unchanged, so the tests that check
    loopback and private ranges still exercise the real decision; only invented
    *names* are answered, and with a public address.
    """
    import ipaddress

    import sigil.search.http as h

    def resolve(host, *_a, **_kw):
        bare = str(host).strip("[]")
        try:
            ipaddress.ip_address(bare)
        except ValueError:
            return [(2, 1, 6, "", ("93.184.216.34", 0))]
        return [(2, 1, 6, "", (bare, 0))]

    monkeypatch.setattr(h.socket, "getaddrinfo", resolve)


class FakeResponse:
    # requests.Response always carries the final URL after redirects, and
    # fetch_image checks it - a double without one is not modelling the thing.
    url = "https://x/a.jpg"

    def __init__(self, status=200, ctype="image/jpeg", chunks=(b"data",), length=None):
        self.status_code = status
        self.headers = {"Content-Type": ctype}
        if length is not None:
            self.headers["Content-Length"] = str(length)
        self._chunks = chunks

    def iter_content(self, size):
        yield from self._chunks

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeSession:
    def __init__(self, response):
        self.response = response

    def get(self, url, timeout=None, stream=None):
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_an_image_is_returned_whole():
    session = FakeSession(FakeResponse(chunks=(b"abc", b"def")))
    assert fetch_image(session, "https://x/a.jpg", 5.0) == b"abcdef"


def test_a_non_image_content_type_is_refused():
    """An HTML error page served where an image was expected is the common case."""
    session = FakeSession(FakeResponse(ctype="text/html", chunks=(b"<html>",)))
    assert fetch_image(session, "https://x/a.jpg", 5.0) is None


def test_a_non_200_is_refused():
    session = FakeSession(FakeResponse(status=404))
    assert fetch_image(session, "https://x/a.jpg", 5.0) is None


def test_an_oversized_declared_length_is_refused_before_reading():
    session = FakeSession(FakeResponse(length=MAX_IMAGE_BYTES + 1))
    assert fetch_image(session, "https://x/a.jpg", 5.0) is None


def test_a_body_that_lies_about_its_length_is_cut_off():
    """Content-Length is a claim, not a guarantee - an unbounded read is a
    trivial way to hang a run, so the cap has to hold while streaming too."""
    oversized = (b"x" * (1024 * 1024) for _ in range(20))
    session = FakeSession(FakeResponse(chunks=oversized, length=10))
    assert fetch_image(session, "https://x/a.jpg", 5.0) is None


def test_a_network_error_is_not_fatal():
    """One dead candidate URL must not end the search."""
    session = FakeSession(requests.ConnectionError("no route"))
    assert fetch_image(session, "https://x/a.jpg", 5.0) is None


def test_a_malformed_content_length_drops_the_image_rather_than_raising():
    """The image is lost, but one bad header must not take the run down with it."""
    session = FakeSession(FakeResponse(chunks=(b"ok",)))
    session.response.headers["Content-Length"] = "not-a-number"
    assert fetch_image(session, "https://x/a.jpg", 5.0) is None


def test_the_session_identifies_itself_and_retries():
    """A shared session with a real UA and backoff, not bare requests.get."""
    s = make_session()
    assert s.headers["User-Agent"] == USER_AGENT
    assert "github.com/nikhilcherry/sigil" in USER_AGENT
    retry = s.get_adapter("https://x").max_retries
    assert retry.total == 3
    assert 429 in retry.status_forcelist


# ------------------------------------- where a candidate is allowed to live


def test_a_public_https_url_is_allowed(monkeypatch):
    import sigil.search.http as h

    monkeypatch.setattr(h.socket, "getaddrinfo",
                        lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))])
    assert h.is_public_http_url("https://cdn.example/img.jpg") is True


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:8099/api/evidence",   # this tool's own web UI
    "http://169.254.169.254/latest/meta-data/",  # cloud metadata
    "http://10.0.0.5/x.jpg",
    "http://192.168.1.1/x.jpg",
    "https://[::1]/x.jpg",
])
def test_a_private_or_loopback_address_is_refused(url):
    """Candidate URLs are chosen by third parties and fetched by this process.

    None of these can host a publicly posted image, so a candidate there is
    wrong on the evidence bundle's own terms as well as unsafe to fetch.
    """
    from sigil.search.http import is_public_http_url

    assert is_public_http_url(url) is False


@pytest.mark.parametrize("url", [
    "file:///etc/passwd",
    "ftp://example.com/x.jpg",
    "gopher://example.com/x",
    "",
    "https://",
    "not a url at all",
])
def test_anything_that_is_not_public_http_is_refused(url):
    from sigil.search.http import is_public_http_url

    assert is_public_http_url(url) is False


def test_a_name_that_does_not_resolve_is_refused(monkeypatch):
    import socket as _socket

    import sigil.search.http as h

    def unresolvable(*_a, **_kw):
        raise _socket.gaierror("Name or service not known")

    monkeypatch.setattr(h.socket, "getaddrinfo", unresolvable)
    assert h.is_public_http_url("https://nope.invalid/x.jpg") is False


def test_a_resolver_returning_nothing_is_refused(monkeypatch):
    """`getaddrinfo` answering with an empty list is not an error, and an
    empty loop over addresses would otherwise fall through to the trailing
    `return True` - approving a host precisely because nothing about it could
    be checked. Nothing on the happy path exercises this, so it is asserted
    directly rather than assumed from the shape of the code.
    """
    import sigil.search.http as h

    monkeypatch.setattr(h.socket, "getaddrinfo", lambda *a, **k: [])
    assert h.is_public_http_url("https://empty.example/x.jpg") is False


def test_an_address_the_resolver_gives_that_will_not_parse_is_refused(monkeypatch):
    """The guard classifies whatever `getaddrinfo` hands back. A value that is
    not an address at all cannot be classified, and the only safe reading of an
    unclassifiable address is to refuse it - the alternative is to skip the one
    entry that could not be checked and approve the host on the others.
    """
    import sigil.search.http as h

    monkeypatch.setattr(h.socket, "getaddrinfo", lambda *a, **k: [
        (2, 1, 6, "", ("not-an-ip", 0)),
    ])
    assert h.is_public_http_url("https://garbled.example/x.jpg") is False


def test_a_name_resolving_to_both_public_and_private_is_refused(monkeypatch):
    """A host that answers with a private address as well is not trustworthy."""
    import sigil.search.http as h

    monkeypatch.setattr(h.socket, "getaddrinfo", lambda *a, **k: [
        (2, 1, 6, "", ("93.184.216.34", 0)),
        (2, 1, 6, "", ("127.0.0.1", 0)),
    ])
    assert h.is_public_http_url("https://sneaky.example/x.jpg") is False


def test_fetch_image_refuses_a_private_url_without_making_a_request(monkeypatch):
    """The guard has to run before the socket, or it guards nothing."""
    import sigil.search.http as h

    class Loud:
        def get(self, *a, **kw):
            raise AssertionError("a request was made to a refused URL")

    assert h.fetch_image(Loud(), "http://127.0.0.1:9/x.jpg", 5.0) is None


def test_a_redirect_into_a_private_address_is_refused(monkeypatch):
    """An https CDN URL that 302s to loopback would walk past the first check."""
    import sigil.search.http as h

    monkeypatch.setattr(h, "is_public_http_url",
                        lambda u: not u.startswith("http://127."))

    class Redirected:
        status_code = 200
        url = "http://127.0.0.1:8099/api/evidence"
        headers = {"Content-Type": "image/jpeg"}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def iter_content(self, _n):
            raise AssertionError("the body was read from a private address")

    class Session:
        def get(self, *a, **kw):
            return Redirected()

    assert h.fetch_image(Session(), "https://cdn.example/img.jpg", 5.0) is None
