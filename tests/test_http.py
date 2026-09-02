"""The guards on every fetch.

Candidate URLs come off a third-party feed, so this is the boundary where a
hostile or merely broken response has to be refused rather than trusted.
"""

import requests

from sigil.search.http import MAX_IMAGE_BYTES, USER_AGENT, fetch_image, make_session


class FakeResponse:
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
