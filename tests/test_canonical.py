"""One spelling per source: what the canonicaliser changes, and what it must not."""

import pytest

from sigil.canonical import normalize_text, normalize_url
from sigil.evidence import Evidence, MatchRef, ProbeRef

# ------------------------------------------------------------- what it changes


@pytest.mark.parametrize("raw,expected", [
    ("HTTPS://Bsky.APP/profile/x", "https://bsky.app/profile/x"),
    ("https://bsky.app:443/profile/x", "https://bsky.app/profile/x"),
    ("http://example.com:80/a", "http://example.com/a"),
    ("https://exämple.de/p", "https://xn--exmple-cua.de/p"),
    ("  https://example.com/a  ", "https://example.com/a"),
])
def test_the_same_resource_gets_the_same_spelling(raw, expected):
    assert normalize_url(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("https://ex.com/p?utm_source=twitter", "https://ex.com/p"),
    ("https://ex.com/p?utm_source=t&id=5", "https://ex.com/p?id=5"),
    ("https://ex.com/p?fbclid=abc&gclid=d", "https://ex.com/p"),
    ("https://ex.com/p?igshid=z&q=1", "https://ex.com/p?q=1"),
])
def test_analytics_parameters_are_dropped(raw, expected):
    """Two runs that found one post through two share links are one finding."""
    assert normalize_url(raw) == expected


def test_nfc_and_nfd_spellings_of_a_name_hash_the_same():
    composed = "Beyoncé"
    decomposed = "Beyoncé"
    assert composed != decomposed
    assert normalize_text(composed) == normalize_text(decomposed)


# --------------------------------------------------------- what it leaves alone


def test_a_signed_cdn_url_is_returned_untouched():
    """Reordering or re-encoding one is how a working image URL becomes a 403.

    `verify --recheck-source` refetches this URL, so a canonicaliser that
    breaks it produces a bundle that fails to verify because of its own
    normalisation - the worst outcome available here.
    """
    signed = ("https://cdn.example.com/i.jpg"
              "?X-Amz-Date=20260101T000000Z&X-Amz-Signature=deadbeef&X-Amz-Expires=900")
    assert normalize_url(signed) == signed


def test_query_order_is_preserved_when_nothing_is_dropped():
    url = "https://ex.com/p?z=1&a=2&m=3"
    assert normalize_url(url) == url


def test_at_protocol_uris_are_not_treated_as_urls():
    uri = "at://did:plc:Example/app.bsky.feed.post/3abc"
    assert normalize_url(uri) == uri


@pytest.mark.parametrize("url", [
    "https://ex.com/a/",          # a trailing slash can be a different resource
    "https://ex.com/A/B",         # paths are case-sensitive
    "https://ex.com/p?ref=nav",   # plenty of sites route on ref
    "https://ex.com/p?source=x",  # and on source
    "https://ex.com/p#section",
])
def test_ambiguous_parts_are_left_exactly_as_found(url):
    assert normalize_url(url) == url


def test_credentials_and_non_default_ports_survive():
    assert normalize_url("https://u:p@ex.com:8443/a") == "https://u:p@ex.com:8443/a"


@pytest.mark.parametrize("value", ["", "not a url", "http://", "https://["])
def test_malformed_input_is_returned_rather_than_raising(value):
    assert isinstance(normalize_url(value), str)


# ------------------------------------------------------- inside the bundle


def _bundle(post_url, display_name="Someone"):
    return Evidence(
        probe=ProbeRef(image_sha256="ab" * 32, embedding_sha256="cd" * 32,
                       backend="opencv", model="yunet+sface", bbox=[0, 0, 10, 10],
                       det_score=0.9),
        match=MatchRef(
            platform="bluesky", post_url=post_url,
            post_uri="at://did:plc:x/app.bsky.feed.post/1",
            author_handle="x.bsky.social", author_did="did:plc:x",
            author_display_name=display_name, text="",
            image_url="https://cdn/a.jpg", image_sha256="ef" * 32,
            created_at="2026-08-01T00:00:00Z", discovered_via="test"),
        similarity=0.7, threshold=0.38, searched_at="2026-09-07T00:00:00Z",
    )


def test_one_finding_reached_two_ways_produces_one_evidence_hash():
    """The reason this module exists.

    The registry refuses duplicates so that the first record of a finding
    stands. Two spellings of one post used to produce two unrelated hashes, so
    the same finding could be anchored twice and nothing on chain could tell.
    """
    plain = _bundle("https://bsky.app/profile/who/post/3abc")
    shared = _bundle("https://bsky.app/profile/who/post/3abc?utm_source=twitter")
    assert plain.evidence_hash_hex() == shared.evidence_hash_hex()


def test_the_stored_url_is_the_one_that_was_hashed():
    """`verify --recheck-source` reads the field, so the field has to be canonical."""
    ev = _bundle("HTTPS://BSKY.app/profile/who/post/3abc?utm_source=x")
    assert ev.match.post_url == "https://bsky.app/profile/who/post/3abc"


def test_a_display_name_in_either_unicode_form_gives_one_hash():
    assert (_bundle("https://ex.com/p", "Beyoncé").evidence_hash_hex()
            == _bundle("https://ex.com/p", "Beyoncé").evidence_hash_hex())


def test_normalisation_survives_a_round_trip_through_json():
    import json

    ev = _bundle("https://bsky.app/profile/who/post/3abc?utm_source=x")
    back = Evidence.from_dict(json.loads(ev.canonical_json()))
    assert back.canonical_json() == ev.canonical_json()
