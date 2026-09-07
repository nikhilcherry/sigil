"""One spelling per source, so that one finding is one hash.

The bundle's hash is taken over its bytes, which means every difference in how
a field is *written* is a difference in what gets anchored. Two runs that found
the same Bluesky post, one of them through a link carrying `?utm_source=`,
produce two unrelated evidence hashes for one finding - and the registry, which
refuses duplicates precisely so that the first record of a thing stands, cannot
see that they are the same thing. Same for a display name that arrives as
composed `é` in one response and decomposed `e` + U+0301 in another: identical
on screen, identical to a reader, different bytes, different hash.

So the strings a third party supplies are put into one form before they are
hashed. What follows is the argument for the *narrowness* of that form, because
the temptation is to canonicalise much harder than is safe.

**Only http and https URLs are touched at all.** `post_uri` is an `at://`
AT-Protocol URI, and a URL normaliser applied to it produces something that no
longer resolves. Anything that is not http(s) gets Unicode normalisation and
nothing else.

**Query parameter order is preserved.** Sorting is the standard move and it is
unsafe here: candidate image URLs are routinely CDN links whose signature
covers the query string, and reordering one is how a working image URL becomes
a 403 at `sigil verify --recheck-source` time - a bundle that fails to verify
because of its own canonicaliser is the worst outcome available.

**Only unambiguous analytics parameters are dropped.** `utm_*`, `fbclid`,
`gclid`, `igshid`, `mc_eid` are tracking and carry no meaning to the server.
`ref` and `source` are *not* dropped, tempting as they look: plenty of sites
route on them, and this cannot tell which.

**Paths are left exactly as they are** - no trailing-slash trimming, no
re-encoding. `/a` and `/a/` are different resources on servers that say they
are, and re-encoding a path is a way to produce a URL that no longer fetches.

The net effect is small on purpose: case and default ports and a handful of
tracking parameters. That is the part that is provably the same resource.
"""

from __future__ import annotations

import unicodedata
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# Dropped from any http(s) URL before hashing. Every one of these is
# unambiguously client-side analytics; see the module docstring for the ones
# that look like they belong here and do not.
TRACKING_PARAMS = frozenset({
    "fbclid", "gclid", "dclid", "msclkid", "igshid", "mc_eid", "mc_cid",
    "_ga", "_gl", "yclid", "twclid", "ttclid", "vero_id", "oly_enc_id",
})
TRACKING_PREFIXES = ("utm_",)

DEFAULT_PORTS = {"http": 80, "https": 443}


def normalize_text(value: str) -> str:
    """Unicode NFC, so that one piece of text is one sequence of bytes.

    NFC rather than NFKC: NFKC also folds compatibility characters, which
    rewrites the text - it would turn `ﬁ` into `fi` and full-width characters
    into ASCII. This has to preserve what the person actually wrote, and only
    settle which of two identical-by-definition encodings of it is used.
    """
    if not value:
        return value
    return unicodedata.normalize("NFC", value)


def _is_tracking(key: str) -> bool:
    lower = key.lower()
    return lower in TRACKING_PARAMS or lower.startswith(TRACKING_PREFIXES)


def normalize_url(value: str) -> str:
    """The same URL, spelled one way. Non-http(s) input gets NFC and nothing else."""
    if not value:
        return value
    text = normalize_text(value).strip()
    try:
        parts = urlsplit(text)
    except ValueError:
        return text
    if parts.scheme.lower() not in DEFAULT_PORTS:
        return text

    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    if not host:
        return text
    try:
        # Punycode, so that an internationalised domain hashes the same however
        # the arm that found it happened to spell the host.
        host = host.encode("idna").decode("ascii")
    except (UnicodeError, UnicodeDecodeError):
        pass
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"  # IPv6 literal

    netloc = host
    try:
        port = parts.port
    except ValueError:
        port = None
    if port is not None and port != DEFAULT_PORTS[scheme]:
        netloc = f"{host}:{port}"
    # Credentials in a URL are preserved rather than normalised away: dropping
    # them would change which resource the bundle cites.
    if parts.username:
        auth = parts.username
        if parts.password:
            auth += f":{parts.password}"
        netloc = f"{auth}@{netloc}"

    query = parts.query
    if query:
        kept = [(k, v) for k, v in parse_qsl(query, keep_blank_values=True)
                if not _is_tracking(k)]
        # Re-encoded from the parse only when something was actually removed.
        # Round-tripping an untouched query string through urlencode would
        # rewrite its escaping, which is exactly the kind of gratuitous
        # rewriting this module exists to avoid.
        if len(kept) != len(parse_qsl(query, keep_blank_values=True)):
            query = urlencode(kept)

    return urlunsplit((scheme, netloc, parts.path, query, parts.fragment))
