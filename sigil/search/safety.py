"""Which candidates this tool refuses to look at, and why it refuses them.

Every candidate URL here was chosen by a third party - an AppView response,
whatever a reverse-image index returned for a face - and a face query in
particular pulls a predictable class of result: adult tube sites, "leaked
nudes" aggregators, and the scraper networks that republish both. Two separate
things go wrong if those are simply scored like anything else.

The first is what ends up on screen. A run prints candidate handles and page
hosts to a terminal, and the web UI renders thumbnails of what it downloaded.
This tool is meant to be demonstrated in front of people.

The second is the one that actually matters, and it is not about squeamishness.
A cleared candidate becomes a *citation in an evidence bundle*, and that bundle
is hashed onto an append-only ledger. Anchoring a record whose `post_url` points
at a nonconsensual-imagery site publishes a permanent, unrevocable association
between a person's face and that site - which is a considerably worse outcome
than the run finding nothing at all. The registry cannot be edited afterwards,
so the refusal has to happen before the anchor, not after somebody notices.

**What is matched, and what deliberately is not.** Hosts are matched against a
list, suffix-wise so that a subdomain counts. Keywords are matched against the
URL path, the account handle and the display name - the parts a *source* is
made of - and are tokenised first, so `sex` does not fire on `sussex`, `ass`
does not fire on `passport` or `Cassidy`, and `leaks` does not fire on
`leaksville`. A substring scan is the obvious implementation and it is wrong in
a way that is invisible in testing, because the strings it wrongly blocks are
ordinary place names and surnames that no adult-content test case would think
to contain.

Post *text* is not scanned at all. A word in a post is not what makes a source
unsafe, and scanning it means a true match is silently discarded because the
person wrote "sexy" - a false negative that looks exactly like "not found".
The source is judged by where it is, not by what it says.

None of this is a claim to have solved the problem. A blocklist is a list, and
the sites on it rename themselves; this catches the bulk and the recognisable
shapes, and `--allow-unsafe` exists for an operator who has a reason to look
anyway and is accepting the consequence knowingly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

# Hosts refused outright, matched on the domain suffix so that
# `cdn.example.com` goes with `example.com`. Adult tube sites and the
# "leak"/scraper aggregators that a face query surfaces most often.
BLOCKED_HOSTS: frozenset[str] = frozenset({
    "pornhub.com", "xvideos.com", "xhamster.com", "xnxx.com", "redtube.com",
    "youporn.com", "spankbang.com", "eporner.com", "tnaflix.com", "txxx.com",
    "beeg.com", "motherless.com", "nudevista.com", "sex.com", "porntrex.com",
    "onlyfans.com", "fansly.com", "coomer.su", "coomer.party", "kemono.su",
    "kemono.party", "thothub.tv", "thothub.lol", "simpcity.su", "bunkr.si",
    "bunkr.la", "bunkrr.su", "cyberdrop.me", "erome.com", "fapello.com",
    "influencersgonewild.com", "leakedzone.com", "nudostar.com",
    "thefappeningblog.com", "celebjihad.com", "sexcelebrity.net",
    "rule34.xxx", "e621.net", "gelbooru.com",
})

# Tokens that mark a URL path, an account handle or a display name as adult or
# leak-site material. Matched whole, never as substrings - see the module
# docstring for why that distinction is the whole design.
BLOCKED_TOKENS: frozenset[str] = frozenset({
    "porn", "porno", "pornstar", "xxx", "nsfw", "nude", "nudes", "nudity",
    "naked", "erotic", "erotica", "hentai", "camgirl", "camgirls", "escort",
    "escorts", "onlyfans", "fansly", "gonewild", "milf", "bdsm", "fetish",
    "stripper", "striptease", "creampie", "blowjob", "handjob", "cumshot",
    "deepfake", "deepfakes", "fakenudes", "nudify", "undress", "thothub",
    "fappening", "celebnudes", "leakedmodels", "sextape", "sexcam",
})

# Subreddits are their own namespace: `reddit.com` cannot be blocklisted (it is
# an ordinary source) but `reddit.com/r/<sub>` can be, and the subreddit name is
# the only part of that URL carrying the signal.
BLOCKED_SUBREDDITS: frozenset[str] = frozenset({
    "gonewild", "nsfw", "realgirls", "nsfw_gif", "nsfwoutfits", "onlyfans",
    "celebnsfw", "watchitfortheplot", "fakeapp", "deepfakes",
})

_TOKEN = re.compile(r"[a-z0-9]+")
_SUBREDDIT = re.compile(r"/r/([A-Za-z0-9_]+)")


@dataclass(frozen=True)
class Verdict:
    """Whether a candidate may be looked at, and the category if not.

    The category is a fixed word from this module rather than the offending
    text. Blocked sources are counted and classified, never echoed: the counts
    travel in the search trace, and the search trace is part of the bundle that
    gets hashed onto a public chain, so the name of the site it saw is exactly
    the string that must not be written there.
    """

    allowed: bool
    category: str = ""

    def __bool__(self) -> bool:
        return self.allowed


ALLOWED = Verdict(True)


def registrable_host(url: str) -> str:
    """The host of a URL, lowercased and stripped of a leading `www.`."""
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def _host_blocked(host: str) -> bool:
    if not host:
        return False
    return any(host == b or host.endswith("." + b) for b in BLOCKED_HOSTS)


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def _subreddit_blocked(url: str) -> bool:
    host = registrable_host(url)
    if host != "reddit.com" and not host.endswith(".reddit.com"):
        return False
    try:
        path = urlparse(url).path
    except ValueError:
        return False
    return any(m.group(1).lower() in BLOCKED_SUBREDDITS
               for m in _SUBREDDIT.finditer(path))


def classify(
    post_url: str = "",
    image_url: str = "",
    handle: str = "",
    display_name: str = "",
) -> Verdict:
    """Decide whether one candidate source may be downloaded and cited."""
    for url in (post_url, image_url):
        if _host_blocked(registrable_host(url)):
            return Verdict(False, "host")
    for url in (post_url, image_url):
        if _subreddit_blocked(url):
            return Verdict(False, "subreddit")
    # The host is deliberately excluded from token matching: a blocked host is
    # already handled above, and tokenising the domain of an unlisted site
    # turns `sextonpublishing.com` into a block on the word `sexton`. Only the
    # path and query carry the per-page signal.
    for url in (post_url, image_url):
        try:
            parts = urlparse(url)
        except ValueError:
            continue
        if _tokens(parts.path + " " + (parts.query or "")) & BLOCKED_TOKENS:
            return Verdict(False, "url")
    if _tokens(handle) & BLOCKED_TOKENS:
        return Verdict(False, "handle")
    if _tokens(display_name) & BLOCKED_TOKENS:
        return Verdict(False, "display_name")
    return ALLOWED


def is_safe(candidate) -> Verdict:
    """Classify a `search.base.Candidate` without importing it (avoids a cycle)."""
    return classify(
        post_url=getattr(candidate, "post_url", "") or "",
        image_url=getattr(candidate, "image_url", "") or "",
        handle=getattr(candidate, "author_handle", "") or "",
        display_name=getattr(candidate, "author_display_name", "") or "",
    )
