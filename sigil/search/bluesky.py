"""Bluesky / AT Protocol search provider.

Bluesky is the primary provider because its AppView serves ``searchActors``,
``getProfile`` and ``getAuthorFeed`` to anonymous callers, so a clone of this
repo performs a real, live search against real accounts with no credentials at
all. ``searchPosts`` is the one endpoint that requires a session, so it is
enabled only when an app password is supplied and is otherwise skipped rather
than faked.
"""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..config import Config
from .base import Candidate, ProviderTrace
from .http import make_session

PUBLIC_API = "https://public.api.bsky.app/xrpc"
AUTH_API = "https://bsky.social/xrpc"

# One getAuthorFeed round trip per matching account, and they do not depend
# on each other. Kept modest deliberately: this is an unauthenticated public
# AppView, and the retry adapter's backoff is the fallback, not the plan.
FEED_WORKERS = 6


def at_uri_to_web_url(uri: str, handle: str) -> str:
    """at://did:plc:xyz/app.bsky.feed.post/3abc -> https://bsky.app/profile/<handle>/post/3abc"""
    rkey = uri.rsplit("/", 1)[-1] if "/" in uri else uri
    return f"https://bsky.app/profile/{handle}/post/{rkey}"


class BlueskyProvider:
    name = "bluesky"
    kind = "social"

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.session = make_session()
        self.trace = ProviderTrace(provider=self.name)
        self._token: str | None = None
        self._authenticate()

    # -- transport -------------------------------------------------------

    def _authenticate(self) -> None:
        if not (self.cfg.bluesky_handle and self.cfg.bluesky_app_password):
            return
        try:
            r = self.session.post(
                f"{AUTH_API}/com.atproto.server.createSession",
                json={
                    "identifier": self.cfg.bluesky_handle,
                    "password": self.cfg.bluesky_app_password,
                },
                timeout=self.cfg.http_timeout,
            )
            if r.status_code == 200:
                self._token = r.json().get("accessJwt")
        except Exception:  # noqa: BLE001 - auth is strictly an upgrade, never required
            self._token = None

    @property
    def authenticated(self) -> bool:
        return self._token is not None

    def _get(self, endpoint: str, params: dict[str, Any], authed: bool = False) -> dict | None:
        base = AUTH_API if (authed and self._token) else PUBLIC_API
        headers = {"Authorization": f"Bearer {self._token}"} if (authed and self._token) else {}
        try:
            r = self.session.get(
                f"{base}/{endpoint}",
                params=params,
                headers=headers,
                timeout=self.cfg.http_timeout,
            )
            if r.status_code != 200:
                return None
            return r.json()
        except Exception:  # noqa: BLE001 - every caller records the zero result
            return None

    # -- extraction ------------------------------------------------------

    @staticmethod
    def _images_from_post(post: dict) -> list[str]:
        embed = post.get("embed") or {}
        images = embed.get("images") or []
        media = embed.get("media") or {}
        images = images or media.get("images") or []
        urls = [i.get("fullsize") or i.get("thumb") for i in images]
        if not urls and embed.get("thumbnail"):
            urls = [embed["thumbnail"]]
        return [u for u in urls if u]

    def _candidates_from_post(self, post: dict) -> Iterator[Candidate]:
        author = post.get("author") or {}
        handle = author.get("handle", "")
        record = post.get("record") or {}
        for url in self._images_from_post(post):
            yield Candidate(
                platform="bluesky",
                source_kind=self.kind,
                image_url=url,
                post_url=at_uri_to_web_url(post.get("uri", ""), handle),
                post_uri=post.get("uri", ""),
                author_handle=handle,
                author_did=author.get("did", ""),
                author_display_name=author.get("displayName", "") or "",
                text=(record.get("text") or "")[:500],
                created_at=record.get("createdAt", "") or post.get("indexedAt", "") or "",
                discovered_via="app.bsky.feed.getAuthorFeed",
            )

    # -- provider surface ------------------------------------------------

    def candidates(self, query: str) -> Iterator[Candidate]:
        yield from self._from_actor_search(query)
        if self.authenticated:
            yield from self._from_post_search(query)

    def _from_actor_search(self, query: str) -> Iterator[Candidate]:
        params = {"q": query, "limit": min(self.cfg.max_actors, 100)}
        data = self._get("app.bsky.actor.searchActors", params)
        actors = (data or {}).get("actors", [])
        self.trace.record("app.bsky.actor.searchActors", params, len(actors))

        def feed_params(actor: dict) -> dict[str, Any]:
            return {
                "actor": actor.get("handle", "") or actor.get("did", ""),
                "limit": min(self.cfg.posts_per_actor, 100),
                "filter": "posts_with_media",
            }

        def fetch_feed(actor: dict) -> dict | None:
            return self._get("app.bsky.feed.getAuthorFeed", feed_params(actor))

        # Every avatar first, across all actors, before any feed image.
        # Measured over two labelled corpora, 538 real Bluesky candidates: 62%
        # of avatars contain a detectable face (24/39) against 33% of feed
        # images (167/499). The margin varies by query - 53% vs 20% on one
        # corpus, 68% vs 49% on the other - but avatars won both, which is what
        # the ordering rests on. The strongest live match in every run so far
        # has also been an avatar; it is the one picture an account chooses to
        # represent a person by.
        #
        # They cost nothing extra, since searchActors already returned them, so
        # ordering them first is free recall for any run whose budget is
        # smaller than what Bluesky offers. Which is every run: `max_images`
        # defaults to 200 and 25 actors at 20 posts each can propose ten times
        # that.
        # The feeds are fetched concurrently but consumed in actor order, and
        # the trace is written here rather than in the workers, so both the
        # candidate stream and the audit record stay identical to a serial run.
        with ThreadPoolExecutor(max_workers=FEED_WORKERS) as pool:
            # Submitted before the first avatar is yielded, so the feeds are
            # already in flight while the encoder works through them. Ordering
            # avatars first would otherwise idle the network for as long as it
            # takes to encode twenty-odd of them.
            #
            # Every one at once rather than through `prefetch`'s bounded
            # window, which exists to stop thousands of *image* downloads
            # piling up in memory ahead of a slow consumer. These are at most
            # a hundred small JSON responses - the actor search is capped
            # there - so the window would only delay them.
            pending = [(actor, pool.submit(fetch_feed, actor)) for actor in actors]

            for actor in actors:
                if actor.get("avatar"):
                    yield Candidate(
                        platform="bluesky",
                        source_kind=self.kind,
                        image_url=actor["avatar"],
                        post_url=f"https://bsky.app/profile/{actor.get('handle', '')}",
                        post_uri=f"at://{actor.get('did', '')}/app.bsky.actor.profile/self",
                        author_handle=actor.get("handle", ""),
                        author_did=actor.get("did", ""),
                        author_display_name=actor.get("displayName", "") or "",
                        text=(actor.get("description") or "")[:500],
                        created_at=actor.get("createdAt", "") or "",
                        discovered_via="app.bsky.actor.searchActors:avatar",
                    )

            for actor, future in pending:
                items = (future.result() or {}).get("feed", [])
                self.trace.record("app.bsky.feed.getAuthorFeed", feed_params(actor), len(items))
                for item in items:
                    post = item.get("post") or {}
                    yield from self._candidates_from_post(post)

    def _from_post_search(self, query: str) -> Iterator[Candidate]:
        params = {"q": query, "limit": 50}
        data = self._get("app.bsky.feed.searchPosts", params, authed=True)
        posts = (data or {}).get("posts", [])
        self.trace.record("app.bsky.feed.searchPosts", params, len(posts))
        for post in posts:
            for cand in self._candidates_from_post(post):
                cand.discovered_via = "app.bsky.feed.searchPosts"
                yield cand
