"""Google Cloud Vision web detection - the open-web arm that needs no hosting.

Bluesky is the zero-credential provider, but its coverage is thin: a probe of
someone who is not on the platform finds nothing, which is honest and also a
poor demonstration. This widens the net to the whole indexed web.

It is preferred over the SerpAPI Lens arm for one practical reason: Lens
matches on a *URL*, so the probe has to be publicly reachable before it can be
searched at all. Vision accepts the image bytes directly, so a local file
works, which removes the hosting step entirely.

What comes back is still only a retrieval heuristic. Google says "this image
appears on these pages"; whether the face in it is the probe's face is decided
here, by the same encoder that read the probe.
"""

from __future__ import annotations

import base64
from collections.abc import Iterator
from typing import Any
from urllib.parse import urlparse

from ..config import Config
from .base import Candidate, ProviderTrace
from .http import make_session

ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"
MAX_RESULTS = 50


class GoogleVisionProvider:
    name = "google-vision-web"

    def __init__(self, cfg: Config, probe_bytes: bytes) -> None:
        self.cfg = cfg
        self.probe_bytes = probe_bytes
        self.session = make_session()
        self.trace = ProviderTrace(provider=self.name)

    @classmethod
    def available_for(cls, cfg: Config, probe_bytes: bytes | None) -> bool:
        return bool(cfg.google_vision_key and probe_bytes)

    def _detect(self) -> dict[str, Any]:
        body = {
            "requests": [{
                "image": {"content": base64.b64encode(self.probe_bytes).decode()},
                "features": [{"type": "WEB_DETECTION", "maxResults": MAX_RESULTS}],
            }]
        }
        try:
            r = self.session.post(
                ENDPOINT,
                params={"key": self.cfg.google_vision_key},
                json=body,
                timeout=self.cfg.http_timeout,
            )
            if r.status_code != 200:
                return {}
            payload = r.json()
        except Exception:  # noqa: BLE001 - an optional arm must not end a run
            return {}
        responses = payload.get("responses") or [{}]
        return (responses[0] or {}).get("webDetection", {}) or {}

    def candidates(self, query: str) -> Iterator[Candidate]:
        """Yield every image Google says matches, page-anchored ones first.

        ``query`` is ignored: this is a reverse-image search, so the probe is
        the query. It stays in the signature because that is the provider
        protocol every arm shares.
        """
        web = self._detect()

        pages = web.get("pagesWithMatchingImages") or []
        loose = [
            (kind, web.get(kind) or [])
            # Ordered by how strong a claim each one makes. Visually similar is
            # the weakest - Google means "this picture looks like that picture",
            # not "this is the same image" - but for a face search it is also
            # where a *different* photo of the same person turns up, and the
            # encoder is the one deciding, not this list.
            for kind in ("fullMatchingImages", "partialMatchingImages",
                         "visuallySimilarImages")
        ]

        counted = len(pages) + sum(len(v) for _, v in loose)
        # The key is never recorded: the trace is written into the evidence
        # bundle, which gets published and hashed on chain.
        self.trace.record("vision.webDetection",
                          {"features": ["WEB_DETECTION"], "maxResults": MAX_RESULTS},
                          counted)

        for page in pages:
            page_url = page.get("url") or ""
            title = page.get("pageTitle") or ""
            images = (page.get("fullMatchingImages") or []) + \
                     (page.get("partialMatchingImages") or [])
            for img in images:
                if img.get("url"):
                    yield self._candidate(img["url"], page_url, title,
                                          "vision:pagesWithMatchingImages")

        for kind, images in loose:
            for img in images:
                url = img.get("url")
                if url:
                    # No page context for these, so the image stands as its own
                    # citation rather than pointing at a page it may not be on.
                    yield self._candidate(url, url, "", f"vision:{kind}")

    def _candidate(self, image_url: str, page_url: str, title: str,
                   via: str) -> Candidate:
        host = urlparse(page_url or image_url).netloc
        return Candidate(
            platform=(host or "web").lower(),
            image_url=image_url,
            post_url=page_url or image_url,
            post_uri=page_url or image_url,
            author_handle=host,
            author_did="",
            author_display_name=host,
            text=title[:500],
            created_at="",
            discovered_via=via,
        )
