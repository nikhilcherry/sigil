"""Google Lens reverse-image search via SerpAPI - the open-web arm of the search.

This provider is optional and has a hard constraint worth being honest about:
Google Lens matches on a *URL*, so it can only run when the probe image is
already reachable on the public internet. With a local file it is skipped
rather than silently pretending to have searched.
"""

from __future__ import annotations

from collections.abc import Iterator
from urllib.parse import urlparse

from ..config import Config
from .base import Candidate, ProviderTrace
from .http import make_session

ENDPOINT = "https://serpapi.com/search.json"


class SerpApiLensProvider:
    name = "serpapi-google-lens"

    def __init__(self, cfg: Config, probe_url: str) -> None:
        self.cfg = cfg
        self.probe_url = probe_url
        self.session = make_session()
        self.trace = ProviderTrace(provider=self.name)

    @classmethod
    def available_for(cls, cfg: Config, probe_url: str | None) -> bool:
        if not (cfg.serpapi_key and probe_url):
            return False
        return urlparse(probe_url).scheme in ("http", "https")

    def candidates(self, query: str) -> Iterator[Candidate]:
        params = {
            "engine": "google_lens",
            "url": self.probe_url,
            "api_key": self.cfg.serpapi_key,
        }
        try:
            r = self.session.get(ENDPOINT, params=params, timeout=self.cfg.http_timeout)
            data = r.json() if r.status_code == 200 else {}
        except Exception:  # noqa: BLE001
            data = {}

        matches = data.get("visual_matches", []) or []
        # Never log the API key into the evidence trace.
        self.trace.record("google_lens", {"url": self.probe_url}, len(matches))

        for m in matches:
            image_url = m.get("image") or m.get("thumbnail")
            link = m.get("link") or ""
            if not image_url or not link:
                continue
            yield Candidate(
                platform=(m.get("source") or urlparse(link).netloc or "web").lower(),
                image_url=image_url,
                post_url=link,
                post_uri=link,
                author_handle=m.get("source") or urlparse(link).netloc,
                author_did="",
                author_display_name=m.get("source") or "",
                text=(m.get("title") or "")[:500],
                created_at="",
                discovered_via="serpapi:google_lens:visual_matches",
            )
