"""Shared types for search providers."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class Candidate:
    """One publicly-posted image that the search turned up, before any matching."""

    platform: str
    image_url: str
    post_url: str
    post_uri: str
    author_handle: str
    author_did: str
    author_display_name: str
    text: str
    created_at: str
    discovered_via: str
    # "social" for a post or profile on a named platform, "web" for a page the
    # open-web arms found. The task asks for a social media post; an open-web
    # page is corroboration, so the two are not interchangeable outputs.
    source_kind: str = "web"


@dataclass
class ProviderTrace:
    """A record of what was actually asked of the network.

    This exists so a run can be audited after the fact: it is the difference
    between "the pipeline searched" and "the pipeline claims it searched".
    """

    provider: str
    calls: list[dict[str, Any]] = field(default_factory=list)

    def record(self, endpoint: str, params: dict[str, Any], result_count: int) -> None:
        self.calls.append(
            {"endpoint": endpoint, "params": params, "results": result_count}
        )


@runtime_checkable
class SearchProvider(Protocol):
    """What the matcher needs from an arm, and what a new arm must supply.

    Runtime-checkable and actually used as the annotation below, so it is a
    contract rather than a comment: `tests/test_search.py` asserts every
    shipped provider satisfies it, which is what catches an arm that forgets
    `kind` or `trace`.
    """

    name: str
    # "social" or "web". Each provider stamps this onto every Candidate it
    # yields, so it is declared once per arm rather than repeated at each
    # construction site - Bluesky builds candidates in two places and would
    # otherwise be one forgotten literal away from silently losing the
    # social preference in `pick_best`.
    kind: str
    trace: ProviderTrace

    def candidates(self, query: str) -> Iterator[Candidate]: ...
