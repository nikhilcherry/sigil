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
    name: str
    trace: ProviderTrace

    def candidates(self, query: str) -> Iterator[Candidate]: ...
