"""The evidence bundle: the exact bytes that get hashed and anchored on-chain.

Everything downstream depends on one property - that a bundle serialises to the
same bytes on every machine, forever. So serialisation is pinned here (sorted
keys, no insignificant whitespace, UTF-8) rather than left to whatever
``json.dumps`` defaults happen to be in force.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from eth_utils import keccak

from . import SCHEMA

SUBJECT_DOMAIN = b"sigil:subject:v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class ProbeRef:
    """The input face, described without ever storing the face itself."""

    image_sha256: str
    embedding_sha256: str
    backend: str
    model: str
    bbox: list[int]
    det_score: float


@dataclass
class MatchRef:
    """A single piece of discovered public content that matched the probe."""

    platform: str
    post_url: str
    post_uri: str
    author_handle: str
    author_did: str
    author_display_name: str
    text: str
    image_url: str
    image_sha256: str
    created_at: str
    discovered_via: str


@dataclass
class Evidence:
    probe: ProbeRef
    match: MatchRef
    similarity: float
    threshold: float
    searched_at: str
    search_trace: list[dict[str, Any]] = field(default_factory=list)
    schema: str = SCHEMA

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Similarity is a float from a model; pin the precision so that a
        # re-run's floating-point noise cannot change the hash.
        d["similarity"] = round(float(self.similarity), 6)
        d["threshold"] = round(float(self.threshold), 6)
        return d

    def canonical_json(self) -> bytes:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")

    def evidence_hash(self) -> bytes:
        """keccak256 over the canonical bytes - this is the bytes32 anchored."""
        return keccak(self.canonical_json())

    def evidence_hash_hex(self) -> str:
        return "0x" + self.evidence_hash().hex()

    def similarity_bps(self) -> int:
        """Similarity as basis points, clamped to uint32 range for on-chain storage."""
        return max(0, min(65535, int(round(self.similarity * 10000))))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Evidence:
        return cls(
            probe=ProbeRef(**d["probe"]),
            match=MatchRef(**d["match"]),
            similarity=d["similarity"],
            threshold=d["threshold"],
            searched_at=d["searched_at"],
            search_trace=d.get("search_trace", []),
            schema=d.get("schema", SCHEMA),
        )


def subject_ref(embedding_sha256: str, salt: str) -> bytes:
    """A salted commitment to the probe face.

    Publishing a face embedding on a public ledger would be an irreversible
    biometric disclosure, so the chain gets this instead: a salted hash over the
    embedding's digest. Anyone holding the original probe image can recompute it
    and prove the record refers to that face; nobody can run it backwards to
    recover the face from chain data.

    It is derived from the digest rather than the raw vector so that every path
    - full pipeline, standalone anchor, standalone verify - produces the same
    commitment from the bundle alone.
    """
    return keccak(SUBJECT_DOMAIN + salt.encode("utf-8") + bytes.fromhex(embedding_sha256))
