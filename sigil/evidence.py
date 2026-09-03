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
    # Whole-image similarity to the probe, and what that makes this match:
    # "identity" for a different photograph of the same face, "provenance" for
    # the probe's own photograph found somewhere else. See sigil/provenance.py.
    probe_photo_similarity: float = 0.0
    claim: str = "identity"
    # "social" or "web" - which kind of arm found it. See sigil/search/base.py.
    source_kind: str = "web"


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
        # Same reason as similarity: a float straight from a computation would
        # let floating-point noise on a re-run change the anchored hash.
        d["match"]["probe_photo_similarity"] = round(
            float(self.match.probe_photo_similarity), 6
        )
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


def field_paths(data: dict[str, Any], prefix: str = "") -> list[str]:
    """Every dotted leaf path in a bundle dict, for error messages and help."""
    out: list[str] = []
    for key, value in data.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out.extend(field_paths(value, f"{path}."))
        else:
            out.append(path)
    return out


def alter_field(data: dict[str, Any], field: str, value: Any = None) -> tuple[Any, Any]:
    """Change one field of a bundle dict in place. Returns (before, after).

    This is the tamper demo's one job, and it is driven by a string the user
    typed - a CLI flag, or a JSON body from the browser. A mistyped path used
    to surface as a raw KeyError traceback, and a path landing on a list gave
    a TypeError from trying to add 0.0001 to it. Both are ordinary user error
    on a command whose whole purpose is to be run by hand during a demo, so
    they are refused with a message that says what the field names actually
    are.
    """
    parts = field.split(".")
    node: Any = data
    for depth, key in enumerate(parts[:-1]):
        if not isinstance(node, dict) or key not in node:
            raise ValueError(_no_such_field(data, ".".join(parts[: depth + 1])))
        node = node[key]

    leaf = parts[-1]
    if not isinstance(node, dict) or leaf not in node:
        raise ValueError(_no_such_field(data, field))

    before = node[leaf]
    if value is not None:
        after: Any = value
    elif isinstance(before, str):
        after = before + "!"
    elif isinstance(before, (int, float)) and not isinstance(before, bool):
        after = before + 0.0001
    else:
        raise ValueError(
            f"{field} holds a {type(before).__name__}, which has no obvious "
            f"small edit - pass --value to set one explicitly"
        )
    node[leaf] = after
    return before, after


def _no_such_field(data: dict[str, Any], field: str) -> str:
    return (f"no field {field!r} in this bundle. Available: "
            + ", ".join(sorted(field_paths(data))))
