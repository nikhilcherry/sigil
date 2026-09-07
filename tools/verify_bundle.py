#!/usr/bin/env python3
"""Recompute an evidence bundle's hash without sigil, and without any dependency.

Anchoring a digest is only worth something if somebody other than the tool that
wrote it can arrive at the same digest. Everywhere else in this project that
number comes out of `eth_utils.keccak` via `sigil.evidence`, so "the hash is
right" has so far meant "sigil agrees with itself" - which is exactly the shape
of claim this project refuses to make about a face match.

So this file is deliberately isolated. It imports nothing from sigil and
nothing outside the standard library: the canonical serialisation is written
out again from the specification below, and Keccak-256 is implemented here from
scratch rather than taken from a library. If this and sigil agree on a bundle,
two independent implementations agree, and the digest is a property of the
bundle rather than of the code that produced it.

It is also the answer to a practical question. Someone handed an evidence
bundle and a transaction hash should not have to install a face-recognition
stack to check that the two correspond. This is 200 lines of stdlib Python and
a `python3 tools/verify_bundle.py evidence.json`.

    $ python3 tools/verify_bundle.py evidence.json
    0x6e3b4a19b9f450c63453249aba20892d592bd52aa9293d8a17622275da1e2119

    $ python3 tools/verify_bundle.py evidence.json --expect 0x6e3b...
    MATCH

**The canonical form, stated so it can be reimplemented again by someone else:**
the bundle is serialised as JSON with keys sorted at every level, no whitespace
between tokens, non-ASCII characters left as themselves rather than escaped,
and encoded UTF-8. The digest is Keccak-256 (the original padding, 0x01 - not
SHA3-256's 0x06) over those bytes. That is the whole specification.

What this does *not* do is decide whether the bundle is true. It recomputes a
number. Whether that number is on a chain, whether the face in it is the right
face, and whether the source still says what the bundle says it said, are the
questions `sigil verify` answers.
"""

from __future__ import annotations

import argparse
import json
import sys

# --------------------------------------------------------------------- keccak
#
# Keccak-f[1600], written from the specification. The rate is 1088 bits (136
# bytes) for a 256-bit digest, and the padding byte is 0x01, which is what
# separates Ethereum's Keccak-256 from the later standardised SHA3-256.

MASK = (1 << 64) - 1

ROUND_CONSTANTS = (
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A,
    0x8000000080008000, 0x000000000000808B, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009, 0x000000000000008A,
    0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089,
    0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
    0x000000000000800A, 0x800000008000000A, 0x8000000080008081,
    0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
)

ROTATION = (
    (0, 36, 3, 41, 18),
    (1, 44, 10, 45, 2),
    (62, 6, 43, 15, 61),
    (28, 55, 25, 21, 56),
    (27, 20, 39, 8, 14),
)

RATE = 136  # bytes


def _rol(value: int, shift: int) -> int:
    shift %= 64
    return ((value << shift) | (value >> (64 - shift))) & MASK


def _permute(state: list[list[int]]) -> None:
    for rc in ROUND_CONSTANTS:
        # theta
        c = [state[x][0] ^ state[x][1] ^ state[x][2] ^ state[x][3] ^ state[x][4]
             for x in range(5)]
        d = [c[(x - 1) % 5] ^ _rol(c[(x + 1) % 5], 1) for x in range(5)]
        for x in range(5):
            for y in range(5):
                state[x][y] ^= d[x]

        # rho and pi
        b = [[0] * 5 for _ in range(5)]
        for x in range(5):
            for y in range(5):
                b[y][(2 * x + 3 * y) % 5] = _rol(state[x][y], ROTATION[x][y])

        # chi - the complement is written as an xor with the mask, since
        # Python's ~ produces a negative number rather than a 64-bit one
        for x in range(5):
            for y in range(5):
                state[x][y] = b[x][y] ^ ((b[(x + 1) % 5][y] ^ MASK) & b[(x + 2) % 5][y])

        # iota
        state[0][0] ^= rc


def keccak256(data: bytes) -> bytes:
    """Keccak-256 over `data`, as Ethereum computes it."""
    state = [[0] * 5 for _ in range(5)]

    padded = bytearray(data)
    padded.append(0x01)
    while len(padded) % RATE != 0:
        padded.append(0x00)
    padded[-1] |= 0x80

    for offset in range(0, len(padded), RATE):
        block = padded[offset:offset + RATE]
        for i in range(RATE // 8):
            lane = int.from_bytes(block[i * 8:(i + 1) * 8], "little")
            state[i % 5][i // 5] ^= lane
        _permute(state)

    out = bytearray()
    while len(out) < 32:
        for i in range(RATE // 8):
            if len(out) >= 32:
                break
            out += state[i % 5][i // 5].to_bytes(8, "little")
        if len(out) < 32:  # pragma: no cover - unreachable at this rate
            _permute(state)
    return bytes(out[:32])


# ------------------------------------------------------------------- the bundle

# Exactly the fields sigil hashes. Listed here rather than "whatever is in the
# file" on purpose: a bundle carrying an extra key would otherwise be hashed
# by this tool as though sigil had hashed it too, and sigil refuses such a file
# rather than attesting a subset of what its reader is looking at.
TOP_LEVEL = frozenset({
    "probe", "match", "similarity", "threshold", "searched_at",
    "search_trace", "schema",
})


def canonical_bytes(bundle: dict) -> bytes:
    """The exact preimage: sorted keys, no whitespace, unescaped, UTF-8."""
    return json.dumps(
        bundle, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def evidence_hash(bundle: dict) -> str:
    return "0x" + keccak256(canonical_bytes(bundle)).hex()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Recompute an evidence bundle's hash using no sigil code.")
    ap.add_argument("bundle", help="path to evidence.json")
    ap.add_argument("--expect", default=None,
                    help="a hash to compare against, e.g. from the chain")
    args = ap.parse_args(argv)

    try:
        with open(args.bundle, encoding="utf-8") as fh:
            bundle = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"cannot read {args.bundle}: {exc}", file=sys.stderr)
        return 2

    if not isinstance(bundle, dict):
        print("that file is not an evidence bundle", file=sys.stderr)
        return 2

    unknown = sorted(set(bundle) - TOP_LEVEL)
    if unknown:
        # The same refusal sigil makes, for the same reason: fields outside the
        # set below are not covered by the digest, so a file carrying them
        # cannot be honestly summarised by one number.
        print(f"this bundle has fields not covered by the hash: "
              f"{', '.join(unknown)}", file=sys.stderr)
        return 2

    digest = evidence_hash(bundle)
    if args.expect is None:
        print(digest)
        return 0

    expected = args.expect if args.expect.startswith("0x") else "0x" + args.expect
    if digest == expected.lower():
        print("MATCH")
        return 0
    print(f"MISMATCH\n  recomputed {digest}\n  expected   {expected.lower()}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
