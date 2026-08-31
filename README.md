# sigil

**Face scan → live social search → tamper-evident on-chain record.**

Give `sigil` a photograph of a face. It encodes the face, performs a real search
against live social platforms, verifies any candidate with the same face model,
and writes a hash of the resulting evidence to a blockchain. Anyone holding the
evidence file can later prove — against chain state, not against a log file —
that it has not been altered since.

```
 photo ──▶ detect + encode ──▶ live search ──▶ face-verify ──▶ keccak256 ──▶ chain
           ArcFace 512-d        AT Protocol     cosine vs        canonical      append-only
                                Google Lens     threshold        bundle         registry
```

Nothing in the pipeline is pre-picked. The search hits the network on every run,
and the match is decided by the face model rather than asserted by the code — if
no candidate clears the threshold, **nothing is anchored**, which is the correct
outcome rather than a failure.

---

## Quickstart

```bash
git clone https://github.com/nikhilcherry/sigil && cd sigil
python3 -m venv .venv && .venv/bin/pip install -e ".[insight,dev]"
.venv/bin/sigil run examples/probe-aoc.jpg -q "AOC"
```

That is the whole demo. No API keys, no wallet, no testnet funds, no Node.js.
The first run downloads the ArcFace model pack (~300 MB) and a `solc` binary.

Prefer no heavy model download? `pip install -e ".[dev]"`, run
`./scripts/fetch_models.sh` (37 MB), and everything works on the OpenCV backend.

### What a run looks like

```
Stage 1/5 · Face scan
  backend           insightface (buffalo_l/w600k_r50)
  face bbox         [244, 168, 481, 492]
  embedding sha256  b279a48ef65f075f…

Stage 2/5 · Web / social search
  images fetched      57
  images with a face   9
  live API calls       5

  #   similarity   account          found via
  1       0.7585   aoc.bsky.social  actor.searchActors:avatar
  2       0.0778   pfrazee.com      feed.getAuthorFeed

Stage 3/5 · Face match          0.7585  (threshold 0.380)
Stage 4/5 · Blockchain anchor   tx 0x9c202d06…  block 2  gas 114,222
Stage 5/5 · Re-verification     VERIFIED
```

---

## The three stages

### 1 · Face identification

Two interchangeable backends, both emitting L2-normalised vectors compared by
cosine similarity:

| backend | detector + recogniser | dims | default threshold | cost |
|---|---|---|---|---|
| `insightface` *(default)* | RetinaFace + ArcFace `w600k_r50` | 512 | 0.38 | ~300 MB model pack |
| `opencv` *(fallback)* | YuNet + SFace | 128 | 0.363 | 37 MB, no extra deps |

The threshold travels with the backend because the two models put same-person
and different-person pairs at different scales. Measured on the committed
example images:

| pair | insightface | opencv |
|---|---|---|
| probe vs. the same person's live Bluesky avatar | **0.67 – 0.76** | — |
| probe vs. a different person (`control-buttigieg.jpg`) | **−0.02** | below threshold |

That gap is what makes a fixed threshold defensible, and `tests/test_face.py`
asserts it on every run so a model or preprocessing change cannot quietly erode
it.

### 2 · Web / social search

**Bluesky (AT Protocol)** is the primary provider, chosen because its AppView
serves `app.bsky.actor.searchActors`, `app.bsky.actor.getProfile` and
`app.bsky.feed.getAuthorFeed` to *anonymous* callers. A fresh clone therefore
performs a genuine live search with no credentials whatsoever.

The search is a retrieval step, not the answer:

1. `searchActors(query)` → real accounts matching the query terms
2. for each account → profile avatar, plus `getAuthorFeed(filter=posts_with_media)`
3. every discovered image is downloaded and run through **the same encoder** as the probe
4. cosine similarity decides the match; everything below threshold is discarded

Supplying `BLUESKY_APP_PASSWORD` additionally unlocks `app.bsky.feed.searchPosts`
(the one endpoint that requires a session), widening the net beyond accounts that
match by name. Without it that endpoint is skipped rather than faked.

**Google Lens via SerpAPI** is the optional open-web arm. It activates only when
`SERPAPI_KEY` is set *and* the probe is passed as a public `https://` URL, because
Lens matches on a URL rather than an upload. With a local file it is skipped.

Every network call is recorded into the evidence bundle's `search_trace` — the
endpoints hit, the parameters sent, and the result counts — so a run can be
audited after the fact rather than taken on trust.

### 3 · Blockchain verification

`contracts/SigilRegistry.sol` is a small append-only registry:

```solidity
struct Record {
    address submitter;      // who anchored it
    uint64  anchoredAt;     // block timestamp
    uint32  similarityBps;  // cosine similarity, basis points
    bytes32 subjectRef;     // salted commitment to the probe face
}
function anchor(bytes32 evidenceHash, uint32 similarityBps, bytes32 subjectRef) external;
```

Re-anchoring an existing hash **reverts**. That is what makes the record
tamper-*evident* rather than merely tamper-resistant: nobody, including the
original submitter, can overwrite a record or backdate one, so the earliest
timestamp for a bundle stands permanently.

What is anchored is `keccak256` over the **canonical serialisation** of the
evidence bundle — sorted keys, no insignificant whitespace, UTF-8, similarity
rounded to a fixed precision so inference jitter cannot move the hash. The file
written to `artifacts/evidence.json` is byte-for-byte the preimage that was
hashed, so there is no re-serialisation step that could drift from what the
chain saw.

**Two chain backends, one code path:**

| `SIGIL_CHAIN` | what it is | setup |
|---|---|---|
| `local` *(default)* | in-process py-evm chain, **state persisted to disk** | none |
| `rpc` | any EVM node — Polygon Amoy, Sepolia, mainnet | RPC URL + funded key |

The local chain is not the usual throwaway test chain. Its key/value store is
snapshotted after every write and rehydrated on load, so a record written by
`sigil anchor` is genuinely read back out of chain state by a later, separate
`sigil verify` process — not replayed from a file.

**To use a public testnet** (Polygon Amoy shown):

```bash
cp .env.example .env      # then set:
#   SIGIL_CHAIN=rpc
#   SIGIL_RPC_URL=https://rpc-amoy.polygon.technology
#   SIGIL_PRIVATE_KEY=0x…        throwaway key, testnet funds only
sigil chain info                 # deploys the registry, prints the address
sigil run examples/probe-aoc.jpg -q "AOC"
```

Set `SIGIL_CONTRACT` to the printed address afterwards to reuse the deployment.
Anchoring costs ~114k gas. The run prints a block-explorer link for the
transaction.

---

## Verifying, and proving that verification bites

```bash
sigil verify --probe examples/probe-aoc.jpg --recheck-source
```

Five independent checks:

| check | what it proves |
|---|---|
| record on chain | the bundle's hash exists in the registry |
| similarity matches | the score was not edited after anchoring |
| subject commitment | the record refers to *this* face |
| probe re-encodes *(`--probe`)* | the supplied photo really produces the bundle's embedding |
| source image intact *(`--recheck-source`)* | the discovered post has not been edited or deleted since |

An optional check that was not requested reads as `None`, never as a pass.

**Demonstrating tamper-evidence.** `sigil tamper` alters one field of a bundle:

```bash
$ sigil tamper --field match.text
match.text: 'Waitress turned Congresswoman…' -> 'Waitress turned Congresswoman…!'
original hash : 0x2cbb368f6fe3851c12c74ec4d637d20d4d645ec714a42d6e678d0ecdea609315
tampered hash : 0xe09a2916407c53d12a483463837ca39c0f165cb624244a1365ccfe5fc30bd604

$ sigil verify -e artifacts/evidence.tampered.json
Verification: NOT VERIFIED   (exit 1)
  record on chain     FAIL
  note  No record for this evidence hash. Either it was never anchored,
        or the bundle has been modified since it was.
```

One character changes the hash, and the altered bundle has no chain record.
`./scripts/demo.sh` runs this whole sequence end to end for a screen recording.

---

## Commands

| command | stage | what it does |
|---|---|---|
| `sigil run IMAGE -q QUERY` | 1–5 | the full pipeline |
| `sigil scan IMAGE` | 1 | detect and encode only |
| `sigil search IMAGE -q QUERY` | 1–3 | search and match, no chain |
| `sigil anchor` | 4 | anchor an existing bundle |
| `sigil verify` | 5 | re-verify against chain state |
| `sigil tamper` | — | produce an altered bundle to prove verification fails |
| `sigil chain info` / `reset` | — | inspect or wipe the chain backend |
| `sigil backends` | — | report which face backends this machine can load |

`sigil run` accepts a local path or an `https://` URL, and exits `0` on a
verified match, `2` when nothing cleared the threshold, `1` on a failed
verification — so it composes in a script.

## Configuration

Everything is optional; see `.env.example`. The knobs that matter most:
`SIGIL_FACE_BACKEND`, `SIGIL_THRESHOLD`, `SIGIL_MAX_IMAGES`, `SIGIL_CHAIN`,
and `SIGIL_SUBJECT_SALT` (change it and previously anchored records stop
verifying against new probes — pick one and keep it).

## Tests

```bash
pytest -m "not network"   # 47 offline tests
pytest -m network         # 3 tests against the live API and a live chain
```

The network tests are deliberate. The realistic failure mode of this project is
not a logic bug — it is Bluesky changing a response shape or an auth
requirement. A green offline suite over a broken search is exactly what needs
catching.

---

## Limitations

**This is a surveillance-shaped tool, and that is worth being direct about.**
Searching for a person's online presence from their face, without their
knowledge or consent, is the core capability here. Under the EU GDPR (Art. 9)
and India's DPDP Act, a face embedding is biometric personal data, and
processing it generally requires an explicit lawful basis that "I found the
photo publicly" does not supply. Several jurisdictions have fined companies
specifically for building face-search indexes from public web images. Use this
on yourself, on a consenting subject, or on a public figure for a demonstration.

**On the technique itself:**

- **Not a face-search engine.** Bluesky exposes no face index, so retrieval is
  seeded by *text* — the query terms find candidate accounts, and only then does
  the face model verify them. A person whose account is unrelated to any query
  you supply will not be found. This is a genuine search over live data, but it
  is not equivalent to PimEyes-style whole-web face indexing.
- **False positives are possible.** A cosine threshold is a decision boundary,
  not proof of identity. Similarity is recorded on-chain precisely so a
  downstream reader can apply their own bar. Twins, close relatives, and heavily
  filtered photos are the known hard cases, and accuracy degrades with age gaps,
  pose, occlusion, and demographic distribution — published face-recognition
  benchmarks show materially different error rates across demographic groups,
  and nothing here corrects for that.
- **The chain proves integrity, not truth.** Anchoring establishes that a
  bundle existed at a time and has not changed. It says nothing about whether
  the match was *correct*. A confidently wrong match, anchored, is a permanent
  record of a confidently wrong match.
- **No biometrics on chain — by design.** The registry stores a salted
  commitment to the probe, never an embedding or an image. Publishing a face
  vector to an append-only public ledger would be an irreversible biometric
  disclosure with no deletion path, which is squarely at odds with the erasure
  rights the regulations above grant.
- **The local chain is a real EVM, not a real network.** py-evm executes the
  actual bytecode and persists state, but it is single-party: it proves
  integrity against your own record, not against a public consensus. Use
  `SIGIL_CHAIN=rpc` for a record a third party can independently check.
- **Search breadth is bounded by cost, not capability.** `SIGIL_MAX_IMAGES`
  defaults to 200. Raising it widens recall and lengthens runtime roughly
  linearly.
- **SerpAPI needs a hosted probe.** Google Lens matches on a URL, so the
  open-web arm cannot run against a local file.

## Licence

MIT — see `LICENSE`. Example images are public domain or CC BY-SA; see
`examples/README.md` for per-image attribution.
