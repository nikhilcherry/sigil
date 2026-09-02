# sigil

[![ci](https://github.com/nikhilcherry/sigil/actions/workflows/ci.yml/badge.svg)](https://github.com/nikhilcherry/sigil/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![licence](https://img.shields.io/badge/licence-MIT-green)](LICENSE)

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

### Where each part lives

| stage | implementation | see |
|---|---|---|
| Face detection + encoding | RetinaFace + ArcFace `w600k_r50`, 512-d; OpenCV YuNet + SFace as a keyless fallback | [`sigil/face/`](sigil/face/) · [§1](#1--face-encoding) |
| Face → name | local index harvested from Wikipedia pageviews → Wikidata `P31=Q5` → Commons portraits | [`sigil/identify.py`](sigil/identify.py) · [§2](#2--identification--turning-a-face-into-a-name) |
| Live social search | anonymous AT Protocol (`searchActors`, `getAuthorFeed`); Google Lens via SerpAPI when configured | [`sigil/search/`](sigil/search/) · [§3](#3--web--social-search) |
| Blockchain record | keccak256 over a canonical evidence bundle → append-only Solidity registry, on a persisted local py-evm chain or any EVM node | [`contracts/SigilRegistry.sol`](contracts/SigilRegistry.sol) · [§4](#4--blockchain-verification) |
| Re-verification | recompute the hash and check it against chain state, not against a log | `sigil verify` · [below](#verifying-and-proving-that-verification-bites) |

The whole sequence, end to end, is `./scripts/demo.sh`.

---

## Quickstart

```bash
git clone https://github.com/nikhilcherry/sigil && cd sigil
python3 -m venv .venv && .venv/bin/pip install -e ".[insight,dev]"
.venv/bin/sigil run examples/probe-aoc.jpg -q "AOC"
```

That is the whole demo. No API keys, no wallet, no testnet funds, no Node.js.
The first run downloads the ArcFace model pack (~300 MB) and a `solc` binary.

Prefer to watch it happen? `sigil serve` opens a local web UI at
<http://127.0.0.1:8099> — drop a face, type a query, and the pipeline streams
live: candidates appear as thumbnails the moment they are downloaded and
encoded, sorted by similarity, with the ones clearing the threshold lit up. Same
pipeline, same code path; the CLI and the UI are two front ends over one
`run_pipeline`.

Prefer no heavy model download? `pip install -e ".[dev]"`, run
`./scripts/fetch_models.sh` (37 MB), and everything works on the OpenCV backend.

The encoder runs on the GPU automatically when onnxruntime has a working CUDA
provider, and on CPU otherwise. `sigil backends` reports which one actually
answered — not which one was requested, since an unusable CUDA provider can
warn and quietly serve CPU.

### What a run looks like

```
Stage 1/5 · Face scan
  backend           insightface (buffalo_l/w600k_r50)
  face bbox         [244, 168, 481, 492]
  embedding sha256  b279a48ef65f075f…

Stage 2/5 · Web / social search
  images fetched     185
  images with a face  45
  faces compared     151
  duplicate images    17  (score reused)
  live API calls      10

  #   similarity   account                       found via
  1       0.7596   aoc.bsky.social               actor.searchActors:avatar
  2       0.6920   aoc.bsky.social               feed.getAuthorFeed
  3       0.5660   africanprincess7.bsky.social  feed.getAuthorFeed

Stage 3/5 · Face match          0.7596  (threshold 0.380)
Stage 4/5 · Blockchain anchor   tx 0x9c202d06…  block 2  gas 114,222
Stage 5/5 · Re-verification     VERIFIED
```

---

## The stages

### 1 · Face encoding

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
| probe vs. the same person's live Bluesky avatar | **0.67 – 0.76** | **0.74** |
| probe vs. a different person (`control-buttigieg.jpg`) | **−0.02** | below threshold |

Both backends find the same live match on a clean clone, so the fallback is a
real alternative rather than a degraded mode.

That gap is what makes a fixed threshold defensible, and `tests/test_face.py`
asserts it on every run so a model or preprocessing change cannot quietly erode
it.

### 2 · Identification — turning a face into a name

This stage exists because of a hole in the obvious design. Bluesky has no face
index, so the social search can only be seeded by *text*. That leaves the
pipeline unable to answer the question the task actually asks — "who is this?" —
because you would have had to know the answer in order to ask. Worse, a query
left over from a previous run silently searches the wrong person and reports
"no match", which reads as the tool failing rather than as a mistake.

So `sigil` builds a local face→name index of public figures:

1. Wikipedia's **most-viewed articles** across ten language editions (English
   plus Hindi, Tamil, Telugu, Malayalam, Bengali, Marathi, Kannada, Spanish,
   French) — this is where public figures concentrate, and the non-English
   editions matter a great deal for non-Western coverage.
2. Each title is resolved through **Wikidata**, keeping only items whose `P31`
   (instance of) includes `Q5` (human). Without that check the index fills with
   film posters and album covers, which do contain faces.
3. The **Wikimedia Commons** portrait for each person is downloaded and encoded
   with the same face model as everything else.

```bash
sigil index build            # harvest + encode; keyless, no API keys at all
sigil index build --limit 500  # a smaller one, if you just want to try the path
sigil index info             # what's in it
sigil identify photo.jpg     # just the naming step
```

Measured on one laptop: 3,820 people harvested from the ten wikis in about four
minutes, then roughly an hour to download and encode their portraits — 3,583
usable faces, since a portrait with no detectable face is dropped. Encoding is
effectively all of that hour, at about 0.8 s per portrait on CPU, so `--limit`
is the knob that matters if you only want to see the path work. An interrupted
build keeps whatever it had already encoded and marks the index partial.

Then a query becomes optional. Omit it and the face is named first, and the name
seeds the social search:

```bash
sigil run photo.jpg          # who is this → find their posts → anchor
sigil run photo.jpg -q "…"   # skip identification, search directly
```

Against that index the committed example probe names itself at **0.9720**
cosine, with the runner-up at 0.2105 — the gap, not the absolute number, is
what makes the call safe.

Naming a stranger is a higher-stakes call than confirming a match you were
already looking for, so this stage uses a stricter bar (0.45 vs 0.38 cosine).
Below it, the pipeline says it does not know rather than guessing — a wrong name
would send the social search after the wrong person entirely.

**This does not make it a whole-web face search.** The index covers people
notable enough for a well-viewed encyclopaedia article. A private individual is
not in it and will not be found, which is a deliberate boundary rather than a
gap to close: being in this index is a consequence of public notability, and the
whole thing is reproducible from public sources by anyone who runs the builder.

### 3 · Web / social search

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

### 4 · Blockchain verification

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
#   SIGIL_RPC_URL=https://polygon-amoy-bor-rpc.publicnode.com
#   SIGIL_PRIVATE_KEY=0x…        throwaway key, testnet funds only
sigil chain address              # which address to fund, and whether it is yet
sigil chain info                 # deploys the registry, prints the address
sigil run examples/probe-aoc.jpg -q "AOC"
```

`chain address` exists because `chain info` deploys before it can report, and
on an unfunded key that fails — leaving no way to find out which address to
fund without first trying to spend from it.

Set `SIGIL_CONTRACT` to the printed address afterwards to reuse the deployment.
Anchoring costs **114,222 gas for the first record and 97,122 for every one
after** — the difference is the registry's array-length slot going from cold to
warm, not anything about the evidence. The run prints a block-explorer link for
the transaction.

Two notes from actually testing this. First, `rpc-amoy.polygon.technology` — the
endpoint most guides cite — was unreachable during development;
`polygon-amoy-bor-rpc.publicnode.com` and `polygon-amoy.drpc.org` both work
(chain id 80002). Second, the RPC path here has been exercised end to end
against a live Amoy node up to the point of funding: it connects, builds and
prices the deployment transaction, and stops at `insufficient funds` with an
unfunded key. Fund a throwaway address from an Amoy faucet and it deploys for
real — that is the only step that needs a human.

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

The web UI has the same two buttons. Tampering there marks every differing
nibble of the digest in red, which makes the avalanche visible: edit one
character of a post's text and essentially the whole hash changes.

### The web UI

```bash
sigil serve            # http://127.0.0.1:8099
```

One HTML file served by `http.server`, with pipeline events pushed over
Server-Sent Events. No framework, no build step, no CDN, no fonts fetched at
runtime — it works with the network cable pulled, apart from the search itself.

It binds to `127.0.0.1` and has no authentication, deliberately. This tool
searches for people by face; it has no business being reachable from off-box.
`--host` exists but you should have a good reason to use it.

---

## Commands

| command | stage | what it does |
|---|---|---|
| `sigil run IMAGE [-q QUERY]` | all | the full pipeline; omit `-q` to identify first |
| `sigil scan IMAGE` | 1 | detect and encode only |
| `sigil identify IMAGE` | 2 | name the face from the local index |
| `sigil index build` / `info` | — | build or inspect the face→name index |
| `sigil search IMAGE -q QUERY` | 1–3 | search and match, no chain |
| `sigil anchor` | 4 | anchor an existing bundle |
| `sigil verify` | 5 | re-verify against chain state |
| `sigil tamper` | — | produce an altered bundle to prove verification fails |
| `sigil serve` | 1–5 | local web UI, streaming the run live |
| `sigil chain info` / `reset` | — | inspect or wipe the chain backend |
| `sigil chain address` | — | show the submitter address and balance, deploying nothing |
| `sigil backends` | — | report which backends load, and what they run on |

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
pytest -m "not network"   # 225 offline tests, 93% line coverage
pytest -m network         # 3 tests against the live API and a live chain
```

The network tests are deliberate. The realistic failure mode of this project is
not a logic bug — it is Bluesky changing a response shape or an auth
requirement. A green offline suite over a broken search is exactly what needs
catching.

The offline suite covers the whole pipeline without a network: the end-to-end
run from face to anchored record and back out of chain state, the guards that
refuse a hostile candidate download, both optional search arms, every panel the
demo prints, and the command sequence `scripts/demo.sh` itself runs.

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

- **Not a whole-web face search.** Bluesky exposes no face index, so retrieval
  is seeded by *text*. The identity index closes that loop for public figures —
  a face becomes a name, and the name seeds the search — but it only covers
  people notable enough for a well-viewed Wikipedia article. A private
  individual is not in it and will not be found. This is a genuine search over
  live data; it is not equivalent to PimEyes-style indexing of the open web, and
  it is not intended to be.
- **Identification can be confidently wrong.** The index returns the nearest
  known face, and "nearest" is not "correct". Lookalikes and poor-quality
  portraits are the failure mode. Candidates below 0.45 cosine are rejected
  rather than reported, and the runner-up scores are always shown so the margin
  is visible.
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
