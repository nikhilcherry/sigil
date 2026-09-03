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

| | |
|---|---|
| **what it does** | above, and [§1–§5](#the-stages) stage by stage |
| **how to run it** | [Quickstart](#quickstart) — one command, no API keys, no wallet, no testnet funds |
| **which blockchain** | [§4](#4--blockchain-verification): a persisted in-process **py-evm** chain by default, so a clone anchors for real with no setup; any EVM node via `SIGIL_CHAIN=rpc`, exercised against **Polygon Amoy** |
| **known limitations** | [Limitations](#limitations) — including the measured false-accept and wrong-name rates, and what this tool is not |

### Where each part lives

| stage | implementation | see |
|---|---|---|
| Face detection + encoding | RetinaFace + ArcFace `w600k_r50`, 512-d; OpenCV YuNet + SFace as a keyless fallback | [`sigil/face/`](sigil/face/) · [§1](#1--face-encoding) |
| Face → name | local index harvested from Wikipedia pageviews → Wikidata `P31=Q5` → Commons portraits | [`sigil/identify.py`](sigil/identify.py) · [§2](#2--identification--turning-a-face-into-a-name) |
| Live social search | anonymous AT Protocol (`searchActors`, `getAuthorFeed`); Google Cloud Vision web detection and Google Lens when configured | [`sigil/search/`](sigil/search/) · [§3](#3--web--social-search) |
| Evidence weighting | whole-image fingerprint separates *this photo again* from *another photo of that face*; social sources outrank open-web ones | [`sigil/provenance.py`](sigil/provenance.py) · [§3](#3--web--social-search) |
| Blockchain record | keccak256 over a canonical evidence bundle → append-only Solidity registry, on a persisted local py-evm chain or any EVM node | [`contracts/SigilRegistry.sol`](contracts/SigilRegistry.sol) · [§4](#4--blockchain-verification) |
| Re-verification | recompute the hash and check it against chain state, not against a log | `sigil verify` · [below](#verifying-and-proving-that-verification-bites) |
| Threshold calibration | 6.4 M real impostor pairs against genuine pairs harvested across language Wikipedias | [`sigil/calibrate.py`](sigil/calibrate.py) · [§5](#5--calibration--what-the-threshold-actually-costs) |

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

If a future dependency release breaks that install, `requirements.lock` is the
exact set this was last verified against — `pip install -r requirements.lock -e .`
reproduces it. The dependency ranges in `pyproject.toml` stay deliberately
loose so CI installs whatever resolves latest and a breaking upstream release
shows up as a red badge rather than silently; the lockfile is the escape hatch,
not the default. Verified by installing a fresh clone from it alone.

The encoder runs on the GPU automatically when onnxruntime has a working CUDA
provider, and on CPU otherwise. `sigil backends` reports which one actually
answered — not which one was requested, since an unusable CUDA provider can
warn and quietly serve CPU.

<details>
<summary><b>The GPU path, verified — 12.4× and three things that will bite you</b></summary>

Measured on an RTX 5050 (Blackwell, driver 580) against the same 100 real
candidate images:

| | per image | 100 images | faces found |
|---|---|---|---|
| `CPUExecutionProvider` | 223 ms | 22.3 s | 86 |
| `CUDAExecutionProvider` | **18 ms** | **1.8 s** | 86 |

Identical detections, **12.4× faster**. That is the difference between a ~50 s
run and a ~5 s one, and between an hour-long index build and about five
minutes. No code change is involved — the automatic detection above simply
finds the provider.

Getting there took three steps that are not obvious:

1. **Install `onnxruntime-gpu` *after* sigil, not before.** `onnxruntime` is a
   hard dependency in `pyproject.toml`, so installing the package replaces the
   GPU build with the CPU one. Doing it in the wrong order looks like success
   and silently gives you CPU.
2. **onnxruntime 1.29 wants CUDA 13 and cuDNN 9**, and the `nvidia-*-cu13`
   names on PyPI are stubs that fail to build. The real wheels come from
   NVIDIA's index: `uv pip install --extra-index-url https://pypi.nvidia.com
   nvidia-cudnn-cu13 nvidia-cublas nvidia-curand nvidia-cufft
   nvidia-cuda-runtime nvidia-nvjitlink`.
3. **Make those wheels' libraries findable.** The provider is `dlopen`ed and
   links CUDA and cuDNN by soname, and the wheels put them in directories the
   linker does not search, so otherwise it loads halfway and reports the first
   missing `.so` it hits. `LD_LIBRARY_PATH` works but has to be set before the
   interpreter starts, for every invocation, including the ones inside
   `scripts/demo.sh`. The tidier fix is a one-line `.pth` in the venv's
   `site-packages` that imports a module preloading them with `RTLD_GLOBAL`:

   ```
   # zzz-sigil-cuda-preload.pth
   import sigil_cuda_preload
   ```

   A `.pth` rather than the more obvious `sitecustomize.py`, because a conda
   base interpreter ships its own `sitecustomize.py` earlier on `sys.path` and
   wins — which costs an afternoon if you do not know it.

Each failure along the way looked exactly like the one this project already
warns about: onnxruntime advertised `CUDAExecutionProvider` from the package
build, failed to load it, and served CPU while `get_available_providers()`
still listed it. That is precisely why `sigil backends` reads the provider back
off the loaded session instead of reporting what was asked for.

**Mixing providers is safe for everything that compares similarities**, which
is what makes the hour-long index build portable. The same 40 images encoded on
both give a cosine agreement of 0.9996 at worst and 0.99996 on average — float
noise, two orders of magnitude below the 0.06 that merely rescaling an image
costs. End to end: the committed probe against the same CPU-built index names
itself at 0.9720/0.2105 on CPU and 0.9722/0.2107 on CUDA.

**It is not safe for the one check that compares a digest.** The bundle records
`embedding_sha256` rather than the vector — deliberately, so it carries no
biometric — so `sigil verify --probe` has only an exact comparison available,
and sha256 turns that 0.9996 agreement into two digests with nothing in common.
Anchor a bundle on CPU, verify it on CUDA, and `probe re-encodes` **fails on an
intact bundle and the correct photograph**. That is real and it is reproducible:

```
$ SIGIL_GPU=0 sigil verify -e evidence.json --probe examples/probe-aoc.jpg
    probe re-encodes  PASS            → VERIFIED

$ sigil verify -e evidence.json --probe examples/probe-aoc.jpg
    probe re-encodes  FAIL            → NOT VERIFIED
    note  this bundle's embedding was produced on CPUExecutionProvider and this
          run is on CUDAExecutionProvider. The same image gives different digests
          on different providers, so this is a provider mismatch rather than a
          different photograph or an altered bundle.
```

The check stays exact, because loosening an integrity check to paper over this
would be the wrong trade. Instead the bundle records which provider encoded it
(that is what moved the schema to `v3`) so the failure can say what it actually
is. A bundle from before that field existed gets a weaker version of the same
note.

None of this is committed. `pyproject.toml` depends on plain `onnxruntime`,
CI and a clean clone run on CPU, and the venv is gitignored — so the numbers
this README quotes elsewhere are the CPU ones, and `sigil backends` is how you
tell which you actually have.

</details>

### What a run looks like

```
Stage 1/5 · Face scan
  backend           insightface (buffalo_l/w600k_r50)
  face bbox         [244, 168, 481, 492]
  embedding sha256  b279a48ef65f075f…

Stage 2/5 · Web / social search
  providers          bluesky, google-vision-web
  images fetched     168
  images with a face  90
  faces compared     106
  duplicate images     7  (score reused)
  live API calls       7

   #    similarity   claim            account                  found via
   1        0.9952   same photo       www.influencewatch.org   vision:pagesWithMatching
   2        0.9935   same photo       aflcio.org               vision:pagesWithMatching
   3        0.9931   same photo       www.ebay.com             vision:pagesWithMatching
   6        0.9892   different photo  images.squarespace-cdn   vision:fullMatchingImages
   7        0.9892   same photo       pbs.twimg.com            vision:fullMatchingImages
  70 ◀      0.7596   different photo  aoc.bsky.social          actor.searchActors:avatar

  ◀ anchored. A social post outranks an open-web page, and a different
    photograph of the same face outranks a higher cosine on the probe's own
    picture republished.

Sixty-nine candidates score higher by raw cosine and every one of them is the
probe's own photograph on someone else's page. The anchored match is the one
account that posted a *different* picture of that face.

Stage 3/5 · Face match          0.7596  (threshold 0.380, identity claim)
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
real alternative rather than a degraded mode. Verified from an actual fresh
clone of `main` on 2026-09-03, with no API keys in the environment and the
insightface extra not installed: `pip install -e ".[dev]"`,
`./scripts/fetch_models.sh`, then all **494 offline tests** collected and 486
green at 96% coverage — above the 95% floor its own CI enforces, the eight
skips being the insightface backend that install does not have — and the
documented quickstart matched `@aoc.bsky.social` at 0.7411 against the opencv
threshold of 0.363, picture-vs-probe 0.0728, so a genuinely different
photograph, anchored at 114,222 gas.

`./scripts/demo.sh` then ran end to end on that clone in **32-35 seconds**
(three consecutive runs: 35 s cold, then 32 s twice) with no tracebacks: it
named the face, found the post, anchored it, re-verified all six checks,
rejected a tampered bundle, listed the registry's contents, and skipped the
identity index and calibration it has not built yet rather than failing on
them. That is the whole deliverable, from `git clone`, with nothing
configured.

That gap is what makes a fixed threshold defensible, and `tests/test_face.py`
asserts it on every run so a model or preprocessing change cannot quietly erode
it. Two images is an illustration rather than evidence, though — for the
measured version, see [§5](#5--calibration--what-the-threshold-actually-costs).

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
is the knob that matters if you only want to see the path work.

Fetching the API's 640px thumbnails instead of the full-resolution originals
looks like the obvious fix and is not: measured over 26 portraits it cuts the
bytes fourfold and changes neither the download time (latency, not bandwidth)
nor the encode time (the detector normalises resolution anyway), while shifting
the embeddings — 0.9886 mean agreement, 0.9323 at worst — against a 0.45
threshold. The hour is inference, and inference does not care how many pixels
arrived. An interrupted
build keeps whatever it had already encoded and marks the index partial.

Then a query becomes optional. Omit it and the face is named first, and the name
seeds the social search:

```bash
sigil run photo.jpg          # who is this → find their posts → anchor
sigil run photo.jpg -q "…"   # skip identification, search directly
```

Against that index the committed example probe names itself at **0.9720**
cosine, with the runner-up at 0.2105 — the gap, not the absolute number, is
what makes the call safe. Those are the CPU figures; the same probe encoded on
CUDA against the same CPU-built index gives 0.9722 and 0.2107, which is the
whole of the cross-provider difference and a good illustration of why only the
gap is worth quoting.

Naming a stranger is a higher-stakes call than confirming a match you were
already looking for, so this stage uses a stricter bar (0.45 vs 0.38 cosine).
Below it, the pipeline says it does not know rather than guessing — a wrong name
would send the social search after the wrong person entirely. That bar is
measured, not asserted, and the measurement is unflattering: at 0.45 it still
names **1.73%** of faces the index does not contain, because one query is 3,583
chances to be wrong
([§5](#5--calibration--what-the-threshold-actually-costs)). At 0.38 it would be
4.16%.

**The built index is deliberately not shipped.** It is 3,583 face embeddings of
real, named people, and committing that to a public repository is the same
irreversible biometric disclosure this project refuses to put on a chain — the
difference between a git history and an append-only ledger is not one that
matters to the person whose face is in it. So `models/` is gitignored and a
clone builds its own from public sources, or runs with an explicit `-q` and no
index at all.

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

**Google Cloud Vision web detection** is the open-web arm, and the one worth
turning on. Bluesky's coverage is its real limit — a probe of someone who is not
on the platform finds nothing, which is honest but narrow — and this widens the
search to the whole indexed web. It activates when `GOOGLE_VISION_API_KEY` is
set. Unlike Lens it takes the image *bytes*, so a local file works with no
hosting step, and it returns the page each matching image sits on, which becomes
the citation in the evidence bundle.

It proposes; it does not decide. Google says "this image appears on these
pages", and every image it returns is downloaded and run through the same
encoder as everything else before anything is called a match.

**The two arms make different claims, the similarity scores show it, and the
code acts on it.** Measured on the committed probe, the Vision arm tops out at
**0.9952** — because reverse image search mostly returns *the same photograph*
republished elsewhere, so the face in it is trivially the same face. The
Bluesky arm scores **0.7596** on a *different* photograph, that account's own
avatar. The lower number is the stronger identity claim: same person, two
unrelated images. The higher one is closer to provenance — "this exact picture
also appears here".

Ranking by cosine alone therefore anchors the *weakest* evidence in the run and
presents it as the strongest, which is what the pipeline used to do — and the
effect is not subtle. Run with no query at all, the top of the table is
sixty-nine republications of the probe, and beneath them two Bluesky accounts:
a fan account whose avatar is a *crop* of the probe (0.9713) and her official
congressional account, whose avatar is a genuinely different photograph
(0.8662). Cosine picks the crop. So each candidate is classified before
anything is chosen:

- **Which photograph is it?** A 32×32 mean-centred greyscale fingerprint of the
  whole image, correlated against the probe's — a signal the face model has no
  part in, so it is not the model grading its own evidence. Over 170 real
  candidates, all 55 exact republications scored ≥ 0.95 and every Bluesky
  candidate scored below 0.67, including the true match at **0.0213**. Above
  **0.75** a candidate is the probe's own picture again — copy or crop — which
  is a *provenance* claim. Below it, a different photograph: an *identity*
  claim. The cutoff sits in the gap that 876 measurements across four runs and
  two probes left empty: of 695 Bluesky candidates not one reached 0.90 and
  694 sat below 0.70, the two that came close were both reposted crops of the
  probe (0.7915 and 0.8501), and the highest genuinely different photograph
  scored 0.6615. The number itself also goes into the bundle, so a reader can
  judge a borderline case rather than trust the label.
  See [`sigil/provenance.py`](sigil/provenance.py).
- **Where was it found?** Providers declare themselves `social` or `web`,
  because the deliverable is a social media post and an open-web page
  corroborates it rather than replacing it.

`pick_best` then prefers a social source, then an identity claim, and only then
the higher cosine — which is how `sigil run examples/probe-aoc.jpg` with no
query at all ends up anchoring `@ocasio-cortez.house.gov` rather than the
higher-scoring crop. The printed table stays ordered by raw similarity — hiding
that would be the dishonest part — every row carries its own label, and the
anchored row is marked. When everything that cleared the threshold is the
probe's picture again, the best of those is still anchored, because a
republication is a real finding; the bundle just records it as `provenance` so
it cannot be read as the other thing.

The evidence bundle carries `claim`, `probe_photo_similarity` and
`source_kind` alongside `discovered_via`, so a reader can re-derive the
decision instead of trusting it. That is what moved the schema to
`sigil/evidence/v2`.

**Google Lens via SerpAPI** is the older open-web arm, kept because it is a
different index. It activates only when `SERPAPI_KEY` is set *and* the probe is
passed as a public `https://` URL, because Lens matches on a URL rather than an
upload. With a local file it is skipped.

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
against a live Amoy node up to the point of funding, and re-checked against it
on 2026-09-03: `sigil chain address --chain rpc` connects, reports chain id
80002, derives the address and reads its balance without deploying anything,
and `sigil chain info` then refuses with

```
Error: 0x0b86bB… has no funds on chain 80002. Fund it from a testnet faucet
(`sigil chain address` shows the address and its balance), or use SIGIL_CHAIN=local.
```

— a message rather than a web3 traceback, exiting 1 so a script can act on it.
Fund a throwaway address from an Amoy faucet and it deploys for real. That is
the only step that needs a human.

### 5 · Calibration — what the threshold actually costs

Every claim this tool makes rests on one number: the cosine similarity above
which two faces are called the same person. 0.38 for ArcFace is the figure the
literature supports, but "the figure the literature supports" is not evidence,
and a reader cannot tell it apart from a number tuned until the demo worked.

`sigil calibrate` measures it, on your machine, against two real populations.

```bash
sigil calibrate --limit 200      # measure and save
sigil calibrate --show           # reprint the last measurement
```

**Impostor pairs come free.** Every pair of distinct Wikidata humans in the
identity index is a pair of different people, and 3,583 identities make
6,417,153 such pairs. The false-positive rate at any threshold is a count over
that entire set — no sampling, no extrapolation.

**Genuine pairs are harvested.** The index keeps one portrait per person, so it
has no same-person pairs. But different language Wikipedias illustrate the same
person with *different photographs*, which is a real second capture — different
year, angle and lighting — still labelled by Wikidata rather than by this code.
Fifteen Wikipedias yield about eleven distinct portraits per person.

Measured over 400 sampled identities on the insightface backend — 379 of them
born after 1900, yielding 1,275 usable portraits across 323 people:

| | pairs | mean | sd | median | max |
|---|---|---|---|---|---|
| same person | 2,069 | 0.6403 | 0.1650 | 0.6385 | 0.9997 |
| different people | 6,417,153 | 0.0048 | 0.0582 | 0.0035 | 1.0000 |

| at threshold 0.380 | |
|---|---|
| true positive rate | **94.44%** of same-person pairs caught |
| false positive rate | **2.836 × 10⁻⁵** — 1 in 35,259 |
| equal error rate | 0.56%, at threshold 0.17 |
| threshold for a 10⁻⁶ false-accept rate | 0.5872 |

The genuine sample was grown 6.5× (320 pairs → 2,069) to check the numbers were
not an artefact of a small sample. They moved by a quarter of a point: 94.69% →
94.44% true positives, 0.58% → 0.56% equal error. That stability is the reason
to quote them at all.

So 0.38 is well to the strict side of the equal-error point: it gives up recall
to buy a much lower false-accept rate, which is the right way round for a tool
that puts a name to a stranger.

Once measured, the number stops being a side report: every subsequent match
prints the error rate of the threshold it cleared.

```
  similarity           0.7596  (threshold 0.380)
  claim                identity — a different photograph of the same face
  measured error rate  2.836e-05 (1 in 35,259) false accepts at this threshold,
                       catching 94.7% of true pairs
```

It is read from `models/calibration.json` and shown, deliberately **not**
written into the evidence bundle. Putting it there would make the anchored hash
depend on whether the machine that produced it happened to have run `sigil
calibrate`, so two correct runs of the same match would disagree. The rate
belongs to the reader, not to the record. A calibration from a different
backend is ignored rather than reported, since two recognisers put similarity
on different scales.

**Three findings worth more than the headline numbers.**

*The impostor tail is bad data, not a bad model.* The closest "different
people" pairs are `G. D. Agrawal` vs `गुरुदास अग्रवाल` at 1.0000 — one man under
two Wikidata entities — and `Chhatrapati Shivaji Maharaj` vs `Soyarabai` at
1.0000, which is one painting used to illustrate two historical figures. Next
down are Mehmed II vs Suleiman the Magnificent at 0.6456: two Ottoman sultans
painted in one studio style. The report prints these pairs rather than
discarding them, because the real false-positive rate is *better* than the
measured one and only the named pairs show why.

*The genuine set needs an era filter, and the number proves it.* Without one,
the hardest same-person pairs were every combination of portraits of Alexander
the Great — a Roman bust against a Pompeian mosaic against a coin. Scoring a
face recogniser on those measures whether two sculptors agreed. Restricting
genuine pairs to people born after 1900 moved the true-positive rate from
74.18% to 94.69% and the equal error rate from 6.33% to 0.58% on the sample it
was first measured on. `--born-after 0` turns the filter off if you want to see
it for yourself.

*The hard cases are the ones you would predict, plus one that is not.* With the
filter on, the low end of the genuine distribution is people photographed
decades apart — Al Pacino at 0.2431, Andre Agassi at 0.2455 — which is age, not
error, and exactly what a threshold this strict chooses to miss. Below even
that, at −0.02 to 0.05, sit a handful of footballers whose lead images differ
across wikis by more than lighting; those are the group-photo mislabelling the
method warns about, left in because a hand-corrected set is not a measurement.

#### Naming a face is a different question, and a much harder one

Everything above is a *pair* rate: two faces, one comparison. `sigil identify`
does not do that. It compares one probe against **every** face in the index at
once, so it gets 3,583 chances to be wrong per question asked — and a rate of 1
in 35,000 per pair becomes percents per query.

Measured leave-one-out, which is exactly the dangerous case: every indexed face
queried against every *other* one, so the correct answer is absent and the only
question is whether the index names it anyway.

| threshold | wrong name per query | ignoring duplicate index entries |
|---|---|---|
| 0.38 *(the match threshold)* | 4.16% | 4.05% |
| 0.42 | 2.26% | 2.15% |
| **0.45** *(the identity threshold)* | **1.73%** | **1.62%** |
| 0.50 | 0.95% | 0.84% |
| 0.55 | 0.45% | 0.33% |
| 0.70 | 0.17% | 0.06% |

This vindicates the reasoning behind a separate, stricter bar for naming — and
it also puts a real number on what that bar buys, which is a 1-in-58 chance of
a confident wrong name rather than one in tens of thousands. Reusing 0.38 here
would have more than doubled it.

The worst offenders are the same index artefacts as before (`G. D. Agrawal` ↔
`गुरुदास अग्रवाल`, one man twice), and after those, genuine lookalikes: the
footballers `Aymeric Laporte` and `Iván Barton` name each other at 0.7496.

`sigil identify` now prints this rate under its results whenever a calibration
exists at the threshold in use — and refuses to quote it at any other
threshold, because the number moves steeply with the bar and a borrowed one
would be worse than none.

Two caveats the report repeats every run, because they cut in opposite
directions: some cross-language portraits are crops of one file rather than a
separate photograph, which flatters the true-positive rate (byte-identical
files are dropped, which catches the easiest case and not the rest); and the
subject of a lead image is taken to be its largest face, which is wrong for the
occasional group photo and depresses it. Neither is corrected by hand, because
a hand-corrected set is not a measurement.

---

## Verifying, and proving that verification bites

```bash
sigil verify --probe examples/probe-aoc.jpg --recheck-source
```

Six independent checks:

| check | what it proves |
|---|---|
| record on chain | the bundle's hash exists in the registry |
| similarity matches | the score was not edited after anchoring |
| subject commitment | the record refers to *this* face, as encoded on *this* provider — see below |
| probe re-encodes *(`--probe`)* | the supplied photo really produces the bundle's embedding |
| source image intact *(`--recheck-source`)* | the discovered post has not been edited or deleted since |
| claim re-derives *(both)* | the identity-vs-provenance verdict follows from the two images |

An optional check that was not requested reads as `None`, never as a pass.

The last one is worth separating from the rest. Every other check answers "has
this bundle been altered", which the hash already largely settles. That one
answers a question the hash cannot reach: **was the claim ever true?** A bundle
asserting `identity` — a different photograph of the same face — while pointing
at a republication of the probe is not tampered with, it is simply false, and
it would pass every other check. Given the probe and the live source image, the
verifier recomputes the fingerprint correlation from scratch and checks both
the number and the verdict it implies. The tool's own assertion is not evidence
for itself.

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
`./scripts/demo.sh` runs this whole sequence end to end for a screen recording,
and closes on `sigil calibrate --show` — the measured cost of the threshold
everything before it rested on.

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
| `sigil chain records` | 5 | list what the registry holds, read from chain state |
| `sigil chain address` | — | show the submitter address and balance, deploying nothing |
| `sigil calibrate` | — | measure what the threshold costs in false accepts and misses |
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
pytest -m "not network"   # 494 offline tests, 97% line coverage
pytest -m network         # 3 tests against the live API and a live chain
```

Both installs collect the same **494 offline tests**. Every insightface gate in
the suite is a runtime skip inside a test body rather than a collection-time
one, so the extra changes how many tests *run*, not how many exist. What the
two installs differ on is the skip count and the last coverage point:

| install | passed | skipped | coverage |
|---|---|---|---|
| Quickstart `.[insight,dev]` | 493 | 1 | **97%** |
| CI `.[dev]` | 486 | 8 | **96%** |

The single skip in the full install is the opencv parametrisation of a test
that only means anything for insightface; CI adds the seven that need the
backend itself. CI installs `.[dev]` alone because the 300 MB model pack is not
worth a minute on every push, and still enforces `--cov-fail-under=95`.

Both numbers are enforced rather than asserted: CI fails under its own coverage
floor, and the test count is checked against this README by the suite itself —
in **both** installs, and against every place the README states it. That check
used to skip without insightface, on the theory that collection varied by
install. It does not, so the one install that runs on every push was the one
install that never checked the number, and two figures in this file drifted to
467 and 402 underneath it.

CI runs the offline suite on 3.10, 3.11, 3.12 and 3.13. 3.10 is there because
it is the floor `pyproject.toml` declares, and a declared floor nobody runs is
a guess rather than a claim.

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
- **Identification can be confidently wrong, and now it is priced.** The index
  returns the nearest known face, and "nearest" is not "correct". Lookalikes
  and poor-quality portraits are the failure mode. Candidates below 0.45 cosine
  are rejected rather than reported, and the runner-up scores are always shown
  so the margin is visible — but the number that matters is **1.73%**:
  measured leave-one-out over the real index, that is how often a face the
  index does not contain gets a confident name anyway
  ([§5](#5--calibration--what-the-threshold-actually-costs)). One in fifty-eight.
  That is small enough for a demonstration and far too large to act on.
- **False positives are possible, at a measured rate.** A cosine threshold is a
  decision boundary, not proof of identity. At 0.380 the measured pair-level
  false-accept rate is 2.836 × 10⁻⁵ and the true-accept rate 94.44%
  ([§5](#5--calibration--what-the-threshold-actually-costs)); similarity is
  recorded on-chain precisely so a downstream reader can apply their own bar.
  Twins, close relatives, and heavily filtered photos are the known hard cases,
  and accuracy degrades with age gaps, pose, occlusion, and demographic
  distribution — published face-recognition benchmarks show materially
  different error rates across demographic groups, and nothing here corrects
  for that. **The calibration here does not correct for it either**: it is
  measured over whoever happens to be well-viewed on Wikipedia, which is not a
  balanced sample, so the aggregate rates above should not be read as holding
  evenly across groups.
- **A similarity is not exact to more digits than it deserves.** The same
  photograph, rescaled and recompressed the way republication does it, scores
  up to **0.0613** apart under the same encoder — measured over 161 real
  candidates at fingerprint correlations of 0.9999, where the two pictures are
  indistinguishable at 32×32. ArcFace is resolution-sensitive at that scale.
  It does not threaten the separation between the two populations, which is
  0.64 against 0.005, but it does mean a candidate within about 0.06 of the
  threshold is undecided rather than decided, whichever side of it the printed
  number happens to fall.

- **The chain proves integrity, not truth.** Anchoring establishes that a
  bundle existed at a time and has not changed. It says nothing about whether
  the match was *correct*. A confidently wrong match, anchored, is a permanent
  record of a confidently wrong match. `sigil verify` re-derives what it can —
  the probe's embedding from its pixels, the source image's bytes, and the
  identity-vs-provenance claim from the two images — but no amount of
  re-derivation makes a face model right.
- **Candidate *text* is chosen by a third party too, and both front ends
  render it.** In the terminal that matters more than it sounds, because the
  terminal output is what a screen recording shows as the evidence. `rich`
  reads `[...]` as styling, so an account whose display name was
  `[bold green]VERIFIED[/bold green]` printed itself in bold green — the same
  styling this tool uses for a passing check — and `[link=…]` becomes a
  clickable link in terminals that support it. An account able to style its own
  row can forge the appearance of a verdict. Every field a provider or Wikidata
  supplies is escaped before `rich` sees it.

  The web UI had the sharper version of the same hole. Those fields went
  straight into `innerHTML`, so a display name of `<img src=x onerror=…>`
  *executed* — in a page served from localhost, with `/api/run`,
  `/api/tamper` and `/api/evidence` one `fetch` away — and a `post_url` of
  `javascript:…` was the same thing with an extra click. Everything untrusted
  is HTML-escaped now, anything becoming an `href` goes through an
  `http(s)`-only allowlist, and a rejected URL becomes `#` rather than passing
  through. Both the hole and the fix were confirmed in a real browser rather
  than reasoned about; a lint-style test guards the shape, since pytest cannot
  run the page.

- **The weights that decide every verdict are pinned by hash.** The ONNX models
  are downloaded on first use, and they determine every embedding — so every
  match, every similarity and every hash anchored on chain. A substituted model
  changes every answer this tool gives while leaving the pipeline, the tests
  and the chain records all looking correct. `scripts/fetch_models.sh`
  previously accepted any file over 10 KB; it now verifies a pinned SHA256,
  deletes a file that fails and exits non-zero. Demonstrated with a 19,500-byte
  substitute, which the old size check would have accepted. The insightface
  pack is fetched by that library rather than by this repo, so its integrity is
  theirs; the opencv path is the one a clean clone actually uses.

- **A candidate URL is chosen by a third party, so it is checked before it is
  fetched.** Every image this tool downloads is at an address supplied by
  someone else — an AppView response, whatever Google's web index returned — and
  "download every candidate" would otherwise mean "issue a GET wherever a search
  result points". That includes `http://127.0.0.1:8099/api/evidence`, which is
  this tool's own web UI, and a cloud metadata endpoint. Non-`http(s)` schemes
  are refused, and so is any host resolving to a loopback, private, link-local
  or otherwise non-global address — before the socket is opened, and again on
  the post-redirect URL, since an `https` CDN link that 302s to loopback would
  walk past the first check. The reason is not only defensive: the bundle
  asserts the match is a *publicly posted* image, and an address on a private
  range cannot be one, so such a candidate is wrong on the bundle's own terms.
  Known limit, stated rather than papered over — the name is resolved by the
  check and again by the request, so a host answering differently between the
  two defeats it; closing that needs the connection pinned to the address that
  was checked.

- **No biometrics on chain — by design.** The registry stores a salted
  commitment to the probe, never an embedding or an image. Publishing a face
  vector to an append-only public ledger would be an irreversible biometric
  disclosure with no deletion path, which is squarely at odds with the erasure
  rights the regulations above grant.
- **The local chain is a real EVM, not a real network.** py-evm executes the
  actual bytecode and persists state, but it is single-party: it proves
  integrity against your own record, not against a public consensus. Use
  `SIGIL_CHAIN=rpc` for a record a third party can independently check. It is
  also not a datastore: the snapshot is rewritten whole on every anchor, at
  about 9 KB per record beyond the first (measured, with anchor time flat at
  ~23 ms out to 40 records), so it is sized for demonstrations rather than for
  thousands of them.
- **Bluesky's coverage is the practical ceiling on the keyless path.** It is a
  small, mostly Western platform, so a probe of someone without an account
  there finds nothing however good the face model is. That is coverage, not
  accuracy, and it is what the Cloud Vision arm exists to widen. Measured on
  three probes: a private individual (no account, no match), a public figure
  with a parked handle and no posts (named correctly from the index, still no
  match), and one with an active account (named, matched, anchored).
  Re-checked on 2026-09-03 from the keyless clone: the committed control image
  against a nonsense query examined 179 images, topped out at 0.2992 against
  the 0.363 opencv threshold, anchored nothing, and exited 2 — which is the
  documented code for "nothing cleared", not for an error.

- **Search breadth is bounded by cost, not capability.** `SIGIL_MAX_IMAGES`
  defaults to 200. Raising it widens recall and lengthens runtime roughly
  linearly. *What* it lengthens depends on the encoder. On CPU the run is
  essentially all inference — **97% of it is the detector**, against about 3%
  waiting on the network — and a default run with both arms enabled takes
  ~50 s for 169 images. On GPU the detector is 12× faster and the balance
  inverts: the same run is network-bound and takes ~16–23 s over four
  measurements — and being network-bound means it is now *variable* rather than
  predictable. `scripts/demo.sh` has been observed between 32 s and 104 s on
  identical code, the spread being entirely third-party latency. Its floor is
  about 24 s regardless: the script runs six commands that each load the model
  in their own process, at ~3.9 s apiece, and that cost is the same on CPU and
  CUDA because it is reading a file rather than warming a device. The download
  worker count follows the encoder for exactly that reason (8 on CPU, 16 on
  CUDA, `SIGIL_DOWNLOAD_WORKERS` to override), which halved the search stage
  on GPU and does nothing at all on CPU.

  That CPU ratio is also why three plausible optimisations were measured and
  rejected rather than adopted — a cheap pre-filter in front of the detector
  (1.24×, and it costs recall), a fingerprint-keyed score cache (1.5×, and it
  is unsound), and fetching smaller portraits for the index build (no faster at
  all). Each is written up where someone would go looking to try it.
- **SerpAPI needs a hosted probe.** Google Lens matches on a URL, so the
  open-web arm cannot run against a local file.

## Licence

MIT — see `LICENSE`. Example images are public domain or CC BY-SA; see
`examples/README.md` for per-image attribution.
