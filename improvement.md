# improvement.md — what the other Task 3 submissions built, and what sigil is missing

**Scope.** GitHub search on 2026-09-07 for HH Goa 2026 Task 3 submissions
(`hh goa task 3`, `hhgoa task3`, `face identification blockchain verification`,
and variants), filtered to repos created on or after 2026-08-31 — the task
launch date. **62 repos** matched. Their READMEs and full file trees were
pulled and scanned; the substantial ones were read.

> **Read this first — the deadline is tonight (2026-09-07 23:59) and there are
> no resubmissions.** Almost nothing in this file should be attempted before
> then. What is still owed is the **screen recording** and the **form**
> (https://forms.gle/oZbQGuwiNeHVcHWo8); `./scripts/demo.sh` produces the exact
> sequence. The one exception worth considering under time pressure is
> [G1 (public testnet deployment)](#g1--a-public-testnet-anchor-anyone-can-click)
> — it is a faucet request and one command, and it is the single most visible
> thing the leading repos have that sigil does not. Everything below G1 is
> post-deadline work.

---

## 1. The field, in one table

Feature frequency across the 62 repos (keyword scan of README + file tree, so
these are approximate — a repo that *names* a thing counts):

| what | repos | sigil |
|---|---:|---|
| SerpAPI / Google Lens reverse image search | 40 | ✅ `search/serpapi.py` |
| Solidity contract of their own | 35 | ✅ `contracts/SigilRegistry.sol` |
| Test suite | 32 | ✅ 506 tests, 95%+ enforced |
| **Deployed to a public testnet (Sepolia / Amoy)** | **29** | ❌ **gap — G1** |
| REST API / HTTP endpoints | 29 | ✅ `sigil serve` |
| **Instagram / X / Reddit / LinkedIn as search targets** | **~22** | ❌ **gap — G3** |
| dlib `face_recognition` backend | 20 | ✅ different stack (ArcFace/SFace) |
| Polygon Amoy support | 20 | ✅ `SIGIL_CHAIN=rpc` |
| Consent / ethics section | 18 | ✅ prose only — ❌ **enforced gate, G2** |
| Local simulated chain | 18 | ✅ py-evm (a *real* EVM, stronger) |
| Hardhat / Ganache tooling | 23 | ➖ deliberately none (no Node dep) |
| **Liveness / anti-spoof check** | **8** | ❌ **gap — G5** |
| **Merkle tree / inclusion proofs** | **7** | ❌ **gap — G7** |
| **IPFS / Pinata pinning + CID on chain** | **6** | ❌ **gap — G6** |
| **Docker / compose** | **6** | ❌ **gap — G9** |
| Batch / multi-face processing | 6 | ⚠️ warns, picks largest — G8 |
| **Deepfake detection** | **5** | ❌ **gap — G5** |
| **Webcam / live capture** | **4** | ❌ **gap — G4** |
| **EXIF / metadata extraction** | **3** | ❌ **gap — G10** |
| Bing Visual Search | 12 | ❌ gap — G3 |
| Yandex reverse image | 5 | ❌ gap — G3 |
| Wikidata/Wikipedia identity | 6 | ✅ 3,583-face index (deeper than any) |
| Bluesky / AT Protocol | 2 | ✅ (only `kushal-…` and `nirmaljosephukken` also) |
| CI (GitHub Actions) | 2 | ✅ (only `ShrujanSunkari` and `sillanaresh` also) |
| Threshold **calibration** with measured FPR/TPR | 2 | ✅ far beyond both |

### Where sigil is already ahead

Worth being clear about, so nothing below is copied at the cost of something
better:

- **Nobody else measured their threshold.** `Ojas37` has a `--calibrate` flag
  and `aashna-boop` mentions calibration; neither publishes a false-accept
  rate. Sigil's 6.4 M impostor pairs, TPR 94.44% / FPR 2.8e-5, and the
  **1.73% wrong-name rate** are unique in the field.
- **Nobody else separates *this photo again* from *another photo of that
  face*.** `ankitvirla` computes a pHash but uses it as a scoring bonus, not as
  a provenance discriminator. Sigil's 32×32 fingerprint cutoff at 0.75 —
  measured over 876 candidates — is the one thing stopping a reverse-image hit
  on the probe's own republication being anchored as evidence.
- **Most "blockchains" here are a JSON file with a `previous_hash` field.**
  18 repos ship a hand-rolled PoW ledger. Sigil runs an actual EVM in-process
  and speaks the same contract to a real node.
- **Verification depth.** Six checks including re-deriving the identity claim
  from pixels. `JinayPatel15` and `Ojas37` re-check the hash only.
- **Adult/leak-site refusal before download or citation** (`search/safety.py`).
  No other repo has anything of the kind, and every one of them anchors a
  `post_url` chosen by a third party.

---

## 2. The gaps, in priority order

### G1 — A public testnet anchor anyone can click

**29 of 62 repos have this. Sigil does not.** It is the most visible
difference.

- `ShrujanSunkari/hh_goa_task3` puts contract `0xCc67296B…`, deployment block
  11621606, proof tx `0xb34f91e1…`, block 11621611, and a timestamp in the
  README, each linked to sepolia.etherscan.io.
- `pk1427/VeriTrace` has a Polygon Amoy deployment table with
  `0x66506309…` linked to amoy.polygonscan.com.
- `alokmuskan` prints local hash and on-chain hash side by side and says
  "verifyRecord(...) returned true".

Sigil supports exactly this path (`SIGIL_CHAIN=rpc`, chain id 80002 verified
live) and has never been funded. `sigil chain address --chain rpc` prints the
address to fund; Amoy faucet → deploy → one `anchor` → put the address, tx
hash, block and explorer link in the README next to the existing gas figures.

**Effort: ~20 minutes, mostly waiting on a faucet. Do this one.**

### G2 — An *enforced* consent gate, not an ethics paragraph

`pk1427/VeriTrace` is the only repo whose ethics claim is executable, and it is
the strongest single idea in the field:

> Phase 2 — enroll an owner-identified subject; the pipeline **refuses to
> search** unless the probe face matches an enrolled subject at ≥ 0.60 cosine.
> Re-enrolling the same `subject_id` is refused (no silent overwrite).

Sigil's Limitations section says "use this on yourself, on a consenting
subject, or on a public figure" — and then does nothing to make that true.
A `sigil enroll <subject_id> <image>` command plus a `--require-consent` mode
(default on, `SIGIL_ALLOW_UNCONSENTED=1` to bypass with a printed warning)
would turn the honest prose into an honest control. It also fits sigil's
existing style: the gate is a measured cosine with a stated threshold, and the
registry stores embeddings, never images.

`Ojas37` has the complementary half — embeddings held in RAM only, never
persisted, never on chain — which sigil should state explicitly since it is
already true.

### G3 — More than one social platform

Sigil searches Bluesky and the open web. 22 repos name Instagram, X/Twitter,
Reddit or LinkedIn; 12 add Bing Visual Search, 5 Yandex, 1 TinEye.

Most of those are shallow (a domain-name check on Lens results, not a real
platform search). Two are not:

- `kushal-naga-sai-balaji` searches **Bluesky and Mastodon** — the two
  platforms with genuinely open, keyless APIs. Mastodon is the obvious
  addition for sigil: `/api/v2/search` and `/api/v1/accounts/:id/statuses` are
  anonymous on most instances, avatars are public, and it drops straight into
  the existing `Provider` interface next to `bluesky.py` with the same
  `social` source-kind.
- `ShampitaBhattacharjee` and `ShrujanSunkari` implement Bing and Yandex as
  automatic fallbacks when Lens returns zero. Sigil's Vision arm covers the
  same ground without hosting the probe, so this is lower value — but a
  **Yandex** arm is worth noting because it is the one engine that indexes
  faces rather than pages.

Ranked: Mastodon provider (real win, ~100 lines, keyless) > Bing fallback >
scraping Instagram/LinkedIn (rate-limited, ToS-hostile, mostly theatre).

### G4 — Webcam capture: "face **scan**" taken literally

`alokmuskan`, `rohantoraskar1110-del`, `piyushgoilkar17` and `trpzz5` capture
from a camera. The brief says "face scan input", and on a screen recording
a live capture reads as a scan in a way that `run examples/probe-aoc.jpg`
does not.

`sigil serve` already has an upload box; adding `getUserMedia()` + a capture
button posting the frame to the same endpoint is a small, self-contained web-UI
change with no new Python dependency.

### G5 — Liveness / anti-spoof and deepfake checks

8 repos claim liveness, 5 deepfake detection. Nearly all of it is one
heuristic — `Arbab1308` is representative: *"we analyse texture variance (blur
density) to ensure liveness"*. That is a Laplacian variance threshold, and it
is trivially defeated by a sharp printed photo.

The honest version, and the one that fits this repo, is not to claim liveness
detection but to **record the signal and its limits**: a blur/texture score in
the evidence bundle with a stated, measured separation (or a stated *lack* of
separation) between screen re-captures and live captures. Given sigil's habit
of pricing every claim, shipping an unmeasured spoof check would be a
regression, not a feature. Pairs naturally with G4.

### G6 — IPFS: pin the evidence, anchor the CID

6 repos (`ADITHYARAMAN7`, `Arbab1308`, `Geetanjali-147`, `ShrujanSunkari`,
`Somnath-builder`, `HariPad2005`) pin the payload to IPFS and put the CID
on-chain. `Arbab1308` states the reason well: on-chain storage is expensive, so
store the CID and keep the bytes in content-addressed storage.

The real argument for sigil is different and better: today the anchored hash
proves an evidence bundle existed, but **if the bundle file is lost the record
is unverifiable** — the truncated-bundle case is already flagged as a permanent
orphan record. A CID gives the bundle an address that anyone can resolve, and
the CID *is* the hash, so it costs nothing extra on chain. Opt-in (`--pin`),
because pinning publishes a `post_url` and a face similarity to a public DHT.

### G7 — Merkle batching and inclusion proofs

7 repos ship Merkle trees; `Solragna` and `kushal-naga-sai-balaji` implement
real inclusion proofs (`MerkleTree.get_inclusion_proof`, binary Merkle DAG with
audit-path walk).

Sigil writes one `SSTORE` per record (114,222 gas cold / 97,122 warm). A
Merkle root over N bundles is one transaction for all N, with a logarithmic
proof per bundle — the standard win, and it turns `sigil verify` into a proof
check that does not need the chain to hold every record. Worth it only if
sigil ever anchors more than a handful of records; note it as designed-for, not
built.

### G8 — Multi-face and batch modes

6 repos process every detected face, or a directory of images.

Sigil detects all faces, warns when there is more than one, and encodes the
largest (`sigil/face/encoder.py:73`). `Ojas37` does the same but additionally
saves `selected_target_face.jpg` so a reviewer can see **which person was
searched** — a genuinely good idea that costs almost nothing and belongs in the
evidence bundle, not just on disk.

Full batch mode (`sigil run *.jpg`) is a straightforward loop over
`run_pipeline` and pairs with G7.

### G9 — Docker

6 repos ship a Dockerfile or compose file; `ShrujanSunkari` has a "Quickstart
with Docker" section.

Sigil's install story is already strong (one `pip install -e`, no Node, no
keys, `requirements.lock` as the escape hatch), so this is convenience rather
than capability. It would however pin the ONNX/numpy/Python matrix that has
caused repeated CI trouble.

### G10 — EXIF and candidate metadata in the evidence

`Ojas37`, `Priyank911` and `Arbab1308` extract EXIF from the probe and page
metadata from candidates.

For sigil the interesting field is not the camera model — it is that EXIF
timestamps and GPS are **personal data that would be anchored permanently**.
The right move is the inverse of what they did: strip and explicitly record
that EXIF was stripped, which is a one-line change and a genuinely better
answer to the same prompt.

### G11 — Small ergonomics worth copying

- **`--offline-mock` / `--dry-run`** (`ShrujanSunkari`, `Ojas37`): run the
  whole pipeline with no keys and no network so a judge can execute it on a
  plane. Sigil's keyless path still needs the network. Cheap, and it makes the
  test suite's mocks reachable from the CLI.
- **`--json` output on `run`** (`ShrujanSunkari`): machine-readable result to
  stdout. Sigil writes `artifacts/evidence.json` but has no `--json` flag.
- **On-chain duplicate rejection** (`JinayPatel15`, `ShrujanSunkari`): the
  contract reverts on a hash already registered. Sigil's registry is
  append-only and silently accepts a re-anchor.
- **Signed evidence** (`Solragna` Ed25519, `kushal-…` ECDSA secp256k1): binds
  *who* produced the claim, not only that it did not change. Sigil has
  `msg.sender` and nothing else; an EIP-712 signature over the canonical bundle
  would be the natural fit next to `canonical.py`.
- **Result caching with a TTL keyed on the probe SHA-256** (`alokmuskan`;
  15 repos have some cache): sigil deliberately re-searches every run and
  should keep that default — but a cache would make repeated demo runs
  instant. Note that a *fingerprint-keyed score* cache is unsound (0.0613
  verdict drift under rescaling); a **probe-bytes-keyed search-result** cache
  is not the same thing and is safe.
- **A recording checklist in the repo** (`rohantoraskar1110-del` ships
  "Screen Recording Script & Presentation Checklist"). `scripts/demo.sh`
  already *is* this; saying so in the README costs one line.

---

## 3. Repos worth reading in full

| repo | why |
|---|---|
| [`pk1427/VeriTrace`](https://github.com/pk1427/VeriTrace) | the enforced consent gate (G2) — the best single idea in the field; deployed on Amoy |
| [`ShrujanSunkari/hh_goa_task3`](https://github.com/ShrujanSunkari/hh_goa_task3) | most complete: live Sepolia proof, Docker, CI, multi-engine fallbacks, `--offline-mock`, honest stack rationale |
| [`Ojas37/face-identification-blockchain-verification`](https://github.com/Ojas37/face-identification-blockchain-verification) | closest in spirit to sigil: RFC 8785 JCS canonicalisation, embeddings in RAM only, a genuinely good Known Limitations section |
| [`alokmuskan/…`](https://github.com/alokmuskan/Face-Identification-Blockchain-Verification) | best documented local-vs-on-chain data table; explicit "what never leaves the machine" |
| [`kushal-naga-sai-balaji/…`](https://github.com/kushal-naga-sai-balaji/Face-Identification-Blockchain-Verification) | Merkle DAG + ECDSA provenance; Bluesky **and** Mastodon |
| [`JinayPatel15/…`](https://github.com/JinayPatel15/Face-Identification-Blockchain-Verification) | strict 0.90 gate, halts and anchors nothing on no-match — same discipline as sigil |
| [`mangal999/…`](https://github.com/mangal999/face-identification-blockchain-verification) | dual anchoring (local + Ethereum simultaneously), four pluggable face backends |
| [`Arbab1308/HHGOA_TASK3`](https://github.com/Arbab1308/HHGOA_TASK3) | IPFS CID → chain, cross-platform "person timeline" clustering |

## 4. What not to copy

- **A hand-rolled PoW JSON ledger.** 18 repos have one. `mangal999` mines to
  difficulty 2. It is a hash chain in a file, and sigil's in-process EVM is
  strictly more than it.
- **Pseudo-faces.** `mangal999` synthesises a "central crop pseudo-face" when
  detection finds nothing, "so the pipeline always has at least one face to
  anchor". That anchors a record about a face that was never detected. Sigil
  exiting 2 with nothing anchored is the correct behaviour and should stay.
- **Unmeasured liveness claims** — see G5.
- **Instagram/LinkedIn scraping.** Rate-limited, ToS-hostile, and in most of
  these repos it is a domain-name substring check on Lens output rather than a
  search.
