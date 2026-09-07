#!/usr/bin/env python3
"""Measure the face encoder on LFW, under both subject-selection policies.

LFW is the standard yardstick for face verification: 6,000 pairs, half of them
the same person, split into ten folds. The protocol matters as much as the
dataset - the decision threshold is fitted on nine folds and tested on the
tenth, so no accuracy here is measured at a threshold chosen on the data it is
scoring. That is what makes the number comparable to a published one.

It is run under both `--subject` policies over *identical* embeddings, because
the first time this was measured the shipped `largest` policy scored 98.46% and
the encoder looked a point and a half short of its published accuracy. It was
not the encoder: 17% of LFW images have a bystander in them, and the largest
face is not always the subject. Selecting the centre face instead - LFW's own
convention, and reasonable given the frames are funneled to centre the subject
- puts the same embeddings at 99.85%. Everything the pipeline rests on is the
same in both columns, so the gap is the picking rule and nothing else.

Usage:
    scripts/fetch_lfw.sh                # ~230 MB, once
    .venv/bin/python scripts/bench_lfw.py

The embeddings are cached, so a re-run costs seconds rather than minutes.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from sigil.face import SUBJECT_POLICIES, cosine, load_encoder, select_subject

DEFAULT_ROOT = Path.home() / "scikit_learn_data" / "lfw_home"


# --------------------------------------------------------------------- dataset

def read_pairs(root: Path) -> list[tuple[Path, Path, bool, int]]:
    """The official pairs.txt: ten folds of 300 matched then 300 mismatched."""
    images = root / "lfw_funneled"
    path = root / "pairs.txt"
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Fetch the dataset first:  scripts/fetch_lfw.sh"
        )

    def img(name: str, n: str) -> Path:
        return images / name / f"{name}_{int(n):04d}.jpg"

    lines = path.read_text().splitlines()
    n_folds, per = (int(x) for x in lines[0].split())
    pairs, i = [], 1
    for fold in range(n_folds):
        for _ in range(per):
            name, a, b = lines[i].split("\t")
            i += 1
            pairs.append((img(name, a), img(name, b), True, fold))
        for _ in range(per):
            na, a, nb, b = lines[i].split("\t")
            i += 1
            pairs.append((img(na, a), img(nb, b), False, fold))
    return pairs


# -------------------------------------------------------------------- encoding

def encode_every_face(paths: list[Path], encoder, cache: Path) -> dict:
    """{path: (embeddings, bboxes, shape)} - every face found, not just one.

    Every face, so that the selection policies below can be compared over one
    set of embeddings rather than two encoding passes that might differ for
    some other reason.
    """
    if cache.exists():
        stored = np.load(cache, allow_pickle=True)["data"].item()
        if all(str(p) in stored for p in paths):
            print(f"using cached embeddings from {cache}")
            return stored
    else:
        stored = {}

    todo = [p for p in paths if str(p) not in stored]
    print(f"encoding {len(todo)} images ({len(stored)} cached)")
    started = time.time()
    for i, p in enumerate(todo, 1):
        image = cv2.imread(str(p))
        faces = encoder.detect_and_encode(image) if image is not None else []
        stored[str(p)] = (
            np.array([f.embedding for f in faces], dtype=np.float32)
            if faces else np.zeros((0, 1), dtype=np.float32),
            np.array([f.bbox for f in faces], dtype=np.float32).reshape(len(faces), 4),
            tuple(image.shape[:2]) if image is not None else (0, 0),
        )
        if i % 1000 == 0 or i == len(todo):
            rate = i / (time.time() - started)
            print(f"  {i}/{len(todo)}  {rate:.0f} img/s  eta {(len(todo) - i) / rate:.0f}s",
                  flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, data=np.array(stored, dtype=object))
    return stored


class _Detected:
    """The two attributes select_subject reads, so the real policy is exercised."""

    __slots__ = ("embedding", "bbox")

    def __init__(self, embedding, bbox):
        self.embedding, self.bbox = embedding, tuple(float(v) for v in bbox)


def pick(entry, policy: str):
    embeddings, boxes, shape = entry
    if len(embeddings) == 0:
        return None
    faces = [_Detected(e, b) for e, b in zip(embeddings, boxes, strict=True)]
    chosen = select_subject(faces, policy, shape)
    return None if chosen is None else chosen.embedding


# --------------------------------------------------------------------- scoring

def best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    candidates = np.unique(np.round(scores, 4))
    return float(max((float(np.mean((scores >= t) == labels)), t) for t in candidates)[1])


def evaluate(pairs, store, policy: str, operating_threshold: float) -> dict:
    scores, labels, folds, dropped = [], [], [], 0
    for a, b, same, fold in pairs:
        ea, eb = pick(store[str(a)], policy), pick(store[str(b)], policy)
        if ea is None or eb is None:
            dropped += 1
            continue
        scores.append(cosine(ea, eb))
        labels.append(int(same))
        folds.append(fold)
    scores, labels, folds = np.array(scores), np.array(labels), np.array(folds)

    # The protocol: fit the threshold on nine folds, score the tenth.
    accuracies, thresholds = [], []
    for held_out in range(10):
        train, test = folds != held_out, folds == held_out
        t = best_threshold(scores[train], labels[train])
        thresholds.append(t)
        accuracies.append(float(np.mean((scores[test] >= t) == labels[test])))
    accuracies = np.array(accuracies)

    order = np.argsort(-scores)
    ranked = labels[order]
    tpr = np.cumsum(ranked) / ranked.sum()
    fpr = np.cumsum(1 - ranked) / (1 - ranked).sum()

    accepted = scores >= operating_threshold
    genuine, impostor = scores[labels == 1], scores[labels == 0]
    return {
        "accuracy": accuracies.mean() * 100,
        "sd": accuracies.std() * 100,
        "folds": accuracies * 100,
        "threshold_range": (min(thresholds), max(thresholds)),
        "auc": float(np.trapezoid(tpr, fpr)),
        "tpr_at_1e2": float(tpr[min(np.searchsorted(fpr, 1e-2), len(tpr) - 1)]) * 100,
        "tpr_at_1e3": float(tpr[min(np.searchsorted(fpr, 1e-3), len(tpr) - 1)]) * 100,
        "shipped_accuracy": float(np.mean(accepted == labels)) * 100,
        "shipped_tpr": float((accepted & (labels == 1)).sum() / (labels == 1).sum()) * 100,
        "false_accepts": int((accepted & (labels == 0)).sum()),
        "impostor_pairs": int((labels == 0).sum()),
        "scored": len(scores),
        "dropped": dropped,
        "genuine_min": float(genuine.min()),
        "genuine_mean": float(genuine.mean()),
        "impostor_max": float(impostor.max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help="Directory holding pairs.txt and lfw_funneled/.")
    ap.add_argument("--cache", type=Path, default=None,
                    help="Where to keep the embeddings (default: <root>/sigil-embeddings.npz).")
    ap.add_argument("--backend", default="auto", help="Face backend to measure.")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Operating threshold to report against (default: the backend's).")
    args = ap.parse_args()

    from sigil.config import DEFAULT_THRESHOLDS

    encoder = load_encoder(args.backend)
    operating = args.threshold
    if operating is None:
        operating = DEFAULT_THRESHOLDS.get(encoder.name, 0.38)

    pairs = read_pairs(args.root)
    unique = sorted({p for a, b, _, _ in pairs for p in (a, b)})
    missing = [p for p in unique if not p.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} images missing, e.g. {missing[0]}")

    cache = args.cache or (args.root / "sigil-embeddings.npz")
    print(f"backend: {encoder.name} ({encoder.model})"
          f" on {getattr(encoder, 'provider', '') or 'cpu'}")
    store = encode_every_face(unique, encoder, cache)

    counts = np.array([len(store[str(p)][0]) for p in unique])
    print(f"\n{len(unique)} images · no face {int((counts == 0).sum())}"
          f" · one face {int((counts == 1).sum())}"
          f" · more than one {int((counts > 1).sum())} (up to {int(counts.max())})")
    print(f"failure to enrol: {(counts == 0).mean() * 100:.2f}%")

    results = {p: evaluate(pairs, store, p, operating) for p in SUBJECT_POLICIES}

    width = 24
    print("\n" + "=" * (26 + width * len(SUBJECT_POLICIES)))
    print(f"LFW · 6,000 pairs · 10 folds · funneled · operating threshold {operating}")
    print("=" * (26 + width * len(SUBJECT_POLICIES)))
    header = f"{'':26}" + "".join(f"{p:>{width}}" for p in SUBJECT_POLICIES)
    print(header)

    def row(label, key, fmt="{:.2f}"):
        cells = "".join(f"{fmt.format(results[p][key]):>{width}}" for p in SUBJECT_POLICIES)
        print(f"{label:26}{cells}")

    row("10-fold accuracy %", "accuracy")
    row("  ± sd", "sd")
    row("ROC AUC", "auc", "{:.5f}")
    row("TPR @ FPR 1e-2 %", "tpr_at_1e2")
    row("TPR @ FPR 1e-3 %", "tpr_at_1e3")
    print()
    row("accuracy at threshold %", "shipped_accuracy")
    row("TPR at threshold %", "shipped_tpr")
    row("impostors accepted", "false_accepts", "{:.0f}")
    row("of impostor pairs", "impostor_pairs", "{:.0f}")
    print()
    row("pairs scored", "scored", "{:.0f}")
    row("pairs dropped, no face", "dropped", "{:.0f}")
    row("worst genuine pair", "genuine_min", "{:.4f}")
    row("mean genuine pair", "genuine_mean", "{:.4f}")
    row("best impostor pair", "impostor_max", "{:.4f}")

    for policy in SUBJECT_POLICIES:
        r = results[policy]
        print(f"\n{policy}: per fold " + ", ".join(f"{a:.2f}" for a in r["folds"]))
        print(f"{' ' * len(policy)}  thresholds "
              f"{r['threshold_range'][0]:.3f} – {r['threshold_range'][1]:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
