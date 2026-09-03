"""Is this candidate a different photograph, or the probe's own photograph again?

The two search arms return evidence of different kinds, and the README already
says so: a reverse-image hit mostly finds *the same picture* republished, so the
face in it is trivially the same face, while a social account's own avatar is a
*different* capture of that person. Same-person-two-photographs is an identity
claim; same-photograph-elsewhere is a provenance claim. They are not
interchangeable, and the higher cosine belongs to the weaker one.

Deciding which is which from the face score alone would be circular - it would
use the model under test to grade its own evidence. So this compares the whole
image instead, with a signal the face model has no part in: a 32x32 greyscale
fingerprint, mean-centred and L2-normalised, correlated against the probe's.
That survives rescaling and recompression, which is what republication does to
a picture, and it collapses for a photograph taken at a different moment.

Measured over 170 real candidates for the committed probe:

    photo similarity   candidates
        >= 0.95        55, every one a Vision hit - the probe's own photograph
                       republished on someone else's page
        0.56 - 0.93    11, all Vision - crops, merchandise prints and Commons
                       variants of that same photograph
        <  0.67        every Bluesky candidate, the true avatar match included

The exact copies sit well clear of everything else, so the cutoff is placed at
the top of that gap rather than in the middle of the crop band.

What this signal does **not** separate is a crop from an independent
photograph, and that limit is real rather than theoretical: a Bluesky account
reposting a cropped version of the probe scored 0.7915, while a *different
person* whose portrait happens to be framed alike scored 0.6615. Two points
that close together cannot be split without fitting the cutoff to them, which
is how a measured threshold turns back into a guessed one.

So the binary label stays deliberately conservative - a crop is reported as an
identity claim rather than demoted on a hunch - and the *number* is written
into the evidence bundle and printed in the match panel next to it. A reader
seeing 0.7915 can draw their own conclusion; a reader seeing only a label
cannot.
"""

from __future__ import annotations

import cv2
import numpy as np

# At or above this, a candidate is the probe's own photograph again. Set above
# every Bluesky value measured (the highest was 0.6615, and that one a
# different person whose portrait happens to be framed alike) and below the
# exact-copy cluster.
PROVENANCE_CUTOFF = 0.90

FINGERPRINT_SIZE = 32

IDENTITY = "identity"
PROVENANCE = "provenance"


def fingerprint(image_bgr: np.ndarray, size: int = FINGERPRINT_SIZE) -> np.ndarray:
    """A small scale- and brightness-invariant descriptor of a whole image.

    Mean-centred and L2-normalised, so a dot product between two of these is a
    correlation: 1.0 for the same composition at any size or exposure, around
    0 for unrelated pictures.
    """
    grey = (cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            if image_bgr.ndim == 3 else image_bgr)
    small = cv2.resize(grey, (size, size),
                       interpolation=cv2.INTER_AREA).astype(np.float32)
    small -= small.mean()
    norm = float(np.linalg.norm(small))
    # A flat image - a solid colour block, or a blank placeholder - has no
    # structure to correlate. It is reported as unrelated to everything rather
    # than dividing by zero.
    return small.ravel() / norm if norm > 0 else small.ravel()


def photo_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation between two fingerprints, clipped to [-1, 1]."""
    return float(np.clip(float(np.dot(a, b)), -1.0, 1.0))


def claim_for(photo_sim: float, cutoff: float = PROVENANCE_CUTOFF) -> str:
    """Which kind of claim a candidate supports, given how close its picture is."""
    return PROVENANCE if photo_sim >= cutoff else IDENTITY
