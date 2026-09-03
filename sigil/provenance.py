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

Measured over 876 real candidates across four runs and two probes - 695 from
Bluesky, 181 from the open web:

    photo similarity    bluesky   open web
        >= 0.95               0        140   the probe's own file, republished
        0.90 - 0.95           0          2
        0.85 - 0.90           1          2   crops and merchandise prints
        0.70 - 0.85           0          4
        <  0.70             694         33   a different photograph

Not one Bluesky candidate in 695 reaches 0.90, and 694 of them sit below 0.70.
The two Bluesky candidates that got anywhere near the top are both crops of the
probe: an account reposting it at 0.7915 and another at 0.8501. The highest
*genuinely different* photograph on that platform scored 0.6615 - and it is a
different person whose portrait happens to be framed alike.

The cutoff is set at 0.75, in the gap that separates those two populations, and
it is set from that table rather than from taste. An earlier version used 0.90,
which was chosen when the only evidence was one probe: it sat above every crop
as well as every copy, so a reposted crop was labelled an independent
photograph. That overstates the evidence, which is the error worth avoiding
here - the alternative error, demoting a real photograph, has 694 candidates of
headroom beneath it. The true match on the committed probe scores 0.0213.

What this signal still cannot do is separate a crop from a different
photograph in the region between them, because on this data there is nothing
there to separate. Should something land in it, the *number* is written into
the evidence bundle and printed beside the label, so a reader seeing 0.78 can
draw their own conclusion where a reader seeing only a label could not.
"""

from __future__ import annotations

import cv2
import numpy as np

# At or above this, a candidate is the probe's own photograph again - a copy or
# a crop. Placed in the empty band between the highest genuinely different
# photograph measured (0.6615) and the lowest crop (0.7915). See the table
# above; this number is the one thing in this file that is a judgement, and it
# is a judgement about where 876 measurements stop overlapping.
PROVENANCE_CUTOFF = 0.75

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
