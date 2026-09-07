"""sigil - face scan -> live social match -> tamper-evident on-chain record."""

__version__ = "0.1.0"
# v2 added the fields saying what kind of claim the match is: the whole-image
# similarity to the probe and the identity/provenance verdict derived from it.
#
# v3 adds the execution provider that produced the probe's embedding. That is
# not bookkeeping: the same image on CPU and on CUDA gives embeddings agreeing
# to 0.9996, which is indistinguishable for a similarity threshold and utterly
# different once put through sha256. Since the bundle records the digest rather
# than the vector - deliberately, so it holds no biometric - `sigil verify
# --probe` compares digests exactly, and a bundle made on one provider cannot
# re-encode on the other. Recording which one made it turns that failure from
# an accusation into an explanation.
#
# v4 does two things, and both change the bytes rather than only adding to
# them. It records how many faces were in the probe, of which the bundle
# describes exactly one - the largest - because a photograph of three people
# silently becomes a search for whichever face was biggest, and a reader could
# not see that ambiguity from a v3 bundle. And it puts every third-party string
# into one spelling before hashing (sigil/canonical.py): without that, the same
# post found through a link carrying `?utm_source=` produced an unrelated
# evidence hash, so one finding had two anchors and the registry's
# duplicate-refusal could not tell they were the same finding.
#
# Bumped rather than added silently each time: an older bundle has no such
# fields and must not be read as though it asserted them.
SCHEMA = "sigil/evidence/v4"
