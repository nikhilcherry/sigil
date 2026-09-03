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
# Bumped rather than added silently both times: an older bundle has no such
# fields and must not be read as though it asserted them.
SCHEMA = "sigil/evidence/v3"
