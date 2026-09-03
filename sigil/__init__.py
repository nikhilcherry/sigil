"""sigil - face scan -> live social match -> tamper-evident on-chain record."""

__version__ = "0.1.0"
# v2 adds the fields that say what kind of claim the match is: the whole-image
# similarity to the probe and the identity/provenance verdict derived from it.
# Bumped rather than added silently, because a v1 bundle has no such fields and
# must not be read as though it asserted an identity claim.
SCHEMA = "sigil/evidence/v2"
