"""Publishing the probe so that Google Lens can be asked about it at all.

Lens matches on a *URL*. It will not take bytes, so a local photograph cannot
be searched with it until the photograph is somewhere public - which is why
`SerpApiLensProvider.available_for` refuses unless the probe was already given
as an https URL, and why running `sigil run ./photo.jpg` with a SerpAPI key
configured silently has no Lens arm at all.

This closes that, and it is off by default, because of what it actually is:
**uploading a photograph of somebody's face to a third-party public host so
that a fourth party can index it.** For the ordinary subject of this tool -
someone who did not ask to be searched for - that is a disclosure, and it is
not one the tool should make on their behalf because a key happened to be in
the environment. So it needs `--publish-probe` or SIGIL_PUBLISH_PROBE, and it
says what it did.

Two details follow from taking that seriously rather than only saying it.

The host is the *temporary* one. The obvious choice is the permanent
paste-style host, and it is the wrong one: it keeps the file at a public URL
indefinitely, so a demonstration leaves a face online forever. This uses the
one-hour bucket, which is longer than any run and shorter than a mistake.

And nothing about the upload enters the evidence bundle. The published URL is
a means of asking a question, not a fact about the subject, and a bundle that
cited it would be citing a copy this tool made rather than something the
subject or anyone else published. What goes in the bundle is what Lens found.
"""

from __future__ import annotations

import logging

from ..config import Config
from .http import make_session

# The one-hour bucket of the same service, deliberately - see the docstring.
ENDPOINT = "https://litterbox.catbox.moe/resources/internals/api.php"
EXPIRY = "1h"

log = logging.getLogger(__name__)


def enabled_for(cfg: Config) -> bool:
    """Only worth doing when something would actually use the URL."""
    return bool(getattr(cfg, "publish_probe", False) and cfg.serpapi_key)


def publish_probe(image_bytes: bytes, cfg: Config) -> str | None:
    """Put the probe at a temporary public URL, or return None if that failed.

    Failure is not an error. The Lens arm is optional and every other arm still
    works, so a host that is down or slow costs coverage rather than the run -
    the same contract every optional provider here has.
    """
    if not image_bytes:
        return None
    session = make_session()
    try:
        r = session.post(
            ENDPOINT,
            data={"reqtype": "fileupload", "time": EXPIRY},
            files={"fileToUpload": ("probe.jpg", image_bytes, "image/jpeg")},
            timeout=cfg.http_timeout,
        )
    except Exception:  # noqa: BLE001 - an optional arm must not end a run
        return None
    if r.status_code != 200:
        return None
    url = (r.text or "").strip()
    # The API answers with the URL as a bare body, and with a plain-text error
    # message in the same place when it refuses - so "did it work" is "does it
    # look like a URL", not "was it a 200".
    if not url.startswith("https://"):
        return None
    return url
