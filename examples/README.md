# Demo probe images

Sample inputs for the pipeline. Both are photographs of public figures who also
maintain public Bluesky accounts, which is what makes them useful here: the
pipeline can find a *different* photograph of the same person through a live
search, rather than matching the file against itself.

| file | subject | licence |
|---|---|---|
| `probe-aoc.jpg` | Alexandria Ocasio-Cortez | Public domain — Franmarie Metzler, U.S. House Office of Photography |
| `probe-jay-graber.jpg` *(fetched)* | Jay Graber | CC BY-SA 4.0 — Jennifer 8. Lee, via Wikimedia Commons |

Only the public-domain image is committed. Run `./fetch_examples.sh` to pull the
CC BY-SA one; it is not redistributed here because share-alike terms do not sit
cleanly inside an MIT repository.

```bash
sigil run examples/probe-aoc.jpg -q "AOC"

./examples/fetch_examples.sh
sigil run examples/probe-jay-graber.jpg -q "jay graber bluesky"
```

Use these, or any photo of someone with a public presence on the platform. A
face with no public footprint will correctly produce no match — which is the
honest outcome, not a bug.
