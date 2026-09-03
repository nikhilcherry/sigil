# Demo probe images

Sample inputs for the pipeline. Both are photographs of public figures who also
maintain public Bluesky accounts, which is what makes them useful here: the
pipeline can find a *different* photograph of the same person through a live
search, rather than matching the file against itself.

| file | subject | role | licence |
|---|---|---|---|
| `probe-aoc.jpg` | Alexandria Ocasio-Cortez | probe | Public domain — Franmarie Metzler, U.S. House Office of Photography |
| `control-buttigieg.jpg` | Pete Buttigieg | negative control | Public domain — U.S. Department of Transportation, [via Commons](https://commons.wikimedia.org/wiki/File:Pete_Buttigieg,_Secretary_of_Transportation.jpg) |
| `probe-jay-graber.jpg` *(fetched)* | Jay Graber | probe | CC BY-SA 4.0 — Jennifer 8. Lee, via Wikimedia Commons |

Only the public-domain images are committed. Run `./fetch_examples.sh` to pull
the CC BY-SA one; it is not redistributed here because share-alike terms do not
sit cleanly inside an MIT repository.

`control-buttigieg.jpg` is the negative control: `tests/test_face.py` asserts
it scores −0.02 against the probe on insightface, which is the gap that makes a
fixed threshold defensible. Its licence sat unstated here for a while, and was
recovered by pointing this tool's own reverse-image arm at it — the run named
the Commons file, whose metadata gives the U.S. DOT as the author and public
domain as the terms.

```bash
sigil run examples/probe-aoc.jpg -q "AOC"

./examples/fetch_examples.sh
sigil run examples/probe-jay-graber.jpg -q "jay graber bluesky"
```

Use these, or any photo of someone with a public presence on the platform. A
face with no public footprint will correctly produce no match — which is the
honest outcome, not a bug.
