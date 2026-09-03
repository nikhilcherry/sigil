"""Terminal rendering. Kept apart from the pipeline so the pipeline stays importable."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


def q(value: Any) -> str:
    """Render a value that somebody else wrote, as text rather than as markup.

    rich reads `[...]` as styling, so a Bluesky display name of
    `[bold green]VERIFIED[/bold green]` printed itself in bold green - the same
    styling this report uses for a passing check - and `[link=...]` makes a
    clickable link in terminals that support it. Every field a provider or
    Wikidata supplies goes through here.

    This matters more than it looks: the terminal output is what a screen
    recording shows as the evidence, so an account able to style its own row
    can forge the appearance of a verdict.
    """
    return escape(str(value))

OK = "[bold green]PASS[/bold green]"
BAD = "[bold red]FAIL[/bold red]"


def stage(n: int, total: int, title: str) -> None:
    console.print()
    console.rule(f"[bold cyan]Stage {n}/{total} · {title}[/bold cyan]", align="left")


def probe_panel(ref, path: str) -> None:
    t = Table.grid(padding=(0, 2))
    t.add_column(style="dim", justify="right")
    t.add_column()
    t.add_row("source", path)
    t.add_row("backend", f"{ref.backend} [dim]({ref.model})[/dim]"
                         + (f" [dim]on {ref.provider.replace('ExecutionProvider', '')}"
                            f"[/dim]" if ref.provider else ""))
    t.add_row("face bbox", str(ref.bbox))
    t.add_row("detector score", f"{ref.det_score:.4f}")
    t.add_row("image sha256", ref.image_sha256)
    t.add_row("embedding sha256", ref.embedding_sha256)
    console.print(Panel(t, title="Face encoded", border_style="cyan", expand=False))


def search_panel(result, threshold: float, providers: list[str]) -> None:
    t = Table.grid(padding=(0, 2))
    t.add_column(style="dim", justify="right")
    t.add_column()
    # Provider names are this project's own constants rather than anything a
    # third party supplies, but they go through q() like every other rendered
    # value: the rule is cheaper to keep than to reason about per field.
    t.add_row("providers", ", ".join(q(p) for p in providers))
    t.add_row("images fetched", str(result.images_examined))
    t.add_row("images with a face", str(result.images_with_faces))
    t.add_row("faces compared", str(result.faces_examined))
    if getattr(result, "inference_reused", 0):
        t.add_row("duplicate images", f"{result.inference_reused} (score reused)")
    t.add_row("threshold", f"{threshold:.3f} cosine")
    calls = sum(len(p["calls"]) for p in result.trace)
    t.add_row("live API calls", str(calls))
    console.print(Panel(t, title="Search completed", border_style="cyan", expand=False))

    if not result.ranked:
        return
    tbl = Table(title="Top candidates by face similarity", header_style="bold",
                border_style="dim")
    tbl.add_column("#", justify="right", style="dim")
    tbl.add_column("similarity", justify="right")
    tbl.add_column("claim")
    tbl.add_column("account")
    tbl.add_column("found via", style="dim")
    anchored = result.best.image_sha256 if result.best else None
    # The anchored candidate is often not in the top few - that is the whole
    # point of the preference - so it is pulled in rather than left off the
    # only table the run prints. Its number is its real rank among everything
    # scored, which is why that rank travels on the candidate.
    shown = result.ranked[:8]
    marked = any(s.image_sha256 == anchored for s in shown) if anchored else False
    if anchored and not marked:
        extra = [s for s in result.ranked if s.image_sha256 == anchored][:1]
        shown = shown + extra
        marked = bool(extra)
    for s in shown:
        i = s.rank or (result.ranked.index(s) + 1)
        hit = s.similarity >= threshold
        # "same photo" is the honest label for a reverse-image hit: the cosine
        # is near 1.0 because it is the probe's own picture, not because the
        # model made a hard call.
        identity = s.claim == "identity"
        tbl.add_row(
            ("[bold]" + str(i) + " ◀[/bold]"
             if anchored and s.image_sha256 == anchored else str(i)),
            Text(f"{s.similarity:.4f}", style="bold green" if hit else "yellow"),
            # The face count only earns space when it is not 1, which is the
            # overwhelming majority - a column of "1" would wrap the table and
            # tell a reader nothing.
            Text(("different photo" if identity else "same photo")
                 + (f" ·{s.faces_in_image}" if s.faces_in_image > 1 else ""),
                 style="green" if identity else "yellow"),
            q(s.candidate.author_handle or s.candidate.platform),
            q(s.candidate.discovered_via.replace("app.bsky.", "")),
        )
    console.print(tbl)
    if marked:
        console.print("[dim]◀ anchored. A social post outranks an open-web "
                      "page, and a different photograph of the same face "
                      "outranks a higher cosine on the probe's own picture "
                      "republished.[/dim]")


def match_panel(evidence, scored=None) -> None:
    """Render the anchored match.

    ``scored`` is the ScoredCandidate behind it, when the caller has it. It
    carries two facts the evidence bundle does not: how many faces were in the
    matched image and which one matched. A match in a twelve-person group photo
    means less than the same score on a portrait, and the calibration section
    names exactly that as a source of error - so it is worth saying rather than
    leaving the reader to assume a headshot.
    """
    m = evidence.match
    t = Table.grid(padding=(0, 2))
    t.add_column(style="dim", justify="right")
    t.add_column()
    t.add_row("platform", f"{q(m.platform)} [dim]({q(m.source_kind)})[/dim]")
    t.add_row("account", f"{q(m.author_display_name)} [cyan]@{q(m.author_handle)}[/cyan]")
    t.add_row("post", q(m.post_url))
    t.add_row("image", q(m.image_url[:96]))
    t.add_row("image sha256", q(m.image_sha256))
    if m.text:
        t.add_row("text", q(m.text[:160]))
    t.add_row("similarity", f"[bold green]{evidence.similarity:.4f}[/bold green] "
                            f"[dim](threshold {evidence.threshold:.3f})[/dim]")
    if m.claim == "identity":
        t.add_row("claim", "[bold green]identity[/bold green] [dim]— a different "
                           "photograph of the same face[/dim]")
    else:
        t.add_row("claim", "[bold yellow]provenance[/bold yellow] [dim]— the "
                           "probe's own photograph, published here[/dim]")
    t.add_row("picture vs probe", f"{m.probe_photo_similarity:.4f} "
                                  f"[dim](whole image, no face model)[/dim]")
    if scored is not None and scored.faces_in_image:
        faces = scored.faces_in_image
        where = f" [dim]at {scored.matched_bbox}[/dim]" if scored.matched_bbox else ""
        t.add_row("faces in that image",
                  (f"{faces}{where}" if faces == 1 else
                   f"[yellow]{faces}[/yellow]{where} [dim]— a group photo, so the "
                   f"match is one face among several[/dim]"))
    t.add_row("evidence hash", evidence.evidence_hash_hex())
    rates = _measured_rates(evidence)
    if rates:
        tpr, fpr = rates
        t.add_row("measured error rate",
                  f"{_sci(fpr)} [dim]false accepts at this threshold, "
                  f"catching {tpr * 100:.1f}% of true pairs — "
                  f"`sigil calibrate --show`[/dim]")
    console.print(Panel(t, title="Match found", border_style="green", expand=False))


def _current_calibration():
    """The saved calibration, only if it still describes the index in place.

    The impostor rates are counts over a specific set of faces. Rebuilding the
    index changes that set, and the count alone cannot tell one index of 3,583
    faces from another - so the calibration records a hash of the vectors it
    measured, and a mismatch means the numbers are about something else. No
    index at all means there is nothing to confirm against, which is the same
    answer.
    """
    try:
        from .calibrate import Calibration, index_digest
        from .identify import IdentityIndex

        cal = Calibration.load()
        if not cal.index_sha256:
            return None  # written before the hash existed; cannot be confirmed
        if cal.index_sha256 != index_digest(IdentityIndex.load()):
            return None
        return cal
    except Exception:  # noqa: BLE001 - absent, stale or unreadable is not an error
        return None


def _false_name_rate(threshold: float) -> float | None:
    """The measured chance of naming an unknown face, if it has been measured.

    Only reported when the calibration was taken at the threshold in use -
    the rate changes steeply with it, so quoting one for a different bar
    would be worse than quoting none.
    """
    cal = _current_calibration()
    if cal is None:
        return None
    if cal.identify_threshold is None or abs(cal.identify_threshold - threshold) > 1e-9:
        return None
    return cal.false_name_rate


def _measured_rates(evidence):
    """The measured rates for this run's threshold, if a calibration exists.

    Read from disk and shown rather than written into the bundle. Putting it in
    the bundle would make the anchored hash depend on whether the machine that
    produced it happened to have run `sigil calibrate`, so two correct runs of
    the same match would disagree. It belongs to the reader, not the record.
    """
    cal = _current_calibration()
    if cal is None or cal.backend != evidence.probe.backend:
        return None
    return cal.rates_at(evidence.threshold)


def no_match_panel(result, threshold: float) -> None:
    top = result.ranked[0].similarity if result.ranked else 0.0
    console.print(
        Panel(
            f"No candidate cleared the {threshold:.3f} threshold.\n"
            f"Best similarity seen: [yellow]{top:.4f}[/yellow] across "
            f"{result.images_examined} images.\n\n"
            "[dim]Nothing is anchored when nothing matched - that is the point. "
            "Try a broader --query, raise --max-images, or use a probe photo of "
            "someone with a public presence on the platform.[/dim]",
            title="No match",
            border_style="yellow",
            expand=False,
        )
    )


def anchor_panel(a: dict[str, Any]) -> None:
    t = Table.grid(padding=(0, 2))
    t.add_column(style="dim", justify="right")
    t.add_column()
    if a.get("already_anchored"):
        t.add_row("status", "[yellow]already anchored[/yellow] (contract rejects duplicates)")
        t.add_row("anchored at", str(a.get("anchored_at")))
    else:
        t.add_row("tx hash", a.get("tx_hash", ""))
        t.add_row("block", str(a.get("block_number", "")))
        t.add_row("gas used", f"{a.get('gas_used', 0):,}")
        t.add_row("chain id", str(a.get("chain_id", "")))
    t.add_row("contract", a.get("contract", ""))
    t.add_row("evidence hash", a.get("evidence_hash", ""))
    t.add_row("subject ref", a.get("subject_ref", ""))
    if a.get("explorer"):
        t.add_row("explorer", f"[link={a['explorer']}]{a['explorer']}[/link]")
    console.print(Panel(t, title="Anchored on chain", border_style="magenta", expand=False))


def verification_panel(v, title: str = "Verification") -> None:
    t = Table.grid(padding=(0, 2))
    t.add_column(style="dim", justify="right")
    t.add_column()
    t.add_row("evidence hash", v.evidence_hash)
    t.add_row("record on chain", OK if v.anchored else BAD)
    t.add_row("similarity matches", OK if v.similarity_matches else BAD)
    t.add_row("subject commitment", OK if v.subject_matches else BAD)
    if v.probe_matches is not None:
        t.add_row("probe re-encodes", OK if v.probe_matches else BAD)
    if v.source_image_intact is not None:
        t.add_row("source image intact", OK if v.source_image_intact else BAD)
    if v.claim_reproduces is not None:
        t.add_row("claim re-derives", OK if v.claim_reproduces else BAD)
    if v.on_chain:
        t.add_row("submitter", v.on_chain["submitter"])
        t.add_row("anchored at", str(v.on_chain["anchored_at"]))
    for note in v.notes:
        t.add_row("note", f"[yellow]{q(note)}[/yellow]")
    console.print(
        Panel(
            t,
            title=f"{title}: {'[bold green]VERIFIED[/bold green]' if v.ok else '[bold red]NOT VERIFIED[/bold red]'}",
            border_style="green" if v.ok else "red",
            expand=False,
        )
    )


def identity_table(event: dict[str, Any], echo: bool = False):
    """Render identity-index candidates, marking which cleared the bar."""
    t = Table(title=f"Identity index · {event.get('index_size', '?')} known faces",
              header_style="bold", border_style="dim")
    t.add_column("similarity", justify="right")
    t.add_column("name")
    t.add_column("source", style="dim")
    threshold = event.get("threshold", 0.45)
    for h in event.get("hits", []):
        ok = h.get("accepted", h["similarity"] >= threshold)
        t.add_row(
            Text(f"{h['similarity']:.4f}", style="bold green" if ok else "yellow"),
            Text(h["name"], style="bold" if ok else "dim"),
            q(h.get("source", "")),
        )
    t.caption = f"accepted at ≥ {threshold:.2f} cosine"
    rate = _false_name_rate(threshold)
    if rate is not None:
        t.caption += (f" · measured: {rate * 100:.2f}% of faces this index does "
                      f"not contain are named anyway")
    if echo:
        console.print(t)
        return None
    return t


def _sci(x: float) -> str:
    """A rate as both a probability and a "one in N", which is the readable half."""
    if x <= 0:
        return "0 [dim](none in the whole set)[/dim]"
    return f"{x:.3e} [dim](1 in {round(1 / x):,})[/dim]"


def calibration_panel(c) -> None:
    """Render a measured threshold calibration: what 0.38 actually costs."""
    head = Table.grid(padding=(0, 2))
    head.add_column(style="dim", justify="right")
    head.add_column()
    head.add_row("backend", f"{c.backend} [dim]({c.model})[/dim]")
    head.add_row("threshold under test", f"[bold]{c.threshold:.3f}[/bold]")
    head.add_row("genuine pairs",
                 f"{c.genuine.pairs:,} [dim]from {c.sampled_identities} people with "
                 f"{c.portraits_encoded} portraits between them[/dim]")
    if c.born_after is not None:
        head.add_row("photographic era",
                     f"born after {c.born_after} [dim]— "
                     f"{c.sampled_photographic} of {c.sampled_requested} sampled "
                     f"identities qualify; the rest are painted or sculpted[/dim]")
    head.add_row("impostor pairs",
                 f"{c.impostor.pairs:,} [dim]every cross-identity pair of "
                 f"{c.index_identities:,} indexed people[/dim]")
    console.print(Panel(head, title="Threshold calibration", border_style="cyan",
                        expand=False))

    d = Table(header_style="bold", border_style="dim", title="Similarity distributions")
    d.add_column("population")
    for col in ("pairs", "mean", "sd", "min", "median", "max"):
        d.add_column(col, justify="right")
    d.add_row("same person", f"{c.genuine.pairs:,}", f"{c.genuine.mean:.4f}",
              f"{c.genuine.sd:.4f}", f"{c.genuine.minimum:.4f}",
              f"{c.genuine.quantiles.get('p50', float('nan')):.4f}",
              f"{c.genuine.maximum:.4f}")
    d.add_row("different people", f"{c.impostor.pairs:,}", f"{c.impostor.mean:.4f}",
              f"{c.impostor.sd:.4f}", f"{c.impostor.minimum:.4f}",
              f"{c.impostor.quantiles.get('p50', float('nan')):.4f}",
              f"{c.impostor.maximum:.4f}")
    console.print(d)

    r = Table.grid(padding=(0, 2))
    r.add_column(style="dim", justify="right")
    r.add_column()
    r.add_row("true positive rate", f"[bold green]{c.tpr * 100:.2f}%[/bold green] "
                                    f"[dim]of same-person pairs are caught[/dim]")
    r.add_row("false positive rate", _sci(c.fpr))
    if c.fpr_excluding_artefacts is None:
        r.add_row("  discounting artefacts",
                  f"[yellow]nothing left to measure[/yellow] [dim]— all "
                  f"{c.artefact_pairs} impostor pairs are at ≥ 0.99, so this "
                  f"index holds duplicates and little else[/dim]")
    else:
        r.add_row("  discounting artefacts",
                  f"{_sci(c.fpr_excluding_artefacts)} [dim]dropping the "
                  f"{c.artefact_pairs} pairs at ≥ 0.99, which are one person "
                  f"indexed twice[/dim]")
    r.add_row("equal error rate",
              f"{c.eer * 100:.2f}% [dim]at threshold {c.eer_threshold:.2f}[/dim]")
    for name, t in c.thresholds_for_fpr.items():
        r.add_row(f"threshold for FPR {name}", f"{t:.4f}")
    console.print(Panel(r, title=f"At threshold {c.threshold:.3f}",
                        border_style="green", expand=False))

    if c.artefact_examples:
        a = Table(header_style="bold", border_style="dim",
                  title="Closest 'different people' pairs — read these before trusting the FPR")
        a.add_column("similarity", justify="right")
        a.add_column("index says")
        a.add_column("and")
        for e in c.artefact_examples:
            a.add_row(Text(f"{e['similarity']:.4f}",
                           style="red" if e["similarity"] >= 0.99 else "yellow"),
                      q(e["a"]), q(e["b"]))
        console.print(a)

    if c.false_name_rate is not None:
        n = Table.grid(padding=(0, 2))
        n.add_column(style="dim", justify="right")
        n.add_column()
        n.add_row("identity threshold", f"[bold]{c.identify_threshold:.2f}[/bold]")
        n.add_row("puts a wrong name to a face",
                  f"[bold yellow]{c.false_name_rate * 100:.2f}%[/bold yellow] "
                  f"[dim]of queries[/dim]")
        n.add_row("  ignoring duplicate entries",
                  f"{c.false_name_rate_excluding_artefacts * 100:.2f}%")
        console.print(Panel(
            n,
            title=f"Naming a face — {c.index_identities:,} candidates per query",
            border_style="yellow", expand=False))
        console.print(
            "[dim]Measured leave-one-out: every indexed face queried against every "
            "other, which is exactly the dangerous case — a probe the index does "
            "not contain. Note how far this is from the pair-level rate above. "
            "One query is thousands of chances to be wrong, so a false-accept "
            "rate of 1 in 35,000 per pair is percents per question.[/dim]"
        )
        if c.wrongly_named:
            w = Table(header_style="bold", border_style="dim",
                      title="Faces the index would misname")
            w.add_column("similarity", justify="right")
            w.add_column("this face")
            w.add_column("would be called")
            w.add_column("", style="dim")
            for e in c.wrongly_named:
                w.add_row(Text(f"{e['similarity']:.4f}", style="yellow"),
                          q(e["queried"]), q(e["named"]),
                          "duplicate index entry" if e["duplicate_entry"] else "")
            console.print(w)

    if c.hardest_genuine:
        h = Table(header_style="bold", border_style="dim",
                  title="Hardest same-person pairs")
        h.add_column("similarity", justify="right")
        h.add_column("person")
        for e in c.hardest_genuine:
            h.add_row(Text(f"{e['similarity']:.4f}",
                           style="green" if e["similarity"] >= c.threshold else "red"),
                      e["name"])
        console.print(h)

    # The one sentence a reader needs: the threshold is not at the point that
    # minimises total error, and that is deliberate.
    side = ("stricter" if c.threshold > c.eer_threshold else "looser")
    console.print(Panel(
        f"At {c.threshold:.3f} the model catches [bold]{c.tpr * 100:.1f}%[/bold] of "
        f"same-person pairs and falsely accepts {_sci(c.fpr)}.\n"
        f"That is {side} than the equal-error point of {c.eer_threshold:.2f}: it "
        f"gives up recall to buy a lower false-accept rate, which is the right way "
        f"round for a tool that puts a name to a stranger.\n\n"
        "[dim]Genuine pairs are different lead photographs of one person across "
        "language Wikipedias. Some are crops of one file, which flatters the true "
        "positive rate; the subject of a group photo is taken to be its largest "
        "face, which depresses it. Neither is corrected by hand. Impostor pairs "
        "are every cross-identity pair in the index as it actually stands, "
        "duplicate entities and all - which is what a probe is really compared "
        "against.[/dim]",
        title="What this argues", border_style="cyan", expand=False))


def records_table(rows: list[dict[str, Any]], total: int) -> None:
    """What the registry holds, read back from chain state rather than a log."""
    if not rows:
        console.print(Panel(
            "The registry is empty. `sigil run` anchors a record; nothing is "
            "anchored when nothing matched.",
            title="Chain records", border_style="yellow", expand=False))
        return

    t = Table(title=f"Registry contents · {total} record(s) anchored",
              header_style="bold", border_style="dim")
    t.add_column("#", justify="right", style="dim")
    t.add_column("evidence hash")
    t.add_column("similarity", justify="right")
    t.add_column("submitter", style="dim")
    t.add_column("anchored at", justify="right", style="dim")
    for r in rows:
        t.add_row(
            str(r["index"]),
            q(r["evidence_hash"][:22]) + "…",
            f"{r['similarity_bps'] / 10000:.4f}",
            q(r["submitter"][:12]) + "…",
            str(r["anchored_at"]),
        )
    console.print(t)
    if len(rows) < total:
        console.print(f"[dim]showing {len(rows)} of {total}; -n to see more[/dim]")
