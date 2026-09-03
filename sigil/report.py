"""Terminal rendering. Kept apart from the pipeline so the pipeline stays importable."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

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
    t.add_row("backend", f"{ref.backend} [dim]({ref.model})[/dim]")
    t.add_row("face bbox", str(ref.bbox))
    t.add_row("detector score", f"{ref.det_score:.4f}")
    t.add_row("image sha256", ref.image_sha256)
    t.add_row("embedding sha256", ref.embedding_sha256)
    console.print(Panel(t, title="Face encoded", border_style="cyan", expand=False))


def search_panel(result, threshold: float, providers: list[str]) -> None:
    t = Table.grid(padding=(0, 2))
    t.add_column(style="dim", justify="right")
    t.add_column()
    t.add_row("providers", ", ".join(providers))
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
    # only table the run prints.
    shown = list(enumerate(result.ranked, 1))[:8]
    if anchored and not any(s.image_sha256 == anchored for _, s in shown):
        shown += [(i, s) for i, s in enumerate(result.ranked, 1)
                  if s.image_sha256 == anchored][:1]
    for i, s in shown:
        hit = s.similarity >= threshold
        # "same photo" is the honest label for a reverse-image hit: the cosine
        # is near 1.0 because it is the probe's own picture, not because the
        # model made a hard call.
        identity = s.claim == "identity"
        tbl.add_row(
            ("[bold]" + str(i) + " ◀[/bold]"
             if anchored and s.image_sha256 == anchored else str(i)),
            Text(f"{s.similarity:.4f}", style="bold green" if hit else "yellow"),
            Text("different photo" if identity else "same photo",
                 style="green" if identity else "yellow"),
            s.candidate.author_handle or s.candidate.platform,
            s.candidate.discovered_via.replace("app.bsky.", ""),
        )
    console.print(tbl)
    if anchored:
        console.print("[dim]◀ anchored. A social post outranks an open-web "
                      "page, and a different photograph of the same face "
                      "outranks a higher cosine on the probe's own picture "
                      "republished.[/dim]")


def match_panel(evidence) -> None:
    m = evidence.match
    t = Table.grid(padding=(0, 2))
    t.add_column(style="dim", justify="right")
    t.add_column()
    t.add_row("platform", f"{m.platform} [dim]({m.source_kind})[/dim]")
    t.add_row("account", f"{m.author_display_name} [cyan]@{m.author_handle}[/cyan]")
    t.add_row("post", f"[link={m.post_url}]{m.post_url}[/link]")
    t.add_row("image", m.image_url[:96])
    t.add_row("image sha256", m.image_sha256)
    if m.text:
        t.add_row("text", m.text[:160])
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
    t.add_row("evidence hash", evidence.evidence_hash_hex())
    console.print(Panel(t, title="Match found", border_style="green", expand=False))


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
    if v.on_chain:
        t.add_row("submitter", v.on_chain["submitter"])
        t.add_row("anchored at", str(v.on_chain["anchored_at"]))
    for note in v.notes:
        t.add_row("note", f"[yellow]{note}[/yellow]")
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
            h.get("source", ""),
        )
    t.caption = f"accepted at ≥ {threshold:.2f} cosine"
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
                      e["a"], e["b"])
        console.print(a)

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
