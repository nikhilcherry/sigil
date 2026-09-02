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
    tbl.add_column("account")
    tbl.add_column("found via", style="dim")
    for i, s in enumerate(result.ranked[:8], 1):
        hit = s.similarity >= threshold
        tbl.add_row(
            str(i),
            Text(f"{s.similarity:.4f}", style="bold green" if hit else "yellow"),
            s.candidate.author_handle or s.candidate.platform,
            s.candidate.discovered_via.replace("app.bsky.", ""),
        )
    console.print(tbl)


def match_panel(evidence) -> None:
    m = evidence.match
    t = Table.grid(padding=(0, 2))
    t.add_column(style="dim", justify="right")
    t.add_column()
    t.add_row("platform", m.platform)
    t.add_row("account", f"{m.author_display_name} [cyan]@{m.author_handle}[/cyan]")
    t.add_row("post", f"[link={m.post_url}]{m.post_url}[/link]")
    t.add_row("image", m.image_url[:96])
    t.add_row("image sha256", m.image_sha256)
    if m.text:
        t.add_row("text", m.text[:160])
    t.add_row("similarity", f"[bold green]{evidence.similarity:.4f}[/bold green] "
                            f"[dim](threshold {evidence.threshold:.3f})[/dim]")
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
