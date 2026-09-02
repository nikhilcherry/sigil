"""sigil command line.

Each pipeline stage is also its own command, so the whole thing can be driven
step by step (useful for a demo, and for debugging one stage in isolation),
while ``sigil run`` executes the full path in one go.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from . import __version__, report
from .chain import ChainClient
from .config import ARTIFACTS_DIR, Config, ensure_dirs
from .evidence import Evidence
from .face import load_encoder
from .pipeline import PipelineError, load_probe_bytes, run_pipeline, scan_probe
from .report import console

DEFAULT_EVIDENCE = ARTIFACTS_DIR / "evidence.json"


def _cfg(**overrides) -> Config:
    cfg = Config()
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    return cfg


def _load_evidence(path: Path) -> Evidence:
    if not path.exists():
        raise click.ClickException(f"evidence bundle not found: {path}")
    return Evidence.from_dict(json.loads(path.read_text()))


def _write_evidence(evidence: Evidence, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written as the exact canonical bytes that were hashed, so the file on disk
    # IS the preimage - no re-serialisation step can drift from what was anchored.
    path.write_bytes(evidence.canonical_json())


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="sigil")
def cli() -> None:
    """Face scan -> live social search -> tamper-evident on-chain record."""
    ensure_dirs()


# --------------------------------------------------------------------------- run


@cli.command()
@click.argument("image")
@click.option("-q", "--query", default="",
              help="Search terms. Omit to identify the face from the index first.")
@click.option("--backend", type=click.Choice(["auto", "insightface", "opencv"]),
              default=None, help="Face recognition backend.")
@click.option("--threshold", type=float, default=None, help="Cosine similarity cut-off.")
@click.option("--max-images", type=int, default=None, help="Cap on images to examine.")
@click.option("--chain", "chain_backend", type=click.Choice(["local", "rpc"]), default=None,
              help="local = persistent in-process EVM; rpc = a real node.")
@click.option("--no-anchor", is_flag=True, help="Stop after the match; write no record.")
@click.option("-o", "--out", type=click.Path(path_type=Path), default=DEFAULT_EVIDENCE,
              show_default=True, help="Where to write the evidence bundle.")
def run(image, query, backend, threshold, max_images, chain_backend, no_anchor, out):
    """Run the whole pipeline on IMAGE (a file path or an https URL)."""
    cfg = _cfg(face_backend=backend, threshold=threshold, max_images=max_images,
               chain_backend=chain_backend)
    total = 3 if no_anchor else 5

    shown = repr(query) if query else "[dim]from face[/dim]"
    console.print(f"[bold]sigil[/bold] {__version__}  ·  query [cyan]{shown}[/cyan]  "
                  f"·  chain [magenta]{cfg.chain_backend}[/magenta]")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  BarColumn(bar_width=30), TextColumn("{task.fields[note]}"),
                  console=console, transient=True) as progress:
        task = progress.add_task("searching", total=cfg.max_images, note="")

        def on_event(event: dict) -> None:
            if event.get("type") == "identify" and event.get("available"):
                progress.console.print(report.identity_table(event))
            elif event.get("type") == "query" and event.get("derived"):
                progress.console.print(
                    f"[dim]identified as[/dim] [bold cyan]{event['query']}[/bold cyan]"
                )
            if event.get("type") == "progress":
                progress.update(
                    task,
                    completed=event["examined"],
                    note=f"{event['scored']} faces scored · best {event['top']:.3f}",
                )

        report.stage(1, total, "Face scan")
        try:
            result = run_pipeline(image, query, cfg, do_anchor=not no_anchor,
                                  on_event=on_event)
        except PipelineError as exc:
            progress.stop()
            raise click.ClickException(str(exc)) from exc

    report.probe_panel(result.probe_ref, image)

    report.stage(2, total, "Web / social search")
    report.search_panel(result.match, cfg.threshold_for(result.probe_ref.backend),
                        result.providers_used)

    report.stage(3, total, "Face match")
    if not result.found:
        report.no_match_panel(result.match, cfg.threshold_for(result.probe_ref.backend))
        sys.exit(2)
    report.match_panel(result.evidence)

    _write_evidence(result.evidence, out)
    console.print(f"[dim]evidence bundle written to {out}[/dim]")

    if no_anchor:
        return

    report.stage(4, total, "Blockchain anchor")
    report.anchor_panel(result.anchor)

    report.stage(5, total, "Re-verification against chain")
    report.verification_panel(result.verification)
    if not result.verification.ok:
        sys.exit(1)


# -------------------------------------------------------------------------- scan


@cli.command()
@click.argument("image")
@click.option("--backend", type=click.Choice(["auto", "insightface", "opencv"]), default=None)
def scan(image, backend):
    """Stage 1 only: detect and encode the face in IMAGE."""
    cfg = _cfg(face_backend=backend)
    try:
        image_bytes, _ = load_probe_bytes(image, cfg)
        _, ref, _ = scan_probe(image_bytes, cfg)
    except PipelineError as exc:
        raise click.ClickException(str(exc)) from exc
    report.probe_panel(ref, image)


# ------------------------------------------------------------------------ search


@cli.command()
@click.argument("image")
@click.option("-q", "--query", default="")
@click.option("--backend", type=click.Choice(["auto", "insightface", "opencv"]), default=None)
@click.option("--threshold", type=float, default=None)
@click.option("--max-images", type=int, default=None)
@click.option("-o", "--out", type=click.Path(path_type=Path), default=DEFAULT_EVIDENCE)
def search(image, query, backend, threshold, max_images, out):
    """Stages 1-3: scan, search and match, without touching a chain."""
    ctx = click.get_current_context()
    ctx.invoke(run, image=image, query=query, backend=backend, threshold=threshold,
               max_images=max_images, chain_backend=None, no_anchor=True, out=out)


# ------------------------------------------------------------------------ anchor


@cli.command()
@click.option("-e", "--evidence", "evidence_path", type=click.Path(path_type=Path),
              default=DEFAULT_EVIDENCE, show_default=True)
@click.option("--chain", "chain_backend", type=click.Choice(["local", "rpc"]), default=None)
def anchor(evidence_path, chain_backend):
    """Stage 4 only: write an existing evidence bundle to the chain."""
    cfg = _cfg(chain_backend=chain_backend)
    ev = _load_evidence(evidence_path)
    client = ChainClient(cfg)
    console.print(f"[dim]contract {client.ensure_deployed()} on chain {client.chain_id}[/dim]")
    report.anchor_panel(client.anchor(ev))


# ------------------------------------------------------------------------ verify


@cli.command()
@click.option("-e", "--evidence", "evidence_path", type=click.Path(path_type=Path),
              default=DEFAULT_EVIDENCE, show_default=True)
@click.option("--chain", "chain_backend", type=click.Choice(["local", "rpc"]), default=None)
@click.option("--probe", type=click.Path(path_type=Path), default=None,
              help="Re-scan this face and confirm it is the one in the bundle.")
@click.option("--recheck-source", is_flag=True,
              help="Re-download the matched post image and confirm its bytes are unchanged.")
def verify(evidence_path, chain_backend, probe, recheck_source):
    """Stage 5: recompute the hash locally and check it against chain state."""
    cfg = _cfg(chain_backend=chain_backend)
    ev = _load_evidence(evidence_path)
    client = ChainClient(cfg)

    probe_digest = None
    if probe:
        image_bytes, _ = load_probe_bytes(str(probe), cfg)
        _, ref, _ = scan_probe(image_bytes, cfg)
        probe_digest = ref.embedding_sha256

    v = client.verify(ev, probe_embedding_sha256=probe_digest, recheck_source=recheck_source)
    report.verification_panel(v, title=f"Verification of {evidence_path.name}")
    sys.exit(0 if v.ok else 1)


# ------------------------------------------------------------------------ tamper


@cli.command()
@click.option("-e", "--evidence", "evidence_path", type=click.Path(path_type=Path),
              default=DEFAULT_EVIDENCE, show_default=True)
@click.option("--field", default="match.text", show_default=True,
              help="Dotted path of the field to alter, e.g. match.text or similarity.")
@click.option("--value", default=None, help="New value (default: append a character).")
@click.option("-o", "--out", type=click.Path(path_type=Path),
              default=ARTIFACTS_DIR / "evidence.tampered.json", show_default=True)
def tamper(evidence_path, field, value, out):
    """Produce an altered copy of a bundle, to demonstrate that verification fails."""
    data = json.loads(_load_evidence(evidence_path).canonical_json())
    node, *rest = field.split(".")
    target = data
    path = [node, *rest]
    for key in path[:-1]:
        target = target[key]
    leaf = path[-1]
    before = target[leaf]
    target[leaf] = value if value is not None else (
        f"{before}!" if isinstance(before, str) else before + 0.0001
    )

    ev = Evidence.from_dict(data)
    _write_evidence(ev, out)
    console.print(
        f"[yellow]{field}[/yellow]: {before!r} -> {target[leaf]!r}\n"
        f"original hash : {_load_evidence(evidence_path).evidence_hash_hex()}\n"
        f"tampered hash : {ev.evidence_hash_hex()}\n"
        f"[dim]written to {out} - now run:[/dim] sigil verify -e {out}"
    )


# ------------------------------------------------------------------------- chain


@cli.group()
def chain() -> None:
    """Inspect or reset the chain backend."""


@chain.command("info")
@click.option("--chain", "chain_backend", type=click.Choice(["local", "rpc"]), default=None)
def chain_info(chain_backend):
    """Show the connected chain, the deployed contract and the record count."""
    cfg = _cfg(chain_backend=chain_backend)
    client = ChainClient(cfg)
    addr = client.ensure_deployed()
    from rich.table import Table

    t = Table.grid(padding=(0, 2))
    t.add_column(style="dim", justify="right")
    t.add_column()
    t.add_row("backend", cfg.chain_backend)
    t.add_row("chain id", str(client.chain_id))
    t.add_row("contract", addr)
    t.add_row("submitter", client.address)
    t.add_row("records anchored", str(client.total_anchored()))
    if cfg.chain_backend == "rpc":
        bal = client.w3.from_wei(client.w3.eth.get_balance(client.address), "ether")
        t.add_row("balance", f"{bal} (native)")
    from rich.panel import Panel

    console.print(Panel(t, title="Chain", border_style="magenta", expand=False))


@chain.command("address")
@click.option("--chain", "chain_backend", type=click.Choice(["local", "rpc"]), default=None)
def chain_address(chain_backend):
    """Show the submitter address and balance, deploying nothing.

    Deliberately separate from `chain info`, which deploys the registry before
    it can report: on an unfunded key that fails, so there was no way to find
    out which address to fund without first trying to spend from it.
    """
    from rich.panel import Panel
    from rich.table import Table

    cfg = _cfg(chain_backend=chain_backend)
    client = ChainClient(cfg)

    t = Table.grid(padding=(0, 2))
    t.add_column(style="dim", justify="right")
    t.add_column()
    t.add_row("backend", cfg.chain_backend)
    t.add_row("chain id", str(client.chain_id))
    t.add_row("address", client.address)

    hint = None
    if cfg.chain_backend == "rpc":
        wei = client.w3.eth.get_balance(client.address)
        t.add_row("balance", f"{client.w3.from_wei(wei, 'ether')} (native)")
        if wei == 0:
            hint = ("[yellow]unfunded[/yellow] - send testnet funds to the address "
                    "above, then run [bold]sigil chain info[/bold] to deploy.")
    else:
        t.add_row("balance", "[dim]n/a - the local chain pre-funds it[/dim]")

    console.print(Panel(t, title="Chain account", border_style="magenta", expand=False))
    if hint:
        console.print(hint)


@chain.command("reset")
@click.confirmation_option(prompt="Delete the local chain state and start over?")
def chain_reset():
    """Wipe the persisted local chain (local backend only)."""
    from .config import STATE_PATH

    if STATE_PATH.exists():
        STATE_PATH.unlink()
        console.print(f"[yellow]removed {STATE_PATH}[/yellow]")
    else:
        console.print("[dim]no local chain state to remove[/dim]")


@cli.command()
@click.argument("image")
@click.option("--backend", type=click.Choice(["auto", "insightface", "opencv"]), default=None)
@click.option("--top", default=5, show_default=True, help="How many candidates to show.")
def identify(image, backend, top):
    """Name the face in IMAGE using the local identity index."""
    from .identify import IdentityIndex
    from .pipeline import IDENTITY_THRESHOLD

    cfg = _cfg(face_backend=backend)
    try:
        image_bytes, _ = load_probe_bytes(image, cfg)
        face, ref, encoder = scan_probe(image_bytes, cfg)
        index = IdentityIndex.load(encoder)
    except (PipelineError, FileNotFoundError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc

    hits = index.query(face.embedding, top=top)
    threshold = IDENTITY_THRESHOLD.get(encoder.name, 0.45)
    report.identity_table({
        "index_size": len(index),
        "threshold": threshold,
        "hits": [{"name": h.identity.name, "similarity": round(h.similarity, 4),
                  "source": h.identity.source, "qid": h.identity.qid,
                  "accepted": h.similarity >= threshold} for h in hits],
    }, echo=True)


@cli.group()
def index() -> None:
    """Build or inspect the face-to-name identity index."""


@index.command("build")
@click.option("--langs", default=None,
              help="Comma-separated wiki languages (default: a 10-language spread).")
@click.option("--months", default=3, show_default=True,
              help="How many months of most-viewed articles to harvest.")
@click.option("--limit", type=int, default=None, help="Cap the number of portraits.")
@click.option("--backend", type=click.Choice(["auto", "insightface", "opencv"]), default=None)
def index_build(langs, months, limit, backend):
    """Harvest public figures from Wikipedia/Wikidata and encode their faces."""
    from .identify import DEFAULT_LANGS, build_index

    encoder = load_encoder(_cfg(face_backend=backend).face_backend)
    chosen = tuple(x.strip() for x in langs.split(",")) if langs else DEFAULT_LANGS
    console.print(f"[dim]building identity index · backend {encoder.name} · "
                  f"langs {', '.join(chosen)}[/dim]")
    count = build_index(encoder, langs=chosen, months=months, limit=limit,
                        on_progress=lambda m: console.print(f"[dim]{m}[/dim]"))
    console.print(f"[green]indexed {count} faces[/green]")


@index.command("info")
def index_info():
    """Show what is currently in the identity index."""
    from .identify import IdentityIndex

    try:
        idx = IdentityIndex.load()
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc
    from rich.panel import Panel
    from rich.table import Table

    t = Table.grid(padding=(0, 2))
    t.add_column(style="dim", justify="right")
    t.add_column()
    t.add_row("faces", str(len(idx)))
    t.add_row("backend", idx.backend)
    t.add_row("dimensions", str(idx.vectors.shape[1]))
    sources = {}
    for i in idx.identities:
        sources[i.source] = sources.get(i.source, 0) + 1
    for src, n in sorted(sources.items(), key=lambda kv: -kv[1]):
        t.add_row(src, str(n))
    console.print(Panel(t, title="Identity index", border_style="cyan", expand=False))


@cli.command()
@click.option("--port", default=8099, show_default=True)
@click.option("--host", default="127.0.0.1", show_default=True,
              help="Bind address. Localhost by default, and it should stay that way.")
@click.option("--no-browser", is_flag=True, help="Do not open a browser window.")
def serve(port, host, no_browser):
    """Launch the local web UI and watch the pipeline run live."""
    from .web import serve as run_server

    run_server(host=host, port=port, open_browser=not no_browser)


@cli.command()
def backends():
    """Report which face backends this machine can actually load."""
    from rich.table import Table

    t = Table(header_style="bold", border_style="dim")
    t.add_column("backend")
    t.add_column("status")
    t.add_column("model")
    t.add_column("runs on")
    for name in ("insightface", "opencv"):
        try:
            enc = load_encoder(name)
            status = "[green]ready[/green]" if enc.name == name else "[yellow]fell back[/yellow]"
            # The provider actually in use, read back from the loaded session -
            # asking for CUDA is not evidence that CUDA answered.
            runs_on = getattr(enc, "provider", "CPUExecutionProvider")
            t.add_row(name, status, enc.model, runs_on.replace("ExecutionProvider", ""))
        except Exception as exc:  # noqa: BLE001
            t.add_row(name, "[red]unavailable[/red]", str(exc)[:60], "-")
    console.print(t)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
