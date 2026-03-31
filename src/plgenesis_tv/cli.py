"""CLI for PL Genesis × TrustAndVerify.

Commands:
  verify          — Run a verification query with optional Impulse AI pre-screening
  generate-dataset — Generate training CSV from reports or synthetic data
  hypercert       — Generate a Hypercert from a verification report
  train-info      — Show Impulse AI training instructions
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="plgenesis-tv",
    help="Verifiable AI Knowledge Verification with Autonomous ML Credibility Scoring",
    add_completion=False,
)
console = Console()


def _opinion_to_p(opinion) -> str:
    """Safely extract projected probability from an Opinion object or dict.

    Opinion.projected_probability() is a METHOD in jsonld-ex, not a property.
    """
    if opinion is None:
        return "N/A"
    if hasattr(opinion, "belief"):
        b = float(opinion.belief)
        u = float(opinion.uncertainty)
        br = float(getattr(opinion, "base_rate", 0.5))
        return f"{b + br * u:.3f}"
    elif isinstance(opinion, dict):
        b = opinion.get("belief", 0)
        u = opinion.get("uncertainty", 1)
        br = opinion.get("base_rate", 0.5)
        return f"{b + br * u:.3f}"
    return "N/A"


@app.command()
def verify(
    query: str = typer.Argument(..., help="The question or claim to verify"),
    impulse: bool = typer.Option(False, "--impulse", help="Enable Impulse AI pre-screening"),
    format: str = typer.Option("markdown", "--format", "-f", help="Output format: markdown, jsonld"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Save report JSON to file"),
    hypercert_out: Optional[str] = typer.Option(
        None, "--hypercert", help="Generate hypercert and save to this path"
    ),
):
    """Verify a knowledge claim using multi-source evidence and Subjective Logic."""

    console.print(Panel(f"[bold cyan]Verifying:[/] {query}", title="TrustAndVerify"))

    async def _run():
        # Import trustandverify
        try:
            from trustandverify import verify as tv_verify
        except ImportError:
            console.print(
                "[red]Error:[/] trustandverify not installed. "
                "Run: pip install trustandverify[tavily,gemini]"
            )
            raise typer.Exit(1)

        # Run verification
        with console.status("[bold green]Running verification pipeline..."):
            report = await tv_verify(query)

        # Display results
        _display_report(report)

        # Impulse AI pre-screening
        if impulse:
            await _run_impulse_screening(report)

        # Export report to JSON
        if output:
            _export_report(report, output)

        # Generate hypercert if requested
        if hypercert_out:
            _generate_hypercert_from_report(report, hypercert_out)

        return report

    asyncio.run(_run())


@app.command("generate-dataset")
def generate_dataset(
    synthetic: bool = typer.Option(True, help="Generate synthetic bootstrap data"),
    samples: int = typer.Option(200, "--samples", "-n", help="Number of synthetic samples"),
    output: str = typer.Option(
        "data/sample_verification_data.csv", "--output", "-o", help="Output CSV path"
    ),
    reports_dir: Optional[str] = typer.Option(
        None, "--reports-dir", help="Directory of JSON report files to extract features from"
    ),
):
    """Generate a training dataset for Impulse AI from verification data."""

    from plgenesis_tv.dataset_generator import generate_csv, generate_synthetic_dataset

    if reports_dir:
        reports_path = Path(reports_dir)
        if not reports_path.exists():
            console.print(f"[red]Directory not found:[/] {reports_dir}")
            raise typer.Exit(1)

        reports = []
        for f in reports_path.glob("*.json"):
            with open(f, encoding="utf-8") as fh:
                reports.append(json.load(fh))

        if not reports:
            console.print(f"[yellow]No JSON report files found in {reports_dir}[/]")
            raise typer.Exit(1)

        path = generate_csv(reports, output)
        console.print(f"[green]Generated dataset from {len(reports)} reports → {path}[/]")

    elif synthetic:
        path = generate_synthetic_dataset(n_samples=samples, output_path=output)
        console.print(
            Panel(
                f"[green]Generated {samples} synthetic samples → {path}[/]\n\n"
                f"Next steps:\n"
                f"1. Upload this CSV to [bold]https://app.impulselabs.ai[/]\n"
                f"2. Select [bold]'verdict'[/] as the target column\n"
                f"3. Click 'Train Model' and wait ~2-5 min\n"
                f"4. Copy the Deployment ID to your .env file",
                title="Dataset Ready for Impulse AI",
            )
        )
    else:
        console.print("[yellow]Specify --synthetic or --reports-dir[/]")
        raise typer.Exit(1)


@app.command("hypercert")
def hypercert_cmd(
    report_file: str = typer.Argument(..., help="Path to a verification report JSON file"),
    output: str = typer.Option("data/hypercert.json", "--output", "-o", help="Output path"),
    contributor: str = typer.Option(
        "trustandverify-agent", "--contributor", help="Contributor identifier"
    ),
):
    """Generate a Hypercert impact attestation from a verification report."""

    from plgenesis_tv.hypercerts_integration import report_to_hypercert, save_hypercert

    report_path = Path(report_file)
    if not report_path.exists():
        console.print(f"[red]File not found:[/] {report_file}")
        raise typer.Exit(1)

    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)

    hc = report_to_hypercert(report, contributor=contributor)
    content_hash = save_hypercert(hc, output)

    _print_hypercert_summary(hc, output, content_hash)


@app.command("train-info")
def train_info():
    """Show Impulse AI training instructions."""
    from plgenesis_tv.impulse_integration import create_training_instructions

    console.print(create_training_instructions())


# ── Internal helpers ─────────────────────────────────────────────────────────


def _display_report(report) -> None:
    """Display a verification report in the terminal."""
    table = Table(title="Verification Results", show_lines=True)
    table.add_column("Claim", style="cyan", max_width=50)
    table.add_column("Verdict", style="bold")
    table.add_column("P(true)", justify="right")
    table.add_column("Evidence", justify="right")

    claims = report.claims if hasattr(report, "claims") else report.get("claims", [])

    for claim in claims:
        if hasattr(claim, "text"):
            text = claim.text
            verdict = claim.verdict.value if hasattr(claim.verdict, "value") else str(claim.verdict)
            p = _opinion_to_p(claim.opinion)
            ev_count = str(len(claim.evidence))
        else:
            text = claim.get("text", "")
            verdict = claim.get("verdict", "unknown")
            p = _opinion_to_p(claim.get("opinion"))
            ev_count = str(len(claim.get("evidence", [])))

        color_map = {
            "supported": "[green]",
            "contested": "[yellow]",
            "refuted": "[red]",
            "no_evidence": "[dim]",
        }
        verdict_display = f"{color_map.get(verdict, '')}{verdict}"
        table.add_row(text[:50], verdict_display, p, ev_count)

    console.print(table)

    summary = report.summary if hasattr(report, "summary") else report.get("summary", "")
    if summary:
        console.print(Panel(summary, title="Summary"))


async def _run_impulse_screening(report) -> None:
    """Run Impulse AI pre-screening on report claims."""
    from plgenesis_tv.dataset_generator import extract_claim_features
    from plgenesis_tv.impulse_integration import ImpulseCredibilityScorer

    async with ImpulseCredibilityScorer() as scorer:
        if not scorer.is_available:
            console.print(
                "[yellow]Impulse AI not configured. Run 'plgenesis-tv train-info' for setup.[/]"
            )
            return

        console.print("\n[bold]Impulse AI Pre-Screening:[/]")
        claims = report.claims if hasattr(report, "claims") else report.get("claims", [])

        for claim in claims:
            if hasattr(claim, "__dict__"):
                claim_dict = _claim_to_dict(claim)
            else:
                claim_dict = claim

            features = extract_claim_features(claim_dict)
            feat_dict = asdict(features)
            result = await scorer.predict_claim(feat_dict)

            text = claim.text if hasattr(claim, "text") else claim.get("text", "")
            if result.get("impulse_available"):
                console.print(
                    f"  {text[:40]}... → "
                    f"Impulse: [bold]{result['impulse_verdict']}[/] "
                    f"({result['impulse_confidence']} confidence)"
                )
            else:
                console.print(f"  {text[:40]}... → [dim]Impulse unavailable[/]")


def _generate_hypercert_from_report(report, output_path: str) -> None:
    """Generate a Hypercert from a live Report object (not a JSON file)."""
    from plgenesis_tv.hypercerts_integration import report_to_hypercert, save_hypercert

    # Convert Report object to dict
    if hasattr(report, "claims"):
        report_dict = {
            "id": report.id if hasattr(report, "id") else "unknown",
            "query": report.query if hasattr(report, "query") else "",
            "claims": [_claim_to_dict(c) for c in report.claims],
            "summary": report.summary if hasattr(report, "summary") else "",
            "created_at": str(report.created_at) if hasattr(report, "created_at") else "",
        }
    else:
        report_dict = report

    hc = report_to_hypercert(report_dict)
    content_hash = save_hypercert(hc, output_path)
    _print_hypercert_summary(hc, output_path, content_hash)


def _print_hypercert_summary(hc, output_path: str, content_hash: str) -> None:
    """Print a Hypercert summary to the console."""
    console.print(
        Panel(
            f"[green]Hypercert generated → {output_path}[/]\n"
            f"Content hash: [dim]{content_hash}[/]\n\n"
            f"Name: {hc.name}\n"
            f"Claims verified: {hc.properties.get('total_claims', 0)}\n"
            f"Evidence gathered: {hc.properties.get('total_evidence', 0)}\n"
            f"Sources consulted: {hc.properties.get('unique_sources', 0)}",
            title="Hypercert Impact Attestation",
        )
    )


def _claim_to_dict(claim) -> dict:
    """Convert a trustandverify Claim object to a dict.

    Interfaces used (from trustandverify.core.models):
      Claim:    .text, .evidence, .opinion, .verdict, .assessment
      Evidence: .text, .supports_claim, .relevance, .confidence_raw, .source, .opinion
      Source:   .url, .title, .content_snippet, .trust_score, .source_type
      Opinion:  .belief, .disbelief, .uncertainty, .base_rate (all float attrs)
      Verdict:  .value (str from enum)
    """
    evidence_list = []
    for ev in claim.evidence:
        ev_dict = {
            "text": ev.text,
            "supports_claim": ev.supports_claim,
            "relevance": ev.relevance,
            "confidence_raw": ev.confidence_raw,
        }
        if ev.source:
            ev_dict["source"] = {
                "url": ev.source.url,
                "title": ev.source.title,
                "trust_score": ev.source.trust_score,
            }
        if ev.opinion:
            ev_dict["opinion"] = {
                "belief": ev.opinion.belief,
                "disbelief": ev.opinion.disbelief,
                "uncertainty": ev.opinion.uncertainty,
                "base_rate": ev.opinion.base_rate,
            }
        evidence_list.append(ev_dict)

    opinion_dict = None
    if claim.opinion:
        opinion_dict = {
            "belief": claim.opinion.belief,
            "disbelief": claim.opinion.disbelief,
            "uncertainty": claim.opinion.uncertainty,
            "base_rate": claim.opinion.base_rate,
        }

    return {
        "text": claim.text,
        "verdict": claim.verdict.value if hasattr(claim.verdict, "value") else str(claim.verdict),
        "evidence": evidence_list,
        "opinion": opinion_dict,
    }


def _export_report(report, output: str) -> None:
    """Export report to JSON file."""
    if hasattr(report, "claims"):
        data = {
            "id": report.id,
            "query": report.query,
            "claims": [_claim_to_dict(c) for c in report.claims],
            "summary": report.summary,
            "created_at": str(report.created_at),
            "metadata": report.metadata,
        }
    else:
        data = report

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    console.print(f"[green]Report exported → {output}[/]")


if __name__ == "__main__":
    app()
