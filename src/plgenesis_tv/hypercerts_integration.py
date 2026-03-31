"""Hypercerts integration for impact attestation from verification reports.

Converts trustandverify reports into Hypercert-compatible impact claims,
enabling decentralized tracking and funding of knowledge verification work.

Each verification report becomes a hypercert that records:
- Who performed the verification (contributor)
- What was verified (work scope = the query + claims)
- When it was verified (work timeframe)
- What impact it had (evidence gathered, conflicts resolved, confidence achieved)
- Provenance metadata (sources consulted, math used)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class HypercertClaim:
    """A Hypercert-compatible impact claim derived from a verification report.

    Follows the Hypercerts schema for interoperability with the Hypercerts protocol.
    See: https://hypercerts.org/docs/developer/api/contracts/protocol
    """

    # Identity
    name: str
    description: str

    # Work scope — what was done
    work_scope: list[str] = field(default_factory=list)
    work_timeframe_start: str = ""
    work_timeframe_end: str = ""

    # Impact scope — what effect it had
    impact_scope: list[str] = field(default_factory=list)
    impact_timeframe_start: str = ""
    impact_timeframe_end: str = ""

    # Contributors
    contributors: list[str] = field(default_factory=list)

    # Evidence & metrics
    properties: dict[str, Any] = field(default_factory=dict)

    # Verification metadata
    verification_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @property
    def content_hash(self) -> str:
        """Deterministic hash of the hypercert content for integrity verification."""
        content = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()


def report_to_hypercert(
    report: dict[str, Any],
    contributor: str = "trustandverify-agent",
) -> HypercertClaim:
    """Convert a trustandverify report into a Hypercert impact claim.

    Args:
        report: A trustandverify report dictionary (from JSON-LD export or agent output).
        contributor: Identifier for who performed the verification.

    Returns:
        HypercertClaim ready for on-chain attestation or JSON export.
    """
    query = report.get("query", "Unknown query")
    claims = report.get("claims", [])
    created_at = report.get("created_at", datetime.now(timezone.utc).isoformat())
    report_id = report.get("id", "unknown")

    # Extract claim texts for work scope
    claim_texts = [c.get("text", "") for c in claims if c.get("text")]

    # Calculate aggregate impact metrics
    total_evidence = sum(len(c.get("evidence", [])) for c in claims)
    verdicts = [c.get("verdict", "no_evidence") for c in claims]
    verdict_counts = {v: verdicts.count(v) for v in set(verdicts)}

    supported = verdict_counts.get("supported", 0)
    contested = verdict_counts.get("contested", 0)
    refuted = verdict_counts.get("refuted", 0)
    no_evidence = verdict_counts.get("no_evidence", 0)

    # Average projected probability across claims with opinions
    opinions = [c.get("opinion") for c in claims if c.get("opinion")]
    avg_proj_prob = 0.0
    if opinions:
        probs = [
            o.get("belief", 0) + o.get("base_rate", 0.5) * o.get("uncertainty", 1.0)
            for o in opinions
        ]
        avg_proj_prob = sum(probs) / len(probs)

    # Extract unique sources
    all_sources = set()
    for claim in claims:
        for ev in claim.get("evidence", []):
            src = ev.get("source", {})
            url = src.get("url", "")
            if url:
                all_sources.add(url)

    # Build the hypercert
    now_iso = datetime.now(timezone.utc).isoformat()

    return HypercertClaim(
        name=f"Knowledge Verification: {query[:80]}",
        description=(
            f"Agentic verification of the query '{query}' using Subjective Logic "
            f"confidence algebra (Jøsang 2016). Decomposed into {len(claims)} claims, "
            f"gathered {total_evidence} pieces of evidence from {len(all_sources)} sources, "
            f"and produced formal mathematical confidence scores with full provenance."
        ),
        work_scope=[
            "knowledge-verification",
            "claim-decomposition",
            "evidence-gathering",
            "confidence-scoring",
            "subjective-logic",
        ] + claim_texts[:5],  # Include up to 5 claim texts
        work_timeframe_start=created_at,
        work_timeframe_end=now_iso,
        impact_scope=[
            "verifiable-ai",
            "misinformation-detection",
            "decision-support",
            "provenance-tracking",
        ],
        impact_timeframe_start=created_at,
        impact_timeframe_end=now_iso,
        contributors=[contributor],
        properties={
            "query": query,
            "report_id": report_id,
            "total_claims": len(claims),
            "total_evidence": total_evidence,
            "unique_sources": len(all_sources),
            "verdict_distribution": verdict_counts,
            "supported_claims": supported,
            "contested_claims": contested,
            "refuted_claims": refuted,
            "no_evidence_claims": no_evidence,
            "average_projected_probability": round(avg_proj_prob, 4),
            "verification_method": "subjective_logic_josang_2016",
            "math_library": "jsonld-ex",
        },
        verification_metadata={
            "tool": "trustandverify",
            "tool_version": "0.1.0",
            "scoring_method": "cumulative_fuse + trust_discount",
            "fusion_type": "subjective_logic_cumulative",
            "provenance_format": "JSON-LD + PROV-O",
            "source_urls": list(all_sources)[:20],  # Cap at 20 for size
        },
    )


def save_hypercert(
    hypercert: HypercertClaim,
    output_path: str = "data/hypercert.json",
) -> str:
    """Save a hypercert to a JSON file.

    Returns the content hash for verification.
    """
    from pathlib import Path

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(hypercert.to_json(), encoding="utf-8")
    return hypercert.content_hash


def batch_reports_to_hypercerts(
    reports: list[dict[str, Any]],
    contributor: str = "trustandverify-agent",
) -> list[HypercertClaim]:
    """Convert multiple reports into hypercerts for batch attestation."""
    return [report_to_hypercert(r, contributor) for r in reports]
