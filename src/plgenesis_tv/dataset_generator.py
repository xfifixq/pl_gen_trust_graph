"""Generate tabular datasets from trustandverify reports for Impulse AI training.

Extracts structured features from verification runs (evidence counts, trust scores,
sentiment alignment, source diversity, etc.) with the claim verdict as the target
column. The resulting CSV can be uploaded directly to Impulse AI for AutoML training.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


@dataclass
class ClaimFeatures:
    """Tabular features extracted from a single claim's verification data."""

    # Claim text features
    claim_word_count: int = 0
    claim_char_count: int = 0

    # Evidence statistics
    evidence_count: int = 0
    supporting_count: int = 0
    contradicting_count: int = 0
    support_ratio: float = 0.0

    # Confidence statistics
    avg_relevance: float = 0.0
    avg_confidence: float = 0.0
    max_confidence: float = 0.0
    min_confidence: float = 1.0
    confidence_spread: float = 0.0

    # Source statistics
    source_count: int = 0
    unique_domains: int = 0
    avg_source_trust: float = 0.0
    max_source_trust: float = 0.0
    min_source_trust: float = 1.0

    # Subjective Logic opinion (from jsonld-ex fusion)
    belief: float = 0.0
    disbelief: float = 0.0
    uncertainty: float = 1.0
    base_rate: float = 0.5
    projected_probability: float = 0.5

    # Target
    verdict: str = "no_evidence"


def _extract_domain(url: str) -> str:
    """Extract the domain from a URL."""
    try:
        parsed = urlparse(url)
        return parsed.netloc or "unknown"
    except Exception:
        return "unknown"


def extract_claim_features(claim: dict[str, Any]) -> ClaimFeatures:
    """Extract tabular features from a claim dictionary.

    Works with trustandverify Claim objects (as dicts) from report JSON output.
    """
    feats = ClaimFeatures()

    # Claim text features
    text = claim.get("text", "")
    feats.claim_word_count = len(text.split())
    feats.claim_char_count = len(text)

    # Evidence
    evidence_list = claim.get("evidence", [])
    feats.evidence_count = len(evidence_list)

    if evidence_list:
        supporting = [e for e in evidence_list if e.get("supports_claim", False)]
        contradicting = [e for e in evidence_list if not e.get("supports_claim", True)]
        feats.supporting_count = len(supporting)
        feats.contradicting_count = len(contradicting)
        feats.support_ratio = (
            feats.supporting_count / feats.evidence_count if feats.evidence_count > 0 else 0.0
        )

        # Confidence stats
        confidences = [e.get("confidence_raw", 0.0) for e in evidence_list]
        relevances = [e.get("relevance", 0.0) for e in evidence_list]

        if confidences:
            feats.avg_confidence = sum(confidences) / len(confidences)
            feats.max_confidence = max(confidences)
            feats.min_confidence = min(confidences)
            feats.confidence_spread = feats.max_confidence - feats.min_confidence

        if relevances:
            feats.avg_relevance = sum(relevances) / len(relevances)

        # Source stats
        sources = [e.get("source", {}) for e in evidence_list if e.get("source")]
        feats.source_count = len(sources)

        domains = set()
        trust_scores = []
        for src in sources:
            url = src.get("url", "")
            if url:
                domains.add(_extract_domain(url))
            trust = src.get("trust_score", 0.5)
            trust_scores.append(trust)

        feats.unique_domains = len(domains)
        if trust_scores:
            feats.avg_source_trust = sum(trust_scores) / len(trust_scores)
            feats.max_source_trust = max(trust_scores)
            feats.min_source_trust = min(trust_scores)

    # Subjective Logic opinion
    opinion = claim.get("opinion")
    if opinion:
        feats.belief = opinion.get("belief", 0.0)
        feats.disbelief = opinion.get("disbelief", 0.0)
        feats.uncertainty = opinion.get("uncertainty", 1.0)
        feats.base_rate = opinion.get("base_rate", 0.5)
        feats.projected_probability = feats.belief + feats.base_rate * feats.uncertainty

    # Verdict (target column)
    feats.verdict = claim.get("verdict", "no_evidence")

    return feats


def report_to_rows(report: dict[str, Any]) -> list[ClaimFeatures]:
    """Convert a full trustandverify report to a list of feature rows."""
    rows = []
    for claim in report.get("claims", []):
        rows.append(extract_claim_features(claim))
    return rows


def generate_csv(
    reports: list[dict[str, Any]],
    output_path: str | Path = "data/verification_features.csv",
) -> Path:
    """Generate a CSV dataset from multiple trustandverify reports.

    Args:
        reports: List of report dictionaries (from JSON-LD export or agent output).
        output_path: Path to write the CSV file.

    Returns:
        Path to the generated CSV file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[ClaimFeatures] = []
    for report in reports:
        all_rows.extend(report_to_rows(report))

    if not all_rows:
        raise ValueError("No claims found in the provided reports.")

    fieldnames = list(ClaimFeatures.__dataclass_fields__.keys())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(asdict(row))

    return output_path


def generate_synthetic_dataset(
    n_samples: int = 200,
    output_path: str | Path = "data/sample_verification_data.csv",
) -> Path:
    """Generate a synthetic training dataset for bootstrapping the Impulse AI model.

    Creates realistic-looking verification data with known patterns:
    - High evidence + high trust + high support → supported
    - Mixed evidence + moderate trust → contested
    - High evidence + low support → refuted
    - Low/no evidence → no_evidence
    """
    import random

    random.seed(42)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    fieldnames = list(ClaimFeatures.__dataclass_fields__.keys())

    for i in range(n_samples):
        verdict_roll = random.random()

        if verdict_roll < 0.35:  # supported
            ev_count = random.randint(3, 8)
            sup_count = random.randint(int(ev_count * 0.7), ev_count)
            con_count = ev_count - sup_count
            avg_trust = random.uniform(0.6, 0.95)
            avg_conf = random.uniform(0.65, 0.95)
            belief = random.uniform(0.55, 0.9)
            disbelief = random.uniform(0.02, 0.15)
            verdict = "supported"

        elif verdict_roll < 0.60:  # contested
            ev_count = random.randint(3, 7)
            sup_count = random.randint(int(ev_count * 0.3), int(ev_count * 0.65))
            con_count = ev_count - sup_count
            avg_trust = random.uniform(0.4, 0.7)
            avg_conf = random.uniform(0.4, 0.7)
            belief = random.uniform(0.25, 0.55)
            disbelief = random.uniform(0.2, 0.45)
            verdict = "contested"

        elif verdict_roll < 0.80:  # refuted
            ev_count = random.randint(2, 6)
            sup_count = random.randint(0, int(ev_count * 0.3))
            con_count = ev_count - sup_count
            avg_trust = random.uniform(0.5, 0.85)
            avg_conf = random.uniform(0.5, 0.85)
            belief = random.uniform(0.02, 0.2)
            disbelief = random.uniform(0.55, 0.9)
            verdict = "refuted"

        else:  # no_evidence
            ev_count = random.randint(0, 1)
            sup_count = 0
            con_count = 0
            avg_trust = random.uniform(0.1, 0.4)
            avg_conf = random.uniform(0.0, 0.3)
            belief = random.uniform(0.0, 0.1)
            disbelief = random.uniform(0.0, 0.1)
            verdict = "no_evidence"

        uncertainty = max(0.0, 1.0 - belief - disbelief)
        base_rate = 0.5
        proj_prob = belief + base_rate * uncertainty

        sup_ratio = sup_count / ev_count if ev_count > 0 else 0.0
        unique_dom = min(ev_count, random.randint(1, max(1, ev_count)))
        max_conf = min(1.0, avg_conf + random.uniform(0.05, 0.2))
        min_conf = max(0.0, avg_conf - random.uniform(0.05, 0.2))
        max_trust = min(1.0, avg_trust + random.uniform(0.03, 0.15))
        min_trust = max(0.0, avg_trust - random.uniform(0.03, 0.15))

        word_count = random.randint(5, 25)

        rows.append({
            "claim_word_count": word_count,
            "claim_char_count": word_count * random.randint(4, 7),
            "evidence_count": ev_count,
            "supporting_count": sup_count,
            "contradicting_count": con_count,
            "support_ratio": round(sup_ratio, 4),
            "avg_relevance": round(random.uniform(0.3, 0.95), 4),
            "avg_confidence": round(avg_conf, 4),
            "max_confidence": round(max_conf, 4),
            "min_confidence": round(min_conf, 4),
            "confidence_spread": round(max_conf - min_conf, 4),
            "source_count": ev_count,
            "unique_domains": unique_dom,
            "avg_source_trust": round(avg_trust, 4),
            "max_source_trust": round(max_trust, 4),
            "min_source_trust": round(min_trust, 4),
            "belief": round(belief, 4),
            "disbelief": round(disbelief, 4),
            "uncertainty": round(uncertainty, 4),
            "base_rate": base_rate,
            "projected_probability": round(proj_prob, 4),
            "verdict": verdict,
        })

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path
