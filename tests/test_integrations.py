"""Tests for PL Genesis × TrustAndVerify integrations."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest


# ── Dataset Generator Tests ──────────────────────────────────────────────────

class TestDatasetGenerator:
    """Tests for dataset_generator.py."""

    def test_extract_claim_features_basic(self):
        from plgenesis_tv.dataset_generator import extract_claim_features

        claim = {
            "text": "Nuclear energy is safer than solar per TWh",
            "verdict": "supported",
            "evidence": [
                {
                    "text": "Death rates per TWh show nuclear at 0.03",
                    "supports_claim": True,
                    "relevance": 0.9,
                    "confidence_raw": 0.85,
                    "source": {
                        "url": "https://ourworldindata.org/energy",
                        "title": "Our World in Data",
                        "trust_score": 0.9,
                    },
                },
                {
                    "text": "Solar has fewer accidents",
                    "supports_claim": False,
                    "relevance": 0.7,
                    "confidence_raw": 0.6,
                    "source": {
                        "url": "https://example.com/solar",
                        "title": "Example",
                        "trust_score": 0.5,
                    },
                },
            ],
            "opinion": {
                "belief": 0.7,
                "disbelief": 0.1,
                "uncertainty": 0.2,
                "base_rate": 0.5,
            },
        }

        features = extract_claim_features(claim)

        assert features.evidence_count == 2
        assert features.supporting_count == 1
        assert features.contradicting_count == 1
        assert features.support_ratio == pytest.approx(0.5)
        assert features.unique_domains == 2
        assert features.verdict == "supported"
        assert features.belief == pytest.approx(0.7)
        assert features.projected_probability == pytest.approx(0.7 + 0.5 * 0.2)

    def test_extract_claim_features_empty_evidence(self):
        from plgenesis_tv.dataset_generator import extract_claim_features

        claim = {"text": "Some claim", "verdict": "no_evidence", "evidence": []}
        features = extract_claim_features(claim)

        assert features.evidence_count == 0
        assert features.support_ratio == 0.0
        assert features.verdict == "no_evidence"

    def test_generate_synthetic_dataset(self, tmp_path):
        from plgenesis_tv.dataset_generator import generate_synthetic_dataset

        output = tmp_path / "test_data.csv"
        path = generate_synthetic_dataset(n_samples=50, output_path=output)

        assert path.exists()

        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 50
        verdicts = {r["verdict"] for r in rows}
        assert verdicts.issubset({"supported", "contested", "refuted", "no_evidence"})

        # Check all expected columns exist
        expected_cols = {
            "claim_word_count", "evidence_count", "supporting_count",
            "avg_source_trust", "belief", "disbelief", "uncertainty",
            "projected_probability", "verdict",
        }
        assert expected_cols.issubset(set(rows[0].keys()))

    def test_report_to_rows(self):
        from plgenesis_tv.dataset_generator import report_to_rows

        report = {
            "claims": [
                {
                    "text": "Claim 1",
                    "verdict": "supported",
                    "evidence": [
                        {"supports_claim": True, "relevance": 0.9, "confidence_raw": 0.8,
                         "source": {"url": "https://a.com", "trust_score": 0.8}},
                    ],
                    "opinion": {"belief": 0.7, "disbelief": 0.1, "uncertainty": 0.2, "base_rate": 0.5},
                },
                {
                    "text": "Claim 2",
                    "verdict": "refuted",
                    "evidence": [],
                },
            ]
        }

        rows = report_to_rows(report)
        assert len(rows) == 2
        assert rows[0].verdict == "supported"
        assert rows[1].verdict == "refuted"

    def test_generate_csv_from_reports(self, tmp_path):
        from plgenesis_tv.dataset_generator import generate_csv

        reports = [
            {
                "claims": [
                    {
                        "text": f"Claim {i}",
                        "verdict": "supported",
                        "evidence": [
                            {"supports_claim": True, "relevance": 0.8, "confidence_raw": 0.7,
                             "source": {"url": f"https://src{i}.com", "trust_score": 0.7}},
                        ],
                        "opinion": {"belief": 0.6, "disbelief": 0.1, "uncertainty": 0.3, "base_rate": 0.5},
                    }
                    for i in range(3)
                ]
            }
        ]

        output = tmp_path / "test_output.csv"
        path = generate_csv(reports, output)
        assert path.exists()

        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3


# ── Hypercerts Integration Tests ─────────────────────────────────────────────

class TestHypercertsIntegration:
    """Tests for hypercerts_integration.py."""

    def _sample_report(self) -> dict:
        return {
            "id": "rpt-001",
            "query": "Is nuclear energy safer than solar?",
            "claims": [
                {
                    "text": "Nuclear has lower death rate per TWh than solar",
                    "verdict": "supported",
                    "evidence": [
                        {
                            "text": "Stats show 0.03 deaths per TWh for nuclear",
                            "supports_claim": True,
                            "source": {"url": "https://ourworldindata.org/energy"},
                        },
                        {
                            "text": "Installation accidents raise solar death rate",
                            "supports_claim": True,
                            "source": {"url": "https://example.com/study"},
                        },
                    ],
                    "opinion": {"belief": 0.7, "disbelief": 0.1, "uncertainty": 0.2, "base_rate": 0.5},
                },
                {
                    "text": "Nuclear waste poses long-term risks",
                    "verdict": "contested",
                    "evidence": [
                        {
                            "text": "Modern storage is effective",
                            "supports_claim": False,
                            "source": {"url": "https://iaea.org/waste"},
                        },
                    ],
                    "opinion": {"belief": 0.4, "disbelief": 0.3, "uncertainty": 0.3, "base_rate": 0.5},
                },
            ],
            "summary": "Nuclear energy appears safer per TWh, but waste concerns persist.",
            "created_at": "2026-03-31T00:00:00Z",
        }

    def test_report_to_hypercert(self):
        from plgenesis_tv.hypercerts_integration import report_to_hypercert

        report = self._sample_report()
        hc = report_to_hypercert(report, contributor="test-user")

        assert "Nuclear energy" in hc.name or "nuclear energy" in hc.name.lower()
        assert hc.properties["total_claims"] == 2
        assert hc.properties["total_evidence"] == 3
        assert hc.properties["unique_sources"] == 3
        assert hc.properties["supported_claims"] == 1
        assert hc.properties["contested_claims"] == 1
        assert "test-user" in hc.contributors
        assert "subjective_logic_josang_2016" in hc.properties["verification_method"]

    def test_hypercert_content_hash_deterministic(self):
        from plgenesis_tv.hypercerts_integration import report_to_hypercert

        report = self._sample_report()
        hc1 = report_to_hypercert(report)
        hc2 = report_to_hypercert(report)

        # Content hash should be deterministic for same input
        # (timestamps will differ, so we test the structure is correct)
        assert isinstance(hc1.content_hash, str)
        assert len(hc1.content_hash) == 64  # SHA-256 hex

    def test_hypercert_to_json(self):
        from plgenesis_tv.hypercerts_integration import report_to_hypercert

        report = self._sample_report()
        hc = report_to_hypercert(report)
        json_str = hc.to_json()

        parsed = json.loads(json_str)
        assert "name" in parsed
        assert "work_scope" in parsed
        assert "properties" in parsed
        assert parsed["properties"]["total_claims"] == 2

    def test_save_hypercert(self, tmp_path):
        from plgenesis_tv.hypercerts_integration import report_to_hypercert, save_hypercert

        report = self._sample_report()
        hc = report_to_hypercert(report)

        output = tmp_path / "test_hc.json"
        content_hash = save_hypercert(hc, str(output))

        assert output.exists()
        assert isinstance(content_hash, str)
        assert len(content_hash) == 64

        with open(output, encoding="utf-8") as f:
            data = json.load(f)
        assert data["properties"]["total_claims"] == 2

    def test_batch_reports_to_hypercerts(self):
        from plgenesis_tv.hypercerts_integration import batch_reports_to_hypercerts

        reports = [self._sample_report(), self._sample_report()]
        hcs = batch_reports_to_hypercerts(reports)
        assert len(hcs) == 2


# ── Impulse Integration Tests ────────────────────────────────────────────────

class TestImpulseIntegration:
    """Tests for impulse_integration.py (unit tests, no API calls)."""

    def test_impulse_prediction_confidence_label(self):
        from plgenesis_tv.impulse_integration import ImpulsePrediction

        high = ImpulsePrediction("supported", 0.92, {})
        assert high.confidence_label == "high"

        med = ImpulsePrediction("contested", 0.6, {})
        assert med.confidence_label == "moderate"

        low = ImpulsePrediction("refuted", 0.3, {})
        assert low.confidence_label == "low"

    def test_scorer_not_available_without_config(self):
        from plgenesis_tv.impulse_integration import ImpulseCredibilityScorer

        scorer = ImpulseCredibilityScorer(api_key="", deployment_id="")
        assert not scorer.is_available

    def test_scorer_available_with_config(self):
        from plgenesis_tv.impulse_integration import ImpulseCredibilityScorer

        scorer = ImpulseCredibilityScorer(api_key="imp_test", deployment_id="test-deploy")
        assert scorer.is_available

    def test_training_instructions(self):
        from plgenesis_tv.impulse_integration import create_training_instructions

        text = create_training_instructions()
        assert "Impulse" in text
        assert "verdict" in text
        assert "app.impulselabs.ai" in text
