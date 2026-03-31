"""Impulse AI integration for autonomous ML credibility scoring.

Wraps the Impulse AI platform to:
1. Upload verification feature datasets for AutoML training
2. Query deployed models for fast credibility pre-screening
3. Compare Impulse predictions against Subjective Logic verdicts

The Impulse model acts as a "fast path" credibility predictor trained on
historical verification data, complementing the rigorous Subjective Logic
scoring in trustandverify.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

IMPULSE_BASE_URL = "https://inference.impulselabs.ai"

# Impulse AI returns integer class indices — map back to verdict labels.
# Alphabetical order matches pandas/sklearn default label encoding.
VERDICT_INDEX_MAP: dict[int, str] = {
    0: "contested",
    1: "no_evidence",
    2: "refuted",
    3: "supported",
}
VERDICT_LABELS = list(VERDICT_INDEX_MAP.values())


@dataclass
class ImpulsePrediction:
    """Result from Impulse AI inference."""

    predicted_verdict: str
    probability: float
    raw_response: dict[str, Any]

    @property
    def confidence_label(self) -> str:
        if self.probability >= 0.8:
            return "high"
        elif self.probability >= 0.5:
            return "moderate"
        else:
            return "low"


class ImpulseCredibilityScorer:
    """Wraps Impulse AI for fast credibility pre-screening of claims.

    After training a model on verification feature data via the Impulse dashboard,
    this class queries the deployed model to predict claim verdicts from tabular
    features — without running the full multi-source evidence pipeline.
    """

    def __init__(
        self,
        api_key: str | None = None,
        deployment_id: str | None = None,
    ):
        # Use `is not None` so passing "" explicitly works (doesn't fall through to env)
        self.api_key = api_key if api_key is not None else os.getenv("IMPULSE_API_KEY", "")
        self.deployment_id = (
            deployment_id if deployment_id is not None
            else os.getenv("IMPULSE_DEPLOYMENT_ID", "")
        )
        self._client = httpx.AsyncClient(
            base_url=IMPULSE_BASE_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )

    @property
    def is_available(self) -> bool:
        """Check if Impulse AI credentials are configured."""
        return bool(self.api_key and self.deployment_id)

    async def check_status(self) -> str:
        """Check if the deployed model is active."""
        if not self.is_available:
            return "NOT_CONFIGURED"
        try:
            resp = await self._client.get(
                "/deploy/status",
                params={"deployment_id": self.deployment_id},
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("status", "UNKNOWN")
        except Exception as e:
            return f"ERROR: {e}"

    async def predict(self, features: dict[str, Any]) -> ImpulsePrediction:
        """Predict claim verdict from tabular features.

        Args:
            features: Dict of feature values matching the training schema
                (evidence_count, avg_source_trust, belief, etc.)
                Exclude the 'verdict' target column.

        Returns:
            ImpulsePrediction with predicted verdict and probability.
        """
        if not self.is_available:
            raise RuntimeError(
                "Impulse AI not configured. Set IMPULSE_API_KEY and IMPULSE_DEPLOYMENT_ID."
            )

        # Remove target column if accidentally included
        input_features = {k: v for k, v in features.items() if k != "verdict"}

        payload = {
            "deployment_id": self.deployment_id,
            "inputs": input_features,
        }

        resp = await self._client.post("/infer", json=payload)
        resp.raise_for_status()
        data = resp.json()

        # Impulse multiclass API returns:
        #   prediction = binary flag (1 = classified)
        #   probability = class index as float (e.g. 3.0 = class 3)
        # For multiclass, the class index is in the "probability" field.
        raw_prediction = data.get("prediction", -1)
        raw_prob = float(data.get("probability", 0.0))

        # Determine which field holds the class index:
        # - If probability > 1, it's a class index (real probabilities are 0-1)
        # - Otherwise fall back to prediction as the class index
        if raw_prob > 1.0:
            # probability field contains the class index
            class_idx = int(raw_prob)
            confidence = 1.0  # Model is categorical, no soft probability available
        else:
            # Standard case: prediction is class index, probability is confidence
            class_idx = int(raw_prediction)
            confidence = raw_prob

        verdict_label = VERDICT_INDEX_MAP.get(class_idx, f"class_{class_idx}")

        return ImpulsePrediction(
            predicted_verdict=verdict_label,
            probability=confidence,
            raw_response=data,
        )

    async def predict_claim(self, claim_features: dict[str, Any]) -> dict[str, Any]:
        """Predict and return enriched result with comparison metadata.

        Returns a dict suitable for inclusion in verification reports.
        """
        try:
            prediction = await self.predict(claim_features)
            return {
                "impulse_verdict": prediction.predicted_verdict,
                "impulse_probability": prediction.probability,
                "impulse_confidence": prediction.confidence_label,
                "impulse_available": True,
            }
        except Exception as e:
            return {
                "impulse_verdict": None,
                "impulse_probability": None,
                "impulse_confidence": None,
                "impulse_available": False,
                "impulse_error": str(e),
            }

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()


def create_training_instructions() -> str:
    """Return human-readable instructions for training the Impulse model."""
    return """
╔══════════════════════════════════════════════════════════════╗
║          Impulse AI Training Instructions                    ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. Generate the dataset:                                    ║
║     plgenesis-tv generate-dataset                            ║
║                                                              ║
║  2. Go to https://app.impulselabs.ai                         ║
║                                                              ║
║  3. Upload the CSV from data/verification_features.csv       ║
║     (or data/sample_verification_data.csv for bootstrap)     ║
║                                                              ║
║  4. Click "Train Model" and select 'verdict' as target       ║
║                                                              ║
║  5. Wait for training to complete (~2-5 minutes)             ║
║                                                              ║
║  6. Copy the Deployment ID from the model card               ║
║                                                              ║
║  7. Set in your .env:                                        ║
║     IMPULSE_DEPLOYMENT_ID=your-deployment-id                 ║
║                                                              ║
║  8. Run verification with Impulse pre-screening:             ║
║     plgenesis-tv verify "Is X true?" --impulse               ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
