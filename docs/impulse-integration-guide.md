# Impulse AI Integration Guide

## Overview

Impulse AI provides **Autonomous ML Engineering** — upload a CSV, describe your target,
and get a production-ready model deployed as an API in minutes.

We use it to train a **credibility pre-screener** that predicts claim verdicts from
tabular features extracted from verification runs.

## Step-by-Step Setup

### 1. Generate Training Data

```bash
# Synthetic bootstrap (for demo/initial training)
plgenesis-tv generate-dataset --samples 500 --output data/training.csv

# From real verification runs (preferred for production)
plgenesis-tv generate-dataset --reports-dir data/reports/ --output data/training.csv
```

### 2. Upload to Impulse AI

1. Go to [app.impulselabs.ai](https://app.impulselabs.ai)
2. Navigate to **Datasets**
3. Upload your CSV file
4. Verify the columns look correct (21 features + 1 target)

### 3. Train the Model

1. Click **Train Model**
2. Select `verdict` as the **target column**
3. Choose training parameters (defaults work well):
   - Epochs: 1-3
   - Batch size: 2-4
4. Click **Start Training**
5. Wait ~2-5 minutes

### 4. Deploy and Configure

1. Once training completes, note the **Deployment ID** on the model card
2. Create an API key: **Settings → API Keys → New API Key**
3. Add to your `.env`:

```bash
IMPULSE_API_KEY=imp_your_key_here
IMPULSE_DEPLOYMENT_ID=your-deployment-id
```

### 5. Use in Verification

```bash
# CLI with Impulse pre-screening
plgenesis-tv verify "Is coffee healthy?" --impulse

# Or in the Streamlit app — check "Enable Impulse AI" in sidebar
streamlit run src/plgenesis_tv/app.py
```

## Feature Schema

The CSV has 22 columns (21 features + 1 target):

| Column | Type | Description |
|---|---|---|
| claim_word_count | int | Words in the claim text |
| claim_char_count | int | Characters in the claim text |
| evidence_count | int | Total evidence pieces gathered |
| supporting_count | int | Evidence supporting the claim |
| contradicting_count | int | Evidence contradicting the claim |
| support_ratio | float | supporting / total evidence |
| avg_relevance | float | Mean relevance score of evidence |
| avg_confidence | float | Mean raw confidence of evidence |
| max_confidence | float | Highest confidence score |
| min_confidence | float | Lowest confidence score |
| confidence_spread | float | max - min confidence |
| source_count | int | Total sources consulted |
| unique_domains | int | Distinct domain names |
| avg_source_trust | float | Mean trust score across sources |
| max_source_trust | float | Highest source trust |
| min_source_trust | float | Lowest source trust |
| belief | float | Subjective Logic belief (fused) |
| disbelief | float | Subjective Logic disbelief (fused) |
| uncertainty | float | Subjective Logic uncertainty (fused) |
| base_rate | float | Prior probability (default 0.5) |
| projected_probability | float | P = belief + base_rate × uncertainty |
| **verdict** | **str** | **Target: supported / contested / refuted / no_evidence** |

## API Usage (Programmatic)

```python
from plgenesis_tv.impulse_integration import ImpulseCredibilityScorer
from plgenesis_tv.dataset_generator import extract_claim_features

async with ImpulseCredibilityScorer() as scorer:
    # Check model is active
    status = await scorer.check_status()
    print(f"Model status: {status}")

    # Predict from features
    features = extract_claim_features(claim_dict)
    result = await scorer.predict_claim(asdict(features))
    print(f"Predicted: {result['impulse_verdict']} ({result['impulse_confidence']})")
```
