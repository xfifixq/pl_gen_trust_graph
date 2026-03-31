# 🔍 TrustAndVerify × PL Genesis

**Verifiable AI Knowledge Verification with Autonomous ML Credibility Scoring**

> Agentic knowledge verification using Subjective Logic confidence algebra, Impulse AI autonomous ML, and Hypercert impact attestations.

[![PL Genesis](https://img.shields.io/badge/PL%20Genesis-Frontiers%20of%20Collaboration-blue)](https://www.plgenesis.com/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![trustandverify](https://img.shields.io/pypi/v/trustandverify?label=trustandverify)](https://pypi.org/project/trustandverify/)
[![jsonld-ex](https://img.shields.io/pypi/v/jsonld-ex?label=jsonld-ex)](https://pypi.org/project/jsonld-ex/)

---

## 🎯 The Problem

AI systems hallucinate. Search engines rank by SEO, not truth. Misinformation spreads faster than corrections. There's no way to **mathematically verify** whether a piece of knowledge is supported, contested, or refuted — with full provenance and confidence scores.

## 💡 The Solution

**TrustAndVerify** decomposes any question into verifiable claims, gathers evidence from multiple independent sources, and scores confidence using **Subjective Logic** — a formal mathematical framework (Jøsang 2016) that explicitly represents belief, disbelief, and uncertainty as a 3-tuple opinion.

For PL Genesis, we've added two powerful integrations:

### 🤖 Impulse AI — Autonomous ML Credibility Scoring
Train a tabular AutoML model on verification features to create a **fast-path credibility pre-screener**. Upload a CSV of past verification data → Impulse trains a production model in minutes → deploy as an API for instant verdict predictions. This complements the rigorous Subjective Logic scoring by providing rapid triage.

### 📜 Hypercerts — Impact Attestation
Each verification report generates a **Hypercert** — a verifiable impact claim recording who verified what, when, with what evidence, and at what confidence level. This enables decentralized tracking and retroactive funding of knowledge verification work.

---

## 🏗️ Architecture

```
                    ┌─────────────────────┐
                    │   User Query        │
                    │ "Is X true?"        │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │  Claim Decomposition │  ← LLM (Gemini/OpenAI)
                    │  Query → N claims   │
                    └─────────┬───────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
    ┌─────────▼──────┐ ┌─────▼──────┐ ┌──────▼─────────┐
    │ Tavily Search  │ │ Brave      │ │ SerpAPI        │
    │ (evidence)     │ │ (evidence) │ │ (evidence)     │
    └─────────┬──────┘ └─────┬──────┘ └──────┬─────────┘
              │               │               │
              └───────────────┼───────────────┘
                              │
                    ┌─────────▼───────────┐
                    │  Evidence Scoring    │
                    │  • scalar_to_opinion │  ← jsonld-ex
                    │  • trust_discount    │    (Subjective Logic)
                    │  • cumulative_fuse   │
                    └─────────┬───────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
    ┌─────────▼──────┐ ┌─────▼──────┐ ┌──────▼─────────┐
    │ Impulse AI     │ │ Verdict    │ │ Hypercert      │
    │ Pre-screening  │ │ + Report   │ │ Attestation    │
    │ (fast path ML) │ │ (full SL)  │ │ (impact claim) │
    └────────────────┘ └────────────┘ └────────────────┘
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repo
git clone https://github.com/jemsbhai/plgenesis-trustandverify.git
cd plgenesis-trustandverify

# Install dependencies
pip install -e ".[dev]"
pip install trustandverify[tavily,gemini]

# Set up environment
cp .env.example .env
# Edit .env with your API keys
```

### CLI Usage

```bash
# Run a verification
plgenesis-tv verify "Is nuclear energy safer than solar per TWh?"

# With Impulse AI pre-screening
plgenesis-tv verify "Is coffee healthy?" --impulse

# Generate training dataset for Impulse AI
plgenesis-tv generate-dataset --samples 200

# Generate Hypercert from a report
plgenesis-tv hypercert report.json --output hypercert.json

# Show Impulse AI training instructions
plgenesis-tv train-info
```

### Streamlit Demo

```bash
streamlit run src/plgenesis_tv/app.py
```

---

## 🤖 Impulse AI Integration

### How It Works

1. **Generate Features**: TrustAndVerify extracts 20+ structured features from each verified claim (evidence count, source trust, confidence spread, Subjective Logic opinions, etc.)

2. **Train Model**: Upload the feature CSV to Impulse AI's dashboard → select `verdict` as target → AutoML trains a classifier in minutes

3. **Deploy & Predict**: The trained model deploys as an API endpoint → query it for instant credibility predictions on new claims

### Feature Schema

| Feature | Description |
|---|---|
| `evidence_count` | Number of evidence pieces gathered |
| `supporting_count` | Evidence supporting the claim |
| `contradicting_count` | Evidence contradicting the claim |
| `support_ratio` | Proportion of supporting evidence |
| `avg_source_trust` | Average trust score across sources |
| `avg_confidence` | Average raw confidence of evidence |
| `unique_domains` | Number of unique source domains |
| `belief` | Subjective Logic belief component |
| `disbelief` | Subjective Logic disbelief component |
| `uncertainty` | Subjective Logic uncertainty component |
| `projected_probability` | P = belief + base_rate × uncertainty |
| **`verdict`** | **Target: supported/contested/refuted/no_evidence** |

### Training Flow

```bash
# 1. Generate synthetic bootstrap data
plgenesis-tv generate-dataset --samples 500

# 2. Upload to Impulse AI dashboard
# → app.impulselabs.ai → Datasets → Upload CSV

# 3. Train model (target: 'verdict')
# → Click "Train Model" → ~2-5 min

# 4. Set deployment ID
echo "IMPULSE_DEPLOYMENT_ID=your-id" >> .env

# 5. Use in verification
plgenesis-tv verify "Is X true?" --impulse
```

---

## 📜 Hypercerts Integration

Each verification report produces a Hypercert-compatible impact claim:

```json
{
  "name": "Knowledge Verification: Is nuclear energy safer than solar?",
  "description": "Agentic verification using Subjective Logic (Jøsang 2016)...",
  "work_scope": ["knowledge-verification", "claim-decomposition", ...],
  "impact_scope": ["verifiable-ai", "misinformation-detection", ...],
  "properties": {
    "total_claims": 4,
    "total_evidence": 12,
    "unique_sources": 8,
    "verdict_distribution": {"supported": 2, "contested": 1, "refuted": 1},
    "average_projected_probability": 0.682,
    "verification_method": "subjective_logic_josang_2016"
  }
}
```

This enables:
- **Agentic impact evaluation** — AI agents that analyze and evaluate verification quality
- **Retroactive funding** — Fund verification work based on demonstrated impact
- **Data integration** — Connect verification provenance to the Hypercerts ecosystem

---

## 🧮 The Math: Subjective Logic

Unlike simple true/false scoring, we use **Subjective Logic** (Jøsang 2016), which represents belief as a 3-tuple opinion:

**ω = (b, d, u, a)** where:
- **b** = belief (evidence for)
- **d** = disbelief (evidence against)  
- **u** = uncertainty (lack of evidence)
- **a** = base rate (prior probability)
- **b + d + u = 1** (additive constraint)

**Projected probability**: P = b + a·u

This means:
- A claim with *no evidence* has high uncertainty, not low probability
- Multiple sources are fused using **cumulative fusion** (more evidence → less uncertainty)
- Source reliability is handled via **trust discounting** (untrusted sources contribute less)
- Conflicting evidence is explicitly modeled, not averaged away

The math is implemented in [jsonld-ex](https://pypi.org/project/jsonld-ex/), our Subjective Logic library.

---

## 🏆 Hackathon Challenges

| Challenge | Track | Why It Fits |
|---|---|---|
| **Fresh Code** | Protocol Labs | New project started during hacking period |
| **AI & Robotics** | Verifiable AI | Decision provenance, audit trails, formal verification |
| **Impulse AI** | Autonomous ML | Tabular AutoML credibility predictor from verification data |
| **Hypercerts** | Impact Evaluation | Agentic impact attestation from verification reports |
| **Community Vote** | Engagement | Tweet thread about the project |

---

## 📦 Dependencies

| Package | Role |
|---|---|
| [trustandverify](https://pypi.org/project/trustandverify/) | Core verification engine |
| [jsonld-ex](https://pypi.org/project/jsonld-ex/) | Subjective Logic confidence algebra |
| [Impulse AI SDK](https://docs.impulselabs.ai/) | Autonomous ML training & inference |
| [Streamlit](https://streamlit.io/) | Demo web interface |
| [Typer](https://typer.tiangolo.com/) | CLI framework |

---

## 🧪 Testing

```bash
pytest tests/ -v
```

---

## 📄 License

MIT — see [LICENSE](LICENSE).

## 👤 Author

**Muntaser Syed** — [@jemsbhai](https://github.com/jemsbhai)

Built for [PL Genesis: Frontiers of Collaboration](https://www.plgenesis.com/) by [Protocol Labs](https://protocol.ai).
