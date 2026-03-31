# Architecture: TrustAndVerify × PL Genesis

## System Overview

TrustAndVerify is a **verifiable AI** system that provides mathematical guarantees about
knowledge claims using Subjective Logic confidence algebra (Jøsang 2016).

### Core Pipeline

```
Query → Decompose → Search → Extract → Score → Fuse → Report
```

1. **Decompose**: LLM breaks query into N independent, verifiable claims
2. **Search**: Multi-backend search (Tavily, Brave, SerpAPI) gathers evidence
3. **Extract**: LLM extracts supporting/contradicting evidence from search results
4. **Score**: Each evidence piece → Subjective Logic opinion via `scalar_to_opinion`
5. **Trust Discount**: Source reliability modulates evidence strength via `trust_discount`
6. **Fuse**: All opinions per claim → single fused opinion via `cumulative_fuse`
7. **Verdict**: Fused opinion → supported/contested/refuted/no_evidence

### PL Genesis Additions

#### Impulse AI Integration (Autonomous ML)

The Impulse AI integration adds a **fast-path credibility pre-screener**:

```
Verification Run → Feature Extraction → CSV → Impulse AI Training → Deployed Model
                                                                         ↓
New Claim Features ──────────────────────────────────────────── → Instant Prediction
```

**Feature extraction** converts each claim's verification data into 21 tabular features:
- Evidence statistics (count, support ratio, confidence spread)
- Source statistics (domain diversity, average trust)
- Subjective Logic opinion components (belief, disbelief, uncertainty)

**Training** uses Impulse AI's AutoML to build a classifier predicting the `verdict` column.
The trained model deploys as a REST API for sub-second inference.

**Use case**: Before running the full verification pipeline (which requires LLM calls and
web searches), query the Impulse model for a quick triage estimate.

#### Hypercerts Integration (Impact Attestation)

Each verification report generates a **Hypercert** — a standardized impact claim:

```
Verification Report → report_to_hypercert() → HypercertClaim → JSON / On-chain
```

The hypercert records:
- **Work scope**: What was verified (query, claims, methods used)
- **Impact scope**: What effect it had (misinformation detected, decisions supported)
- **Properties**: Quantitative metrics (evidence count, source diversity, confidence)
- **Verification metadata**: Tool versions, math methods, source URLs
- **Content hash**: SHA-256 for integrity verification

This enables the Hypercerts ecosystem to **discover, evaluate, and fund** verification work.

## Subjective Logic Primer

Traditional binary classification: P(true) = 0.8

Subjective Logic opinion: ω = (b=0.6, d=0.1, u=0.3, a=0.5)

The key difference is **explicit uncertainty**. When evidence is scarce, uncertainty is high
— the opinion doesn't pretend to know more than it does.

### Key Operations

| Operation | Formula | Purpose |
|---|---|---|
| Projected probability | P = b + a·u | Point estimate from opinion |
| Cumulative fusion | ω₁ ⊕ ω₂ | Combine independent evidence |
| Trust discount | ω_A:B ⊗ ω_B:X | Adjust for source reliability |
| Conflict detection | 1 - (b₁b₂ + d₁d₂ + ... ) | Measure evidence disagreement |

### Why This Matters for Verifiable AI

1. **No false confidence**: Low evidence → high uncertainty, not low probability
2. **Source diversity**: More independent sources → lower uncertainty via fusion
3. **Trust modeling**: Unreliable sources automatically contribute less
4. **Conflict transparency**: Disagreeing evidence is flagged, not averaged away
5. **Mathematical provenance**: Every score traces back to formal operations

## Technology Stack

| Layer | Technology | Role |
|---|---|---|
| Confidence algebra | jsonld-ex (PyPI) | Subjective Logic operations |
| Verification engine | trustandverify (PyPI) | Full pipeline orchestration |
| ML credibility | Impulse AI | AutoML training & inference |
| Impact attestation | Hypercerts Protocol | Decentralized impact tracking |
| Search | Tavily, Brave, SerpAPI | Multi-source evidence |
| LLM | Gemini, OpenAI, Anthropic | Claim decomposition & extraction |
| UI | Streamlit | Interactive demo |
| CLI | Typer + Rich | Command-line interface |
