# Changelog

## [0.1.0] - 2026-03-31

### Added
- Initial project scaffold for PL Genesis: Frontiers of Collaboration hackathon
- **Dataset Generator**: Extract 21 tabular features from verification reports → CSV
  - `extract_claim_features()` — single claim feature extraction
  - `report_to_rows()` — full report to feature rows
  - `generate_csv()` — batch reports to CSV
  - `generate_synthetic_dataset()` — synthetic bootstrap data for training
- **Impulse AI Integration**: Autonomous ML credibility pre-screening
  - `ImpulseCredibilityScorer` — async client for Impulse AI inference API
  - `ImpulsePrediction` — typed prediction result with confidence labels
  - Model status checking and error handling
- **Hypercerts Integration**: Impact attestation from verification reports
  - `report_to_hypercert()` — convert report to Hypercert-compatible claim
  - `HypercertClaim` — dataclass following Hypercerts schema
  - SHA-256 content hashing for integrity verification
  - Batch conversion support
- **CLI** (Typer + Rich):
  - `verify` — run verification with optional Impulse pre-screening
  - `generate-dataset` — generate training CSV (synthetic or from reports)
  - `hypercert` — generate impact attestation from report JSON
  - `train-info` — display Impulse AI training instructions
- **Streamlit Demo App**: 4-tab interface
  - Verify tab with Subjective Logic opinion visualization
  - Impulse AI tab with dataset generation and download
  - Hypercerts tab with attestation generation
  - About tab with architecture and links
- **Documentation**: Architecture guide, Impulse integration guide
- **Tests**: 14 unit tests covering all integrations
- **Sample Data**: 200-row synthetic verification dataset
