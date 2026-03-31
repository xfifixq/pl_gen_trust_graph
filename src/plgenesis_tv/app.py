"""Streamlit demo app for PL Genesis × TrustAndVerify.

A unified interface showing:
1. Knowledge verification with Subjective Logic confidence algebra
2. Impulse AI autonomous ML credibility pre-screening
3. Hypercert impact attestation generation
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

# Page config
st.set_page_config(
    page_title="TrustAndVerify — Verifiable AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    st.title("🔍 TrustAndVerify")
    st.markdown(
        "**Verifiable AI Knowledge Verification** with Subjective Logic "
        "confidence algebra, Impulse AI credibility scoring, and Hypercert impact attestations."
    )

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        st.subheader("🔑 API Keys")
        tavily_key = st.text_input(
            "Tavily API Key",
            value=os.getenv("TAVILY_API_KEY", ""),
            type="password",
        )
        gemini_key = st.text_input(
            "Gemini API Key",
            value=os.getenv("GEMINI_API_KEY", ""),
            type="password",
        )

        st.divider()
        st.subheader("🤖 Impulse AI")
        impulse_key = st.text_input(
            "Impulse API Key",
            value=os.getenv("IMPULSE_API_KEY", ""),
            type="password",
        )
        impulse_deployment = st.text_input(
            "Deployment ID",
            value=os.getenv("IMPULSE_DEPLOYMENT_ID", ""),
        )
        impulse_enabled = st.checkbox(
            "Enable Impulse AI pre-screening",
            value=bool(impulse_key and impulse_deployment),
        )

        st.divider()
        st.subheader("📊 Settings")
        num_claims = st.slider("Max claims to decompose", 2, 8, 4)
        max_sources = st.slider("Max sources per claim", 2, 5, 3)

        st.divider()
        st.markdown("---")
        st.markdown(
            "Built for **PL Genesis: Frontiers of Collaboration** hackathon\n\n"
            "🏗️ [Protocol Labs](https://protocol.ai) · "
            "🤖 [Impulse AI](https://impulselabs.ai) · "
            "📜 [Hypercerts](https://hypercerts.org)"
        )

    # Main tabs
    tab_verify, tab_impulse, tab_hypercert, tab_about = st.tabs([
        "🔍 Verify", "🤖 Impulse AI", "📜 Hypercerts", "ℹ️ About"
    ])

    with tab_verify:
        _render_verify_tab(tavily_key, gemini_key, num_claims, max_sources, impulse_enabled, impulse_key, impulse_deployment)

    with tab_impulse:
        _render_impulse_tab()

    with tab_hypercert:
        _render_hypercert_tab()

    with tab_about:
        _render_about_tab()


def _render_verify_tab(tavily_key, gemini_key, num_claims, max_sources, impulse_enabled, impulse_key, impulse_deployment):
    """Main verification tab."""
    query = st.text_area(
        "Enter a question or claim to verify:",
        placeholder="Is nuclear energy safer than solar power per TWh?",
        height=80,
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        run_btn = st.button("🚀 Verify", type="primary", use_container_width=True)

    if run_btn and query:
        if not tavily_key or not gemini_key:
            st.error("Please provide Tavily and Gemini API keys in the sidebar.")
            return

        # Set env vars for trustandverify
        os.environ["TAVILY_API_KEY"] = tavily_key
        os.environ["GEMINI_API_KEY"] = gemini_key

        with st.spinner("🔄 Running verification pipeline..."):
            report = _run_verification(query, num_claims, max_sources)

        if report is None:
            st.error("Verification failed. Check your API keys and try again.")
            return

        # Store in session state
        st.session_state["last_report"] = report
        st.session_state["last_query"] = query

        # Display results
        _display_results(report)

        # Impulse AI screening
        if impulse_enabled and impulse_key and impulse_deployment:
            st.divider()
            st.subheader("🤖 Impulse AI Credibility Pre-Screening")
            os.environ["IMPULSE_API_KEY"] = impulse_key
            os.environ["IMPULSE_DEPLOYMENT_ID"] = impulse_deployment
            _run_impulse_screening_ui(report)

    elif "last_report" in st.session_state:
        # Show cached results
        st.info(f"Showing cached results for: *{st.session_state.get('last_query', '')}*")
        _display_results(st.session_state["last_report"])


def _run_verification(query: str, num_claims: int, max_sources: int):
    """Run the trustandverify pipeline."""
    try:
        from trustandverify import TrustAgent, TrustConfig

        config = TrustConfig(num_claims=num_claims, max_sources_per_claim=max_sources)
        agent = TrustAgent(config=config)
        report = asyncio.run(agent.verify(query))
        return report
    except ImportError:
        st.error(
            "trustandverify not installed. Run:\n"
            "```\npip install trustandverify[tavily,gemini]\n```"
        )
        return None
    except Exception as e:
        st.error(f"Verification error: {e}")
        return None


def _display_results(report):
    """Display verification results."""
    claims = report.claims if hasattr(report, "claims") else report.get("claims", [])

    # Summary
    summary = report.summary if hasattr(report, "summary") else report.get("summary", "")
    if summary:
        st.success(summary)

    # Claims table
    st.subheader(f"📋 Claims ({len(claims)})")

    for i, claim in enumerate(claims):
        text = claim.text if hasattr(claim, "text") else claim.get("text", "")
        verdict = claim.verdict.value if hasattr(claim, "verdict") and hasattr(claim.verdict, "value") else str(claim.get("verdict", "unknown"))
        evidence = claim.evidence if hasattr(claim, "evidence") else claim.get("evidence", [])
        opinion = claim.opinion if hasattr(claim, "opinion") else claim.get("opinion")

        # Verdict coloring
        verdict_colors = {
            "supported": "🟢", "contested": "🟡", "refuted": "🔴", "no_evidence": "⚪"
        }
        icon = verdict_colors.get(verdict, "❓")

        # Projected probability
        if opinion:
            if hasattr(opinion, "projected_probability"):
                p = opinion.projected_probability
            else:
                b = opinion.get("belief", 0)
                u = opinion.get("uncertainty", 1)
                br = opinion.get("base_rate", 0.5)
                p = b + br * u
        else:
            p = None

        with st.expander(f"{icon} **{verdict.upper()}** — {text[:80]}...", expanded=(i == 0)):
            col_v, col_p, col_e = st.columns(3)
            col_v.metric("Verdict", verdict)
            col_p.metric("P(true)", f"{p:.3f}" if p is not None else "N/A")
            col_e.metric("Evidence", len(evidence))

            if opinion:
                if hasattr(opinion, "belief"):
                    b, d, u = opinion.belief, opinion.disbelief, opinion.uncertainty
                else:
                    b = opinion.get("belief", 0)
                    d = opinion.get("disbelief", 0)
                    u = opinion.get("uncertainty", 1)

                st.markdown("**Subjective Logic Opinion:**")
                col_b, col_d, col_u = st.columns(3)
                col_b.progress(b, text=f"Belief: {b:.3f}")
                col_d.progress(d, text=f"Disbelief: {d:.3f}")
                col_u.progress(u, text=f"Uncertainty: {u:.3f}")

            if evidence:
                st.markdown("**Evidence:**")
                for ev in evidence:
                    supports = ev.supports_claim if hasattr(ev, "supports_claim") else ev.get("supports_claim", True)
                    ev_text = ev.text if hasattr(ev, "text") else ev.get("text", "")
                    direction = "✅ Supports" if supports else "❌ Contradicts"
                    st.markdown(f"- {direction}: {ev_text[:120]}...")


def _run_impulse_screening_ui(report):
    """Run Impulse AI screening and display results."""
    try:
        from plgenesis_tv.dataset_generator import extract_claim_features
        from plgenesis_tv.impulse_integration import ImpulseCredibilityScorer
        from plgenesis_tv.cli import _claim_to_dict

        async def _screen():
            async with ImpulseCredibilityScorer() as scorer:
                status = await scorer.check_status()
                if status != "ACTIVE":
                    st.warning(f"Impulse model status: {status}")
                    return []

                results = []
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
                    results.append((text, result))
                return results

        results = asyncio.run(_screen())

        for text, result in results:
            if result.get("impulse_available"):
                st.markdown(
                    f"- **{text[:60]}...** → "
                    f"Impulse: `{result['impulse_verdict']}` "
                    f"({result['impulse_confidence']} confidence, P={result['impulse_probability']:.3f})"
                )
            else:
                st.markdown(f"- **{text[:60]}...** → Impulse unavailable")

    except Exception as e:
        st.warning(f"Impulse AI screening error: {e}")


def _render_impulse_tab():
    """Impulse AI dataset & training tab."""
    st.subheader("🤖 Impulse AI — Autonomous ML Credibility Scoring")

    st.markdown("""
    **How it works:**

    1. TrustAndVerify runs verification queries and extracts structured features from each claim
    2. Features (evidence count, source trust, confidence scores, etc.) are exported as a CSV
    3. Upload the CSV to Impulse AI — it trains a production-ready model in minutes
    4. The deployed model provides instant credibility pre-screening for new claims

    This creates a **fast-path predictor** that complements the rigorous Subjective Logic scoring.
    """)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Generate Training Data")
        n_samples = st.number_input("Samples", min_value=50, max_value=1000, value=200)

        if st.button("📊 Generate Synthetic Dataset", type="primary"):
            from plgenesis_tv.dataset_generator import generate_synthetic_dataset

            path = generate_synthetic_dataset(n_samples=n_samples)
            st.success(f"Generated {n_samples} samples → {path}")

            import pandas as pd
            df = pd.read_csv(path)
            st.dataframe(df.head(10))
            st.bar_chart(df["verdict"].value_counts())

            with open(path, "rb") as f:
                st.download_button(
                    "⬇️ Download CSV for Impulse AI",
                    f,
                    file_name="verification_features.csv",
                    mime="text/csv",
                )

    with col2:
        st.markdown("### Training Instructions")
        st.markdown("""
        1. Go to [app.impulselabs.ai](https://app.impulselabs.ai)
        2. Upload the CSV under **Datasets**
        3. Click **Train Model**, select `verdict` as target
        4. Wait ~2-5 minutes for training
        5. Copy the **Deployment ID** from the model card
        6. Paste it in the sidebar configuration
        """)


def _render_hypercert_tab():
    """Hypercert generation tab."""
    st.subheader("📜 Hypercerts — Impact Attestation")

    st.markdown("""
    Each verification report generates a **Hypercert** — a verifiable impact attestation
    recording who verified what, when, with what evidence, and at what confidence level.

    Hypercerts enable:
    - **Decentralized tracking** of verification work
    - **Retroactive funding** for knowledge verification contributions
    - **Reputation building** for verification agents and contributors
    """)

    if "last_report" in st.session_state:
        report = st.session_state["last_report"]

        if st.button("🏗️ Generate Hypercert", type="primary"):
            from plgenesis_tv.hypercerts_integration import report_to_hypercert
            from plgenesis_tv.cli import _claim_to_dict

            # Convert report to dict
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

            st.success(f"Hypercert generated: **{hc.name}**")

            col1, col2, col3 = st.columns(3)
            col1.metric("Claims Verified", hc.properties.get("total_claims", 0))
            col2.metric("Evidence Gathered", hc.properties.get("total_evidence", 0))
            col3.metric("Sources Consulted", hc.properties.get("unique_sources", 0))

            # Verdict distribution
            st.markdown("**Verdict Distribution:**")
            dist = hc.properties.get("verdict_distribution", {})
            if dist:
                import pandas as pd
                st.bar_chart(pd.Series(dist))

            # Content hash
            st.code(f"Content Hash: {hc.content_hash}", language="text")

            # Download
            st.download_button(
                "⬇️ Download Hypercert JSON",
                hc.to_json(),
                file_name="hypercert.json",
                mime="application/json",
            )

            # Show full JSON
            with st.expander("View Full Hypercert JSON"):
                st.json(hc.to_dict())
    else:
        st.info("Run a verification first to generate a Hypercert from the results.")


def _render_about_tab():
    """About tab with project info."""
    st.subheader("About TrustAndVerify × PL Genesis")

    st.markdown("""
    ### What is this?

    **TrustAndVerify** is a Python library for agentic knowledge verification using
    **Subjective Logic confidence algebra** (Jøsang 2016). It decomposes research questions
    into verifiable claims, gathers evidence from multiple search backends, scores confidence
    using formal mathematics, and produces provenance-rich reports.

    ### PL Genesis Integration

    For the **PL Genesis: Frontiers of Collaboration** hackathon, we've added:

    - **🤖 Impulse AI Integration** — Train a tabular AutoML model on verification features
      to create a fast credibility pre-screener that complements Subjective Logic scoring

    - **📜 Hypercerts Integration** — Generate impact attestations from verification reports,
      enabling decentralized tracking and retroactive funding of verification work

    ### Technical Stack

    | Component | Technology |
    |---|---|
    | Core verification | `trustandverify` (PyPI) |
    | Confidence algebra | `jsonld-ex` (Subjective Logic) |
    | ML credibility scoring | Impulse AI (AutoML) |
    | Impact attestation | Hypercerts Protocol |
    | Search backends | Tavily, Brave, SerpAPI |
    | LLM backends | Gemini, OpenAI, Anthropic |

    ### Challenges Submitted

    | Challenge | Track |
    |---|---|
    | Fresh Code | Protocol Labs — $5,000 |
    | AI & Robotics | Verifiable AI — $3,000 |
    | Impulse AI | Autonomous ML — $300 |
    | Hypercerts | Impact Evaluation — $1,500 |
    | Community Vote | X Engagement — $1,000 |

    ### Links

    - 📦 [trustandverify on PyPI](https://pypi.org/project/trustandverify/)
    - 🧮 [jsonld-ex on PyPI](https://pypi.org/project/jsonld-ex/)
    - 🐙 [GitHub Repository](https://github.com/jemsbhai/plgenesis-trustandverify)
    - 📄 [Architecture Plan](https://github.com/jemsbhai/trustandverify)
    """)


if __name__ == "__main__":
    main()
