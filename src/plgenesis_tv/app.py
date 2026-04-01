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
from dotenv import load_dotenv

load_dotenv()

# Page config
st.set_page_config(
    page_title="TrustAndVerify — Verifiable AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Cyberpunk / terminal aesthetic CSS (mirrors TrustGraphPLGenesis) ──────────
_CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&display=swap');

:root {
    --bg-deep: #030712;
    --bg-card: rgba(15, 23, 42, 0.6);
    --bg-elevated: rgba(30, 41, 59, 0.4);
    --border: rgba(100, 116, 139, 0.15);
    --border-glow: rgba(6, 182, 212, 0.3);
    --text-primary: #f1f5f9;
    --text-muted: #94a3b8;
    --accent-cyan: #06b6d4;
    --accent-green: #10b981;
    --accent-amber: #f59e0b;
    --accent-red: #ef4444;
    --accent-purple: #a855f7;
    --accent-pink: #ec4899;
}

/* ── Animations ───────────────────────────────────────────── */
@keyframes gradient-shift {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}
@keyframes pulse-glow {
    0%, 100% { opacity: 0.4; }
    50% { opacity: 0.9; }
}
@keyframes text-shine {
    0% { background-position: -100% 0; }
    100% { background-position: 200% 0; }
}
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-12px); }
}
@keyframes neo-sweep {
    0%   { color: rgba(6, 182, 212, 0.35); text-shadow: 0 0 0 transparent; }
    25%  { color: rgba(168, 85, 247, 0.45); text-shadow: 0 0 30px rgba(168, 85, 247, 0.15); }
    50%  { color: rgba(236, 72, 153, 0.45); text-shadow: 0 0 30px rgba(236, 72, 153, 0.15); }
    75%  { color: rgba(168, 85, 247, 0.45); text-shadow: 0 0 30px rgba(168, 85, 247, 0.15); }
    100% { color: rgba(6, 182, 212, 0.35); text-shadow: 0 0 0 transparent; }
}
@keyframes border-dance {
    0%, 100% { border-color: rgba(6, 182, 212, 0.2); }
    33% { border-color: rgba(168, 85, 247, 0.2); }
    66% { border-color: rgba(236, 72, 153, 0.2); }
}
@keyframes glow-sweep {
    0%   { box-shadow: 0 0 20px rgba(6, 182, 212, 0.15), inset 0 0 20px rgba(6, 182, 212, 0.03); }
    33%  { box-shadow: 0 0 20px rgba(168, 85, 247, 0.15), inset 0 0 20px rgba(168, 85, 247, 0.03); }
    66%  { box-shadow: 0 0 20px rgba(236, 72, 153, 0.15), inset 0 0 20px rgba(236, 72, 153, 0.03); }
    100% { box-shadow: 0 0 20px rgba(6, 182, 212, 0.15), inset 0 0 20px rgba(6, 182, 212, 0.03); }
}

/* ── Global overrides ─────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"], .main, .block-container,
[data-testid="stApp"] {
    background-color: var(--bg-deep) !important;
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', 'Courier New', Courier, monospace !important;
}

/* Scanline overlay */
[data-testid="stApp"]::after {
    content: '';
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0, 0, 0, 0.04) 2px,
        rgba(0, 0, 0, 0.04) 4px
    );
    pointer-events: none;
    z-index: 9999;
}

/* Radial gradient glow — neon sweep */
[data-testid="stApp"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(6, 182, 212, 0.15), transparent 50%),
        radial-gradient(ellipse 60% 40% at 80% 60%, rgba(168, 85, 247, 0.10), transparent 50%),
        radial-gradient(ellipse 50% 30% at 20% 80%, rgba(236, 72, 153, 0.08), transparent 50%);
    pointer-events: none;
    z-index: 0;
    animation: neon-bg-sweep 8s ease-in-out infinite;
}
@keyframes neon-bg-sweep {
    0%   { opacity: 1; filter: hue-rotate(0deg); }
    33%  { opacity: 1; filter: hue-rotate(15deg); }
    66%  { opacity: 1; filter: hue-rotate(-15deg); }
    100% { opacity: 1; filter: hue-rotate(0deg); }
}

/* ── Scrollbar ────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--accent-cyan); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-purple); }

/* ── Sidebar ──────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: rgba(10, 15, 30, 0.85) !important;
    backdrop-filter: blur(30px) !important;
    -webkit-backdrop-filter: blur(30px) !important;
    border-right: 1px solid var(--border-glow) !important;
    animation: border-dance 6s ease-in-out infinite !important;
}
[data-testid="stSidebar"] * {
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
    color: var(--accent-cyan) !important;
}

/* ── Headers ──────────────────────────────────────────────── */
h1 {
    background: linear-gradient(135deg, var(--accent-cyan) 0%, var(--accent-purple) 50%, var(--accent-pink) 100%) !important;
    background-size: 200% auto !important;
    -webkit-background-clip: text !important;
    background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    animation: text-shine 4s linear infinite !important;
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    letter-spacing: -0.02em !important;
}
h2, h3 {
    color: var(--accent-cyan) !important;
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    letter-spacing: 0.03em !important;
}

/* ── Tabs ─────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(10, 15, 30, 0.7) !important;
    border: 1px solid var(--border) !important;
    border-radius: 0 !important;
    gap: 0 !important;
    padding: 2px !important;
    animation: border-dance 6s ease-in-out infinite !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    font-size: 12px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
    border-radius: 0 !important;
    padding: 12px 24px !important;
    border: none !important;
    transition: all 0.2s ease !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple), var(--accent-pink)) !important;
    background-size: 200% 200% !important;
    animation: gradient-shift 4s ease infinite !important;
    color: #000 !important;
    font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {
    display: none !important;
}

/* ── Buttons ──────────────────────────────────────────────── */
.stButton > button {
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-radius: 2px !important;
    transition: all 0.2s ease !important;
    border: 1px solid var(--accent-cyan) !important;
}
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple), var(--accent-pink)) !important;
    background-size: 200% 200% !important;
    animation: gradient-shift 4s ease infinite !important;
    color: #000 !important;
    font-weight: 600 !important;
    border: none !important;
    box-shadow: 0 0 20px rgba(6, 182, 212, 0.2) !important;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    box-shadow: 0 0 40px rgba(6, 182, 212, 0.3), 0 0 40px rgba(168, 85, 247, 0.2) !important;
}
.stButton > button[kind="secondary"],
.stButton > button[data-testid="stBaseButton-secondary"] {
    background: transparent !important;
    color: var(--accent-cyan) !important;
    border: 1px solid var(--accent-cyan) !important;
}
.stDownloadButton > button {
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    background: transparent !important;
    color: var(--accent-purple) !important;
    border: 1px solid var(--accent-purple) !important;
    border-radius: 2px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

/* ── Inputs ───────────────────────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stNumberInput > div > div > input {
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    background: var(--bg-deep) !important;
    border: none !important;
    color: var(--text-primary) !important;
    border-radius: 2px !important;
}
/* Animated gradient border wrapper on inputs — neon sweep */
.stTextInput > div > div,
.stTextArea > div > div {
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple), var(--accent-pink), var(--accent-cyan)) !important;
    background-size: 300% 300% !important;
    animation: gradient-shift 3s ease infinite !important;
    padding: 2px !important;
    border-radius: 3px !important;
    box-shadow: 0 0 15px rgba(6, 182, 212, 0.12), 0 0 30px rgba(168, 85, 247, 0.08) !important;
}
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    margin: 0 !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    box-shadow: none !important;
}
/* Number input keeps simpler style */
.stNumberInput > div > div > input {
    border: 1px solid var(--border-glow) !important;
    background: rgba(10, 15, 30, 0.7) !important;
}

/* ── Sliders ──────────────────────────────────────────────── */
[data-testid="stSlider"] [role="slider"] {
    background: var(--accent-cyan) !important;
}

/* ── Expanders (glassmorphism cards with animated corner brackets) ── */
[data-testid="stExpander"] {
    background: rgba(10, 15, 30, 0.7) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 1px solid var(--border) !important;
    border-radius: 0 !important;
    position: relative !important;
    margin-bottom: 8px !important;
    animation: glow-sweep 6s ease-in-out infinite, border-dance 6s ease-in-out infinite !important;
}
[data-testid="stExpander"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 16px; height: 16px;
    border-top: 2px solid var(--accent-cyan);
    border-left: 2px solid var(--accent-cyan);
    pointer-events: none;
    z-index: 1;
    animation: border-dance 6s ease-in-out infinite;
}
[data-testid="stExpander"]::after {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 16px; height: 16px;
    border-top: 2px solid var(--accent-cyan);
    border-right: 2px solid var(--accent-cyan);
    pointer-events: none;
    z-index: 1;
    animation: border-dance 6s ease-in-out infinite 0.5s;
}
[data-testid="stExpander"] summary {
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    letter-spacing: 0.02em !important;
}

/* ── Metrics ──────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: rgba(10, 15, 30, 0.6) !important;
    border: 1px solid var(--border) !important;
    padding: 16px !important;
    border-radius: 0 !important;
    animation: border-dance 6s ease-in-out infinite !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--text-muted) !important;
    font-size: 11px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple), var(--accent-pink)) !important;
    background-size: 200% auto !important;
    -webkit-background-clip: text !important;
    background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    animation: text-shine 4s linear infinite !important;
}

/* ── Progress bars ────────────────────────────────────────── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple), var(--accent-pink)) !important;
    background-size: 200% 200% !important;
    animation: gradient-shift 4s ease infinite !important;
    border-radius: 0 !important;
}
.stProgress > div > div {
    background: rgba(30, 41, 59, 0.4) !important;
    border-radius: 0 !important;
}

/* ── Alert boxes ──────────────────────────────────────────── */
[data-testid="stAlert"] {
    font-family: 'JetBrains Mono', monospace !important;
    border-radius: 0 !important;
    border-left-width: 4px !important;
    background: rgba(10, 15, 30, 0.7) !important;
}

/* ── Dataframes / tables ──────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 0 !important;
}

/* ── Code blocks ──────────────────────────────────────────── */
code, pre, .stCodeBlock {
    font-family: 'JetBrains Mono', monospace !important;
    background: rgba(10, 15, 30, 0.8) !important;
    border: 1px solid var(--border) !important;
    border-radius: 0 !important;
}

/* ── Dividers ─────────────────────────────────────────────── */
hr, [data-testid="stDivider"] {
    border-color: var(--border) !important;
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple), var(--accent-pink), transparent) !important;
    background-size: 200% 100% !important;
    animation: gradient-shift 4s ease infinite !important;
    height: 1px !important;
    opacity: 0.5 !important;
}

/* ── Links ────────────────────────────────────────────────── */
a {
    color: var(--accent-cyan) !important;
    text-decoration: none !important;
    transition: all 0.3s ease !important;
}
a:hover {
    text-shadow: 0 0 20px var(--accent-cyan) !important;
}

/* ── Bar charts ───────────────────────────────────────────── */
[data-testid="stVegaLiteChart"] {
    background: rgba(10, 15, 30, 0.5) !important;
    border: 1px solid var(--border) !important;
    border-radius: 0 !important;
    padding: 8px !important;
}

/* ── JSON viewer ──────────────────────────────────────────── */
[data-testid="stJson"] {
    background: rgba(10, 15, 30, 0.8) !important;
    border: 1px solid var(--border) !important;
    border-radius: 0 !important;
}

/* ── Generic text ─────────────────────────────────────────── */
p, li, span, label, div {
    font-family: 'JetBrains Mono', 'Courier New', monospace;
}

/* ── Toast / notifications ────────────────────────────────── */
[data-testid="stToast"] {
    background: rgba(10, 15, 30, 0.9) !important;
    border: 1px solid var(--accent-cyan) !important;
    border-radius: 0 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Number input buttons ─────────────────────────────────── */
[data-testid="stNumberInput"] button {
    color: var(--accent-cyan) !important;
}
</style>
"""
st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


def _secret(s: str) -> str:
    """Strip accidental whitespace from pasted API keys."""
    return (s or "").strip()


def _get_opinion_values(opinion):
    """Safely extract b, d, u, p from an opinion (object or dict).

    Opinion.projected_probability() is a METHOD in jsonld-ex, not a property.
    We compute P = b + a*u manually to avoid that footgun.
    """
    if opinion is None:
        return None, None, None, None
    if hasattr(opinion, "belief"):
        b = float(opinion.belief)
        d = float(opinion.disbelief)
        u = float(opinion.uncertainty)
        br = float(getattr(opinion, "base_rate", 0.5))
    elif isinstance(opinion, dict):
        b = float(opinion.get("belief", 0))
        d = float(opinion.get("disbelief", 0))
        u = float(opinion.get("uncertainty", 1))
        br = float(opinion.get("base_rate", 0.5))
    else:
        return None, None, None, None
    p = b + br * u
    return b, d, u, p


def main():
    # "TrustGraph" — big retro cyberpunk hero title
    st.markdown(
        """
<div style="text-align:center; padding:48px 0 20px; position:relative;">
  <!-- Neon chrome reflection line above -->
  <div style="
    width: 220px; height: 2px; margin: 0 auto 18px;
    background: linear-gradient(90deg, transparent, #06b6d4, #a855f7, #ec4899, transparent);
    background-size: 200% 100%;
    animation: gradient-shift 3s ease infinite;
    opacity: 0.7;
  "></div>

  <!-- TrustGraph — main title -->
  <div style="
    font-family: 'JetBrains Mono', 'Courier New', monospace;
    font-size: 72px;
    font-weight: 700;
    letter-spacing: 0.04em;
    line-height: 1;
    position: relative;
    display: inline-block;
    background: linear-gradient(90deg, #06b6d4, #a855f7, #ec4899, #06b6d4);
    background-size: 300% auto;
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: text-shine 3s linear infinite;
    filter: drop-shadow(0 0 20px rgba(6, 182, 212, 0.35))
            drop-shadow(0 0 40px rgba(168, 85, 247, 0.2))
            drop-shadow(0 0 60px rgba(236, 72, 153, 0.1));
  ">TrustGraph</div>

  <!-- Neon chrome reflection line below -->
  <div style="
    width: 220px; height: 2px; margin: 18px auto 0;
    background: linear-gradient(90deg, transparent, #ec4899, #a855f7, #06b6d4, transparent);
    background-size: 200% 100%;
    animation: gradient-shift 3s ease infinite;
    opacity: 0.7;
  "></div>
</div>

<p style="text-align:center; color:#94a3b8; font-size:14px; letter-spacing:0.05em; line-height:1.8; max-width:560px; margin:0 auto 28px auto;">
  &gt; Agentic knowledge verification powered by
  <a href="https://pypi.org/project/trustandverify/" target="_blank" style="color:#06b6d4 !important; text-decoration:none; transition: text-shadow 0.3s ease;">trustandverify</a>
  with Subjective Logic_<span style="animation: pulse-glow 1s ease-in-out infinite;">|</span>
</p>
""",
        unsafe_allow_html=True,
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
        st.markdown(
            '<div style="font-size:11px; color:#94a3b8; letter-spacing:0.05em; line-height:1.8;">'
            '// Built for <span style="color:#06b6d4;">PL Genesis</span>: '
            'Frontiers of Collaboration<br><br>'
            '<a href="https://protocol.ai">Protocol Labs</a> · '
            '<a href="https://impulselabs.ai">Impulse AI</a> · '
            '<a href="https://hypercerts.org">Hypercerts</a>'
            '</div>',
            unsafe_allow_html=True,
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
        run_btn = st.button("[VERIFY]", type="primary", use_container_width=True)

    if run_btn and query:
        tavily_key, gemini_key = _secret(tavily_key), _secret(gemini_key)
        if not tavily_key or not gemini_key:
            st.error("Please provide Tavily and Gemini API keys in the sidebar.")
            return

        # Set env vars for trustandverify / litellm (both names are accepted for Gemini)
        os.environ["TAVILY_API_KEY"] = tavily_key
        os.environ["GEMINI_API_KEY"] = gemini_key
        os.environ["GOOGLE_API_KEY"] = gemini_key

        with st.spinner("[VERIFYING CLAIMS...]"):
            report = _run_verification(query, num_claims, max_sources)

        if report is None:
            st.error(
                "Verification failed. If you see an authentication error above, confirm your "
                "**Gemini** key is from [Google AI Studio](https://aistudio.google.com/apikey) "
                "(starts with `AIza`), with no extra spaces or quotes. Tavily keys come from "
                "[tavily.com](https://tavily.com)."
            )
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
        from trustandverify.search import TavilySearch
        from trustandverify.llm import GeminiBackend

        config = TrustConfig(num_claims=num_claims, max_sources_per_claim=max_sources)
        agent = TrustAgent(
            config=config,
            search=TavilySearch(),
            llm=GeminiBackend(),
        )
        report = asyncio.run(agent.verify(query))
        return report
    except ImportError as e:
        st.error(
            f"trustandverify import error: {e}\n\n"
            "Run:\n```\npip install trustandverify[tavily,gemini]\n```"
        )
        return None
    except Exception as e:
        err = str(e)
        st.error(f"Verification error: {e}")
        if "API_KEY_INVALID" in err or "API key not valid" in err or "AuthenticationError" in err:
            st.warning(
                "Gemini rejected the API key. Create a new key at "
                "[aistudio.google.com/apikey](https://aistudio.google.com/apikey), paste it "
                "into the sidebar (not a Google Cloud *OAuth* secret), and ensure any `.env` "
                "value has no surrounding quotes."
            )
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

        # Safely extract opinion values
        b, d, u, p = _get_opinion_values(opinion)

        with st.expander(f"{icon} **{verdict.upper()}** — {text[:80]}...", expanded=(i == 0)):
            col_v, col_p, col_e = st.columns(3)
            col_v.metric("Verdict", verdict)
            col_p.metric("P(true)", f"{p:.3f}" if p is not None else "N/A")
            col_e.metric("Evidence", len(evidence))

            if b is not None:
                st.markdown("**Subjective Logic Opinion:**")
                col_b, col_d, col_u = st.columns(3)
                col_b.progress(max(0.0, min(1.0, b)), text=f"Belief: {b:.3f}")
                col_d.progress(max(0.0, min(1.0, d)), text=f"Disbelief: {d:.3f}")
                col_u.progress(max(0.0, min(1.0, u)), text=f"Uncertainty: {u:.3f}")

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

        if st.button("[GENERATE_DATASET]", type="primary"):
            from plgenesis_tv.dataset_generator import generate_synthetic_dataset

            path = generate_synthetic_dataset(n_samples=n_samples)
            st.success(f"Generated {n_samples} samples → {path}")

            import pandas as pd
            df = pd.read_csv(path)
            st.dataframe(df.head(10))
            st.bar_chart(df["verdict"].value_counts())

            with open(path, "rb") as f:
                st.download_button(
                    "[DOWNLOAD_CSV]",
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

        if st.button("[GENERATE_HYPERCERT]", type="primary"):
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
                "[DOWNLOAD_JSON]",
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
    | Fresh Code | Protocol Labs |
    | AI & Robotics | Verifiable AI |
    | Impulse AI | Autonomous ML |
    | Hypercerts | Impact Evaluation |
    | Community Vote | X Engagement |

    ### Links

    - [trustandverify on PyPI](https://pypi.org/project/trustandverify/)
    - [jsonld-ex on PyPI](https://pypi.org/project/jsonld-ex/)
    - [GitHub Repository](https://github.com/jemsbhai/plgenesis-trustandverify)
    - [Architecture Plan](https://github.com/jemsbhai/trustandverify)
    """)

    # Terminal-style footer
    st.markdown(
        '<div style="margin-top:40px; padding-top:20px; border-top:1px solid rgba(100,116,139,0.15); '
        'text-align:center; font-size:11px; color:#94a3b8; letter-spacing:0.05em;">'
        '// Powered by <a href="https://pypi.org/project/trustandverify/">trustandverify</a> '
        '& <a href="https://pypi.org/project/jsonld-ex/">jsonld-ex</a> '
        '// Subjective Logic - Josang 2016</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
