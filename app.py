"""
Tanglish Movie Recommender — Streamlit Web Application

A refined, minimal discovery interface for English and code-mixed Tanglish queries.
Preserves existing recommender pipeline, constraint extraction, and evaluation data.
"""

import os
import sys
import time
import html
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.query_understanding import normalize_query, extract_constraints
from src.recommend import recommend

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Tanglish Movie Recommender",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — Restrained dark theme & compact layout
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Global dark palette */
    :root {
        --bg-main: #0c0e12;
        --surface-panel: #13161f;
        --surface-card: #171b24;
        --surface-card-hover: #1c212c;
        --surface-border: #222634;
        --surface-border-subtle: #1c202b;
        --accent-blue: #3b82f6;
        --accent-blue-hover: #2563eb;
        --accent-blue-subtle: rgba(59, 130, 246, 0.12);
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --tag-exclusion: #ef4444;
        --tag-exclusion-bg: rgba(239, 68, 68, 0.10);
    }

    /* Streamlit core overrides */
    .stApp {
        background-color: var(--bg-main) !important;
        color: var(--text-primary) !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    }

    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 760px !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0f1218 !important;
        border-right: 1px solid var(--surface-border) !important;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem !important;
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
    }

    /* Header */
    .app-header {
        margin-bottom: 1.25rem;
    }
    .app-title {
        font-size: 1.65rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.025em;
        margin: 0 0 0.2rem 0;
        line-height: 1.2;
    }
    .app-subtitle {
        font-size: 0.92rem;
        color: var(--text-secondary);
        margin: 0;
        font-weight: 400;
    }

    /* Search input field */
    .stTextInput > div > div > input {
        background-color: var(--surface-panel) !important;
        border: 1px solid var(--surface-border) !important;
        color: var(--text-primary) !important;
        border-radius: 6px !important;
        font-size: 0.95rem !important;
        padding: 0.6rem 0.85rem !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }
    .stTextInput > div > div > input:focus {
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 0 1px var(--accent-blue) !important;
    }

    /* Primary button */
    .stButton > button[data-testid="stBaseButton-primary"],
    .stButton > button[data-testid="baseButton-primary"],
    button[kind="primary"] {
        background-color: var(--accent-blue) !important;
        border: 1px solid var(--accent-blue) !important;
        color: #ffffff !important;
        font-weight: 500 !important;
        border-radius: 6px !important;
        font-size: 0.9rem !important;
        padding: 0.55rem 1rem !important;
        transition: background-color 0.15s ease, border-color 0.15s ease;
    }
    .stButton > button[data-testid="stBaseButton-primary"]:hover,
    .stButton > button[data-testid="baseButton-primary"]:hover,
    button[kind="primary"]:hover {
        background-color: var(--accent-blue-hover) !important;
        border-color: var(--accent-blue-hover) !important;
    }

    /* Example query secondary buttons */
    .stButton > button[data-testid="stBaseButton-secondary"],
    .stButton > button[data-testid="baseButton-secondary"],
    .stButton > button:not([kind="primary"]):not([data-testid="stBaseButton-primary"]):not([data-testid="baseButton-primary"]) {
        background-color: var(--surface-panel) !important;
        border: 1px solid var(--surface-border) !important;
        color: var(--text-secondary) !important;
        border-radius: 6px !important;
        font-size: 0.82rem !important;
        padding: 0.45rem 0.75rem !important;
        text-align: left !important;
        white-space: normal !important;
        line-height: 1.35 !important;
        width: 100% !important;
        height: auto !important;
        transition: border-color 0.15s ease, color 0.15s ease, background-color 0.15s ease;
    }
    .stButton > button:not([kind="primary"]):not([data-testid="stBaseButton-primary"]):not([data-testid="baseButton-primary"]):hover {
        border-color: var(--accent-blue) !important;
        color: var(--text-primary) !important;
        background-color: var(--surface-card) !important;
    }

    /* Section headings */
    .section-label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-muted);
        margin: 0.8rem 0 0.4rem 0;
    }
    .results-heading {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 1.4rem 0 0.8rem 0;
        letter-spacing: -0.015em;
    }

    /* Constraints summary container */
    .constraints-card {
        background-color: var(--surface-panel);
        border: 1px solid var(--surface-border);
        border-radius: 6px;
        padding: 10px 14px;
        margin: 1rem 0 1.25rem 0;
    }
    .constraints-line {
        font-size: 0.86rem;
        color: var(--text-primary);
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 6px 10px;
    }
    .constraint-item {
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }
    .constraint-key {
        color: var(--text-muted);
        font-weight: 500;
    }
    .constraint-val {
        color: var(--text-primary);
        font-weight: 600;
    }
    .constraint-divider {
        color: var(--surface-border);
        user-select: none;
    }
    .exclusion-line {
        margin-top: 6px;
        padding-top: 6px;
        border-top: 1px solid var(--surface-border-subtle);
        font-size: 0.82rem;
        color: #fca5a5;
    }
    .exclusion-key {
        color: #ef4444;
        font-weight: 600;
    }
    .technical-meta-line {
        margin-top: 6px;
        padding-top: 6px;
        border-top: 1px solid var(--surface-border-subtle);
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.76rem;
        color: var(--text-muted);
    }
    .technical-meta-line code {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 1px 4px;
        border-radius: 3px;
        color: #cbd5e1;
        font-size: 0.75rem;
    }

    /* Recommendation card */
    .rec-item {
        background-color: var(--surface-card);
        border: 1px solid var(--surface-border);
        border-radius: 6px;
        padding: 14px 16px;
        margin-bottom: 10px;
        transition: border-color 0.15s ease;
    }
    .rec-item:hover {
        border-color: #2b3242;
    }
    .rec-header {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 4px;
        gap: 12px;
    }
    .rec-title-text {
        font-size: 0.98rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0;
        line-height: 1.3;
    }
    .rec-score-badge {
        font-size: 0.74rem;
        color: var(--accent-blue);
        background-color: var(--accent-blue-subtle);
        padding: 2px 7px;
        border-radius: 4px;
        font-weight: 500;
        font-variant-numeric: tabular-nums;
        white-space: nowrap;
    }
    .rec-meta-line {
        font-size: 0.81rem;
        color: var(--text-secondary);
        margin-bottom: 8px;
    }
    .rec-overview-text {
        font-size: 0.87rem;
        color: #cbd5e1;
        line-height: 1.5;
        margin: 0;
    }
    .rec-more-toggle {
        margin-top: 5px;
        font-size: 0.77rem;
    }
    .rec-more-toggle summary {
        cursor: pointer;
        color: var(--accent-blue);
        user-select: none;
    }
    .rec-more-toggle summary:hover {
        text-decoration: underline;
    }
    .rec-full-body {
        margin-top: 6px;
        font-size: 0.83rem;
        color: var(--text-secondary);
        line-height: 1.45;
        border-left: 2px solid var(--surface-border);
        padding-left: 9px;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 18px;
        border-bottom: 1px solid var(--surface-border);
        margin-bottom: 1.25rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 4px !important;
        color: var(--text-muted) !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        border: none !important;
        background: transparent !important;
    }
    .stTabs [aria-selected="true"] {
        color: var(--accent-blue) !important;
        border-bottom: 2px solid var(--accent-blue) !important;
    }

    /* Empty state */
    .empty-state {
        padding: 2.5rem 1rem;
        text-align: center;
        color: var(--text-muted);
        font-size: 0.9rem;
        border: 1px dashed var(--surface-border);
        border-radius: 6px;
        margin-top: 1rem;
    }

    /* Metric card in Evaluation tab */
    .eval-card {
        background-color: var(--surface-card);
        border: 1px solid var(--surface-border);
        border-radius: 6px;
        padding: 12px 14px;
        text-align: left;
    }
    .eval-card-label {
        font-size: 0.75rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 4px;
    }
    .eval-card-val {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    .eval-card-sub {
        font-size: 0.75rem;
        color: var(--text-secondary);
        margin-top: 2px;
    }

    /* Footer */
    .app-footer {
        margin-top: 2.5rem;
        padding-top: 1rem;
        border-top: 1px solid var(--surface-border);
        font-size: 0.76rem;
        color: var(--text-muted);
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — About & System Context
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="section-label">About</div>', unsafe_allow_html=True)
    st.markdown(
        """
        Search for Tamil and English films in English or Tanglish, 
        with recommendations grounded in movie metadata.
        """
    )
    st.divider()

    st.markdown('<div class="section-label">System Specs</div>', unsafe_allow_html=True)
    st.caption(
        """
        • **Catalog:** 4,684 films (4,502 EN, 182 TA)  
        • **Model:** paraphrase-multilingual-MiniLM-L12-v2  
        • **Classifier:** Dual-Head Feedforward (Genre & Mood)  
        • **Retrieval:** FAISS IndexFlatIP + Metadata Scoring  
        • **Language compliance:** 100% on benchmark set  
        """
    )

# ---------------------------------------------------------------------------
# Main Application Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1 class="app-title">Tanglish Movie Recommender</h1>
        <p class="app-subtitle">Find Tamil and English films using natural language.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Navigation Tabs: Discovery & Evaluation
# ---------------------------------------------------------------------------
tab_search, tab_eval = st.tabs(["Search", "Evaluation"])

with tab_search:
    # Session state initialization
    if "search_query" not in st.session_state:
        st.session_state["search_query"] = ""

    def on_example_click(example_text: str):
        st.session_state["search_query"] = example_text

    # Search Bar
    col_input, col_btn = st.columns([4.2, 1.1])
    with col_input:
        query_input = st.text_input(
            "Query input",
            key="search_query",
            placeholder="Search in English or Tanglish, e.g., nalla comedy padam venum",
            label_visibility="collapsed",
        )
    with col_btn:
        recommend_clicked = st.button("Recommend", type="primary", use_container_width=True)

    # 4 Quick example query buttons in a 2x2 grid
    st.markdown('<div class="section-label">Example Queries</div>', unsafe_allow_html=True)
    sample_queries = [
        "nalla comedy padam venum Tamil la",
        "emotional Tamil movie without horror",
        "dark psychological thriller with mystery",
        "romba pazhaya Tamil action padam venum",
    ]

    ex_col1, ex_col2 = st.columns(2)
    with ex_col1:
        st.button(
            sample_queries[0],
            key="btn_sample_0",
            on_click=on_example_click,
            args=(sample_queries[0],),
            use_container_width=True,
        )
        st.button(
            sample_queries[1],
            key="btn_sample_1",
            on_click=on_example_click,
            args=(sample_queries[1],),
            use_container_width=True,
        )
    with ex_col2:
        st.button(
            sample_queries[2],
            key="btn_sample_2",
            on_click=on_example_click,
            args=(sample_queries[2],),
            use_container_width=True,
        )
        st.button(
            sample_queries[3],
            key="btn_sample_3",
            on_click=on_example_click,
            args=(sample_queries[3],),
            use_container_width=True,
        )

    # Resolve active query
    active_query = query_input.strip()

    if active_query:
        st.session_state["last_processed_query"] = active_query

        # Run recommendation pipeline
        try:
            t0 = time.perf_counter()
            results, constraints = recommend(active_query, top_k=5)
            latency_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as e:
            st.error(f"Error generating recommendations: {e}")
            st.stop()

        # Understood Constraints Display
        genre = constraints.get("genre", "")
        mood = constraints.get("mood", "")
        lang_pref = constraints.get("language_pref", "any")
        era = constraints.get("era", "any")
        exclusions = constraints.get("exclusions", "")
        ref_title = constraints.get("reference_title", "")

        constraint_items = []
        if genre and genre.lower() not in ["", "unknown", "none"]:
            constraint_items.append(
                f'<span class="constraint-item"><span class="constraint-key">Genre:</span> <span class="constraint-val">{html.escape(genre.title())}</span></span>'
            )
        if mood and mood.lower() not in ["", "unknown", "none", "neutral"]:
            constraint_items.append(
                f'<span class="constraint-item"><span class="constraint-key">Mood:</span> <span class="constraint-val">{html.escape(mood.title())}</span></span>'
            )

        lang_label_map = {"ta": "Tamil", "en": "English", "any": "Any"}
        lang_text = lang_label_map.get(lang_pref, "Any")
        constraint_items.append(
            f'<span class="constraint-item"><span class="constraint-key">Language:</span> <span class="constraint-val">{lang_text}</span></span>'
        )

        if era == "old":
            constraint_items.append(
                '<span class="constraint-item"><span class="constraint-key">Era:</span> <span class="constraint-val">Classic (pre-2005)</span></span>'
            )
        elif era == "new":
            constraint_items.append(
                '<span class="constraint-item"><span class="constraint-key">Era:</span> <span class="constraint-val">Recent (2016+)</span></span>'
            )

        if ref_title:
            constraint_items.append(
                f'<span class="constraint-item"><span class="constraint-key">Reference:</span> <span class="constraint-val">{html.escape(ref_title)}</span></span>'
            )

        # Build constraints HTML
        sep = ' <span class="constraint-divider">|</span> '
        constraints_line_html = sep.join(constraint_items)

        exclusion_html = ""
        if exclusions and exclusions.strip():
            exclusion_html = f'<div class="exclusion-line"><span class="exclusion-key">Excluding:</span> {html.escape(exclusions.title())}</div>'

        # Normalized query & Latency metadata
        normalized = normalize_query(active_query)
        norm_html = ""
        if normalized.lower().strip() != active_query.lower().strip():
            norm_html = f'<span>Normalized: <code>{html.escape(normalized)}</code></span>'
        else:
            norm_html = '<span>Cross-lingual retrieval</span>'

        meta_html = f"""
        <div class="technical-meta-line">
            {norm_html}
            <span>Retrieval latency: {latency_ms:.1f} ms</span>
        </div>
        """

        st.markdown(
            f"""
            <div class="constraints-card">
                <div class="constraints-line">{constraints_line_html}</div>
                {exclusion_html}
                {meta_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Recommendation Results
        st.markdown('<div class="results-heading">Top Recommendations</div>', unsafe_allow_html=True)

        if not results:
            st.info("No matching recommendations found for this query. Try adjusting your constraints.")
        else:
            for i, rec in enumerate(results, 1):
                title = rec.get("title", "Untitled")
                year = rec.get("release_year")
                lang = "Tamil" if rec.get("language") == "ta" else "English"
                genre_str = rec.get("genre", "")
                score = rec.get("similarity_score", 0.0)
                overview = rec.get("overview", "").strip()

                # Build clean metadata line: e.g. "2023 · Tamil · Comedy, Romance"
                meta_parts = []
                if year is not None and not pd.isna(year):
                    try:
                        meta_parts.append(str(int(float(year))))
                    except (ValueError, TypeError):
                        meta_parts.append(str(year))
                meta_parts.append(lang)
                if genre_str:
                    meta_parts.append(str(genre_str))
                meta_line = " · ".join(meta_parts)

                # Grounded explanation from overview
                if overview:
                    if len(overview) > 175:
                        short_overview = overview[:175].rsplit(" ", 1)[0] + "..."
                        overview_html = f"""
                        <p class="rec-overview-text">{html.escape(short_overview)}</p>
                        <details class="rec-more-toggle">
                            <summary>Full overview</summary>
                            <div class="rec-full-body">{html.escape(overview)}</div>
                        </details>
                        """
                    else:
                        overview_html = f'<p class="rec-overview-text">{html.escape(overview)}</p>'
                else:
                    overview_html = f'<p class="rec-overview-text">{html.escape(rec.get("explanation", ""))}</p>'

                # Render result card
                st.markdown(
                    f"""
                    <div class="rec-item">
                        <div class="rec-header">
                            <h3 class="rec-title-text">{i}. {html.escape(title)}</h3>
                            <span class="rec-score-badge">Score: {score:.3f}</span>
                        </div>
                        <div class="rec-meta-line">{html.escape(meta_line)}</div>
                        {overview_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    else:
        st.markdown(
            """
            <div class="empty-state">
                Enter a movie request above or choose an example query to begin.
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Tab 2: Model Evaluation & Performance
# ---------------------------------------------------------------------------
with tab_eval:
    st.markdown('<div class="results-heading">Model Evaluation & Benchmark Performance</div>', unsafe_allow_html=True)
    st.markdown(
        """
        The recommendation pipeline uses a trained dual-head neural classifier built atop 
        `paraphrase-multilingual-MiniLM-L12-v2` embeddings (384 → 128 → 64 → 11/9 output heads) 
        to extract structured genre and mood slots from raw English and Tanglish queries.
        """
    )

    # Metric summaries in clean cards
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown(
            """
            <div class="eval-card">
                <div class="eval-card-label">Composite Slot Accuracy</div>
                <div class="eval-card-val">93.33%</div>
                <div class="eval-card-sub">28/30 test queries</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col2:
        st.markdown(
            """
            <div class="eval-card">
                <div class="eval-card-label">Language Compliance</div>
                <div class="eval-card-val">100.0%</div>
                <div class="eval-card-sub">Zero language leakage</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col3:
        st.markdown(
            """
            <div class="eval-card">
                <div class="eval-card-label">Pipeline Latency</div>
                <div class="eval-card-val">~16.3 ms</div>
                <div class="eval-card-sub">Component-level estimate</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # Confusion matrix display
    cm_path = os.path.join(PROJECT_ROOT, "results", "confusion_matrix.png")
    if os.path.exists(cm_path):
        st.markdown('<div class="section-label">Dual-Head Classifier Confusion Matrix</div>', unsafe_allow_html=True)
        st.image(
            cm_path,
            caption="Confusion matrix across 11 genre classes and 9 mood classes on test queries.",
            use_container_width=True,
        )

    # Benchmark results table
    eval_csv_path = os.path.join(PROJECT_ROOT, "results", "evaluation_results.csv")
    if os.path.exists(eval_csv_path):
        with st.expander("View 30-Query Evaluation Benchmark Table", expanded=False):
            try:
                df = pd.read_csv(eval_csv_path)
                display_cols = [
                    "query_text",
                    "language_origin",
                    "pred_genre",
                    "pred_mood",
                    "pred_language_pref",
                    "genre_correct",
                    "mood_correct",
                    "lang_hit",
                ]
                avail_cols = [c for c in display_cols if c in df.columns]
                st.dataframe(df[avail_cols], use_container_width=True, hide_index=True)
            except Exception as e:
                st.caption(f"Could not load benchmark table: {e}")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-footer">
        Tanglish Movie Recommender • Multilingual Sentence Transformers & FAISS • Undergraduate Project
    </div>
    """,
    unsafe_allow_html=True,
)
