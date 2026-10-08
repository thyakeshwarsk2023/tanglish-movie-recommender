"""
Tanglish Movie Recommender — Streamlit Web App

A simple, functional demo UI for the Tanglish query movie recommender system.
Supports both English and Tanglish (code-mixed Tamil-English) typed queries.
"""

import os
import sys
import time
import streamlit as st

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.query_understanding import normalize_query, extract_constraints
from src.recommend import recommend


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Tanglish Movie Recommender",
    page_icon="🎬",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🎬 About")
    st.markdown(
        """
        **Tanglish Movie Recommender**
        
        This system understands Tanglish movie requests and recommends both 
        English and Tamil films, grounding each recommendation in the movie's 
        actual overview.
        
        ---
        
        **How it works:**
        1. Your query is normalized (Tanglish → English)
        2. Structured constraints are extracted (genre, mood, language)
        3. Cross-lingual retrieval finds matching movies
        4. Each recommendation includes a grounded explanation
        
        ---
        """
    )
    cm_path = os.path.join(PROJECT_ROOT, "results", "confusion_matrix.png")
    if os.path.exists(cm_path):
        with st.expander("📊 Model Evaluation & Confusion Matrix"):
            st.image(cm_path, caption="Trained Dual-Head Classifier (Genre & Mood)", use_container_width=True)

# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.title("🎬 Tanglish Movie Recommender")
st.caption(
    "Ask for movie recommendations in English or Tanglish — "
    "the system will understand both!"
)

# Sample queries for instant testing
SAMPLE_QUERIES = [
    "nalla comedy padam venum Tamil la",
    "romba pazhaya Tamil action padam venum",
    "emotional Tamil movie pls no horror",
    "I want a dark psychological thriller with mystery",
    "Suggest a sci-fi movie with time travel elements",
    "crime thriller padam sollunga Tamil la",
]

# Quick select pills or buttons
st.write("💡 **Quick Examples:**")
cols = st.columns(3)
for i, sample in enumerate(SAMPLE_QUERIES[:6]):
    col_target = cols[i % 3]
    if col_target.button(sample, key=f"quick_{i}"):
        st.session_state["active_query"] = sample

# Text input
query = st.text_input(
    "🔍 Enter your movie request:",
    value=st.session_state.get("active_query", ""),
    placeholder="e.g., nalla comedy padam venum / I want a thriller movie",
)

# Optional: Audio input
try:
    audio = st.audio_input("🎤 Or record your voice (optional)")
    if audio is not None:
        st.info("🎤 Voice input detected. Transcription requires SpeechRecognition + PyAudio. "
                "Using text input for now.")
except AttributeError:
    pass

# Process query
if query.strip():
    with st.spinner("Analyzing your query and finding recommendations..."):
        try:
            t_start = time.perf_counter()
            results, constraints = recommend(query.strip())
            latency_ms = (time.perf_counter() - t_start) * 1000.0
        except Exception as e:
            st.error(f"Error processing query: {e}")
            st.stop()

    # Show normalized query
    normalized = normalize_query(query)
    if normalized != query.lower().strip():
        st.info(f"📝 **Normalized query:** `{normalized}`")

    # Show extracted constraints
    st.subheader("🧠 Understood Constraints")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Genre", constraints["genre"].title())
    with col2:
        st.metric("Mood", constraints["mood"].title())
    with col3:
        lang_display = {
            "ta": "🇮🇳 Tamil",
            "en": "🇬🇧 English",
            "any": "🌐 Any",
        }
        st.metric(
            "Language", lang_display.get(constraints["language_pref"], "Any")
        )
    with col4:
        st.metric("Latency", f"{latency_ms:.1f} ms")

    if constraints.get("era") and constraints.get("era") != "any":
        era_label = "Classic / Vintage (Pre-2005)" if constraints["era"] == "old" else "Recent / Modern (2016+)"
        st.info(f"⏳ **Era Preference:** {era_label}")
    if constraints.get("exclusions"):
        st.warning(f"🚫 **Excluding:** {constraints['exclusions']}")
    if constraints.get("reference_title"):
        st.info(f"🔗 **Reference:** {constraints['reference_title']}")

    # Show recommendations
    st.subheader("🎬 Top Recommendations")

    if not results:
        st.warning("No recommendations found for your query. Try adjusting your request.")
    else:
        for i, rec in enumerate(results, 1):
            lang_emoji = "🇮🇳" if rec["language"] == "ta" else "🇬🇧"
            lang_label = "Tamil" if rec["language"] == "ta" else "English"
            year_badge = f" ({rec['release_year']})" if rec.get("release_year") else ""

            with st.container():
                st.markdown(f"### {i}. {rec['title']}{year_badge} {lang_emoji}")
                
                tag_col1, tag_col2, tag_col3 = st.columns(3)
                with tag_col1:
                    st.caption(f"🎭 {rec['genre']}")
                with tag_col2:
                    st.caption(f"🌐 {lang_label}")
                with tag_col3:
                    st.caption(f"📊 Score: {rec['similarity_score']:.3f}")

                st.markdown(f"**Why this?** {rec['explanation']}")
                st.divider()
else:
    st.markdown("👈 *Click one of the quick examples above or type your own request to begin.*")

# Footer
st.markdown("---")
st.caption(
    "Built with Streamlit • Powered by multilingual sentence-transformers "
    "and FAISS • Course project demo"
)
