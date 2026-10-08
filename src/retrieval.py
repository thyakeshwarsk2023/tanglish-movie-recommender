"""
retrieval.py — Cross-lingual movie retrieval using FAISS + multilingual embeddings

This module:
1. Builds a FAISS index from the movie catalog embeddings
2. Retrieves candidate movies based on extracted constraints
3. Uses the SAME multilingual embedding model as query_understanding.py
   so that queries and catalog items live in the same vector space

The key design decision: we use paraphrase-multilingual-MiniLM-L12-v2 for BOTH
the catalog embeddings AND the query embeddings. This means an English query
like "good action movie" and a Tamil movie's overview exist in the same 384-dim
space, enabling genuine cross-lingual retrieval.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import faiss

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.query_understanding import get_embedding_model


MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CATALOG_PATH = os.path.join(DATA_DIR, "catalog.csv")
FAISS_INDEX_PATH = os.path.join(MODEL_DIR, "faiss_index.bin")
ID_MAP_PATH = os.path.join(MODEL_DIR, "id_map.pkl")
EMBEDDINGS_PATH = os.path.join(MODEL_DIR, "catalog_embeddings.npy")

# Cache for catalog and embeddings
_catalog_cache = None
_embeddings_cache = None


def _load_catalog():
    """Load and cache the catalog DataFrame."""
    global _catalog_cache
    if _catalog_cache is None:
        _catalog_cache = pd.read_csv(CATALOG_PATH)
    return _catalog_cache


def _load_embeddings():
    """Load and cache the catalog embeddings."""
    global _embeddings_cache
    if _embeddings_cache is None:
        if os.path.exists(EMBEDDINGS_PATH):
            _embeddings_cache = np.load(EMBEDDINGS_PATH)
        else:
            raise FileNotFoundError(
                f"Embeddings not found at {EMBEDDINGS_PATH}. Run build_index() first."
            )
    return _embeddings_cache


# ---------------------------------------------------------------------------
# 1. Build FAISS index from catalog
# ---------------------------------------------------------------------------
def build_index(catalog_csv_path: str = None):
    """
    Build a FAISS index from the movie catalog.
    
    Steps:
    1. Load catalog.csv
    2. Embed each movie's (title + overview) with the multilingual model
    3. Normalize embeddings for cosine similarity via inner product
    4. Build FAISS IndexFlatIP (inner product on L2-normalized vectors = cosine sim)
    5. Save index, id mapping, and embeddings to models/
    
    Args:
        catalog_csv_path: Path to catalog CSV. Defaults to data/catalog.csv.
    """
    if catalog_csv_path is None:
        catalog_csv_path = CATALOG_PATH

    print("[1/4] Loading catalog...")
    catalog = pd.read_csv(catalog_csv_path)
    print(f"  {len(catalog)} movies loaded")

    # Prepare text for embedding: title + release_year + genres + overview
    texts = []
    for _, row in catalog.iterrows():
        title = str(row["title"]) if pd.notna(row["title"]) else ""
        overview = str(row["overview"]) if pd.notna(row["overview"]) else ""
        year_str = f"({int(row['release_year'])}) " if pd.notna(row.get("release_year")) else ""
        genres_str = f"Genres: {row['genres']}. " if pd.notna(row.get("genres")) else ""
        texts.append(f"{title} {year_str}{genres_str}Overview: {overview}")

    print("[2/4] Embedding catalog with multilingual model...")
    model = get_embedding_model()
    embeddings = model.encode(
        texts, show_progress_bar=True, convert_to_numpy=True, batch_size=64
    )
    print(f"  Embedding shape: {embeddings.shape}")

    # Normalize for cosine similarity
    print("[3/4] Building FAISS index...")
    faiss.normalize_L2(embeddings)
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product on normalized = cosine
    index.add(embeddings)
    print(f"  FAISS index built: {index.ntotal} vectors, dim={dim}")

    # Save
    print("[4/4] Saving index and mappings...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    faiss.write_index(index, FAISS_INDEX_PATH)
    np.save(EMBEDDINGS_PATH, embeddings)

    # Save movie_id mapping: index position -> movie_id
    id_map = {i: int(row["movie_id"]) for i, row in catalog.iterrows()}
    with open(ID_MAP_PATH, "wb") as f:
        pickle.dump(id_map, f)

    print(f"  Index saved to: {FAISS_INDEX_PATH}")
    print(f"  ID map saved to: {ID_MAP_PATH}")
    print(f"  Embeddings saved to: {EMBEDDINGS_PATH}")
    print(f"\n[DONE] Index built successfully.")


def _is_genre_match(movie_genres: str, target_genre: str) -> bool:
    """Check if movie genres string contains target genre or common aliases."""
    if not target_genre or not movie_genres:
        return False
    target = target_genre.lower().strip()
    mg = str(movie_genres).lower()
    if target in mg:
        return True
    if target in ["science fiction", "sci-fi", "scifi"] and ("science fiction" in mg or "sci-fi" in mg):
        return True
    if target in ["romance", "romantic"] and ("romance" in mg or "romantic" in mg):
        return True
    return False


def _genre_match_mask(series: pd.Series, genre_term: str) -> pd.Series:
    """Generate boolean Series indicating whether series contains target genre."""
    if not genre_term:
        return pd.Series([False] * len(series), index=series.index)
    import re
    s_lower = series.fillna("").astype(str).str.lower()
    target = genre_term.lower().strip()
    if target in ["science fiction", "sci-fi", "scifi"]:
        return s_lower.str.contains("science fiction|sci-fi", regex=True, na=False)
    elif target in ["romance", "romantic"]:
        return s_lower.str.contains("romance|romantic", regex=True, na=False)
    else:
        return s_lower.str.contains(re.escape(target), regex=True, na=False)


# ---------------------------------------------------------------------------
# 2. Retrieve candidates based on constraints
# ---------------------------------------------------------------------------
def retrieve_candidates(constraints: dict, top_k: int = 10) -> list:
    """
    Retrieve top-K movie candidates based on extracted constraints.
    
    Strategy:
    1. Hard filter: apply language_pref and genre exclusions to narrow the catalog
    2. Era filter: filter release_year (<=2005 for old, >=2016 for new) if candidates exist
    3. Genre filter: filter to requested genre if sufficient candidates exist
    4. Soft match: embed a synthetic query enriched with era, genre, mood, reference title
    5. FAISS candidate search + scoring bonuses for exact genre and era alignment
    
    Args:
        constraints: dict with keys genre, mood, language_pref, exclusions, reference_title, era
        top_k: Number of results to return
        
    Returns:
        List of dicts: {movie_id, title, overview, language, genres, release_year, similarity_score}
    """
    import re
    catalog = _load_catalog()
    all_embeddings = _load_embeddings()

    # --- Hard filtering ---
    mask = pd.Series([True] * len(catalog), index=catalog.index)

    # Filter by language preference
    lang_pref = constraints.get("language_pref", "any")
    if lang_pref != "any":
        mask &= (catalog["language"] == lang_pref)

    # Filter by exclusions (genre-based)
    exclusions = constraints.get("exclusions", "")
    excluded_genres = []
    if exclusions:
        excluded_genres = [e.strip().lower() for e in exclusions.split(",") if e.strip()]
        for exc in excluded_genres:
            mask &= ~catalog["genres"].str.lower().str.contains(re.escape(exc), na=False)

    # Filter by era (temporal constraint: old <= 2005, new >= 2016)
    era = constraints.get("era", "any")
    if era == "old":
        era_mask = mask & (catalog["release_year"] <= 2005)
        if era_mask.sum() >= min(top_k, 5):
            mask = era_mask
    elif era == "new":
        era_mask = mask & (catalog["release_year"] >= 2016)
        if era_mask.sum() >= min(top_k, 5):
            mask = era_mask

    # Filter by genre if sufficient matching titles exist
    genre = constraints.get("genre", "").strip()
    if genre:
        genre_mask = mask & _genre_match_mask(catalog["genres"], genre)
        if genre_mask.sum() >= min(top_k, 3):
            mask = genre_mask

    filtered_indices = catalog[mask].index.to_numpy()

    if len(filtered_indices) == 0:
        # Stepwise fallback: relax era and genre, retain exclusions and language
        fallback_mask = pd.Series([True] * len(catalog), index=catalog.index)
        if lang_pref != "any":
            fallback_mask &= (catalog["language"] == lang_pref)
        if exclusions:
            for exc in excluded_genres:
                fallback_mask &= ~catalog["genres"].str.lower().str.contains(re.escape(exc), na=False)
        filtered_indices = catalog[fallback_mask].index.to_numpy()

    if len(filtered_indices) == 0:
        # Ultimate fallback: relax language if needed
        print("[WARN] No movies match the hard constraints. Relaxing language filter...")
        mask = pd.Series([True] * len(catalog), index=catalog.index)
        if exclusions:
            for exc in excluded_genres:
                mask &= ~catalog["genres"].str.lower().str.contains(re.escape(exc), na=False)
        filtered_indices = catalog[mask].index.to_numpy()

    if len(filtered_indices) == 0:
        return []

    # --- Build synthetic query for semantic search ---
    query_parts = []
    if era == "old":
        query_parts.append("classic vintage 80s 90s")
    elif era == "new":
        query_parts.append("recent modern latest")

    mood = constraints.get("mood", "")
    ref_title = constraints.get("reference_title", "")

    if genre:
        query_parts.append(f"{genre} movie")
    if mood:
        query_parts.append(f"{mood} film")
    if ref_title:
        query_parts.append(f"similar to {ref_title}")
    if not query_parts:
        query_parts.append("good movie")

    synthetic_query = " ".join(query_parts)

    # Embed the synthetic query
    model = get_embedding_model()
    query_embedding = model.encode([synthetic_query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)

    # --- Use precomputed normalized embeddings for candidate search ---
    filtered_embeddings = all_embeddings[filtered_indices].copy()

    # Build temporary FAISS index for filtered candidate embeddings
    dim = filtered_embeddings.shape[1]
    temp_index = faiss.IndexFlatIP(dim)
    temp_index.add(filtered_embeddings)

    # Retrieve candidate pool for re-ranking
    search_k = min(max(top_k * 3, 20), len(filtered_indices))
    scores, indices = temp_index.search(query_embedding, search_k)

    # Build and re-rank results
    candidates = []
    for i in range(search_k):
        sub_idx = indices[0][i]
        if sub_idx == -1:
            continue
        orig_idx = filtered_indices[sub_idx]
        row = catalog.iloc[orig_idx]
        base_score = float(scores[0][i])
        boost = 0.0

        # Genre alignment bonus
        if genre and _is_genre_match(str(row["genres"]), genre):
            boost += 0.20

        # Era alignment bonus
        rel_year = row.get("release_year")
        parsed_year = None
        if pd.notna(rel_year):
            parsed_year = int(rel_year)
            if era == "old" and parsed_year <= 2005:
                boost += 0.15
            elif era == "new" and parsed_year >= 2016:
                boost += 0.10

        final_score = base_score + boost

        candidates.append({
            "movie_id": int(row["movie_id"]),
            "title": str(row["title"]),
            "overview": str(row["overview"]),
            "language": str(row["language"]),
            "genres": str(row["genres"]),
            "release_year": parsed_year,
            "similarity_score": round(final_score, 4),
        })

    # Sort candidates by boosted score descending
    candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
    return candidates[:top_k]


# ---------------------------------------------------------------------------
# Test block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Retrieval Module Test ===\n")

    # First build the index
    if not os.path.exists(FAISS_INDEX_PATH):
        print("Building index first...\n")
        build_index()
        print()

    # Test 1: Tamil action movies
    print("--- Test 1: Tamil Action Movies ---")
    constraints1 = {
        "genre": "action",
        "mood": "exciting",
        "language_pref": "ta",
        "exclusions": "",
        "reference_title": "",
    }
    results1 = retrieve_candidates(constraints1, top_k=5)
    for r in results1:
        print(f"  [{r['language']}] {r['title']} ({r['genres']}) — score: {r['similarity_score']:.4f}")
    print()

    # Test 2: English comedy movies
    print("--- Test 2: English Comedy Movies ---")
    constraints2 = {
        "genre": "comedy",
        "mood": "feel-good",
        "language_pref": "en",
        "exclusions": "",
        "reference_title": "",
    }
    results2 = retrieve_candidates(constraints2, top_k=5)
    for r in results2:
        print(f"  [{r['language']}] {r['title']} ({r['genres']}) — score: {r['similarity_score']:.4f}")
    print()

    # Test 3: Tamil horror (no comedy)
    print("--- Test 3: Tamil Horror (no comedy) ---")
    constraints3 = {
        "genre": "horror",
        "mood": "scary",
        "language_pref": "ta",
        "exclusions": "comedy",
        "reference_title": "",
    }
    results3 = retrieve_candidates(constraints3, top_k=5)
    for r in results3:
        print(f"  [{r['language']}] {r['title']} ({r['genres']}) — score: {r['similarity_score']:.4f}")
    print()
