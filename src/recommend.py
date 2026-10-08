"""
recommend.py — Full recommendation pipeline

Orchestrates:
1. Query normalization (Tanglish -> English)
2. Constraint extraction (genre, mood, language, exclusions, reference title)
3. Candidate retrieval (FAISS cross-lingual search)
4. Explanation generation (template-based, grounded in movie overviews)
"""

import os
import sys

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.query_understanding import normalize_query, extract_constraints
from src.retrieval import retrieve_candidates


def _generate_explanation(movie: dict, constraints: dict) -> str:
    """
    Generate a short, grounded explanation for why a movie was recommended.
    Uses template-filling with actual movie overview content, NOT an external LLM.
    
    Args:
        movie: Dict with title, overview, genres, language, release_year, etc.
        constraints: Dict with genre, mood, language_pref, era, etc.
        
    Returns:
        A human-readable explanation string.
    """
    genre = constraints.get("genre", "")
    mood = constraints.get("mood", "")
    era = constraints.get("era", "any")
    overview = movie.get("overview", "")
    movie_genres = movie.get("genres", "")
    year = movie.get("release_year")

    # Get a short excerpt from overview (first ~100 chars)
    if len(overview) > 100:
        excerpt = overview[:100].rsplit(" ", 1)[0] + "..."
    else:
        excerpt = overview

    # Build explanation
    parts = []
    year_str = f" ({year})" if year else ""
    parts.append(f"Recommended because it's a {movie_genres} film{year_str}")

    if mood:
        parts[0] += f" matching your request for a {mood} experience"

    if genre and (genre.lower() in movie_genres.lower() or (genre.lower() in ["science fiction", "sci-fi"] and "sci" in movie_genres.lower())):
        parts.append(f"matches your preferred genre ({genre})")

    if era == "old" and year and year <= 2005:
        parts.append(f"released in {year} matching your preference for classic cinema")
    elif era == "new" and year and year >= 2016:
        parts.append(f"released in {year} matching your request for recent releases")

    lang_label = "Tamil" if movie.get("language") == "ta" else "English"
    parts.append(f"language: {lang_label}")

    parts.append(f'overview mentions: "{excerpt}"')

    return ". ".join(parts) + "."


def recommend(query_text: str, top_k: int = 5) -> list:
    """
    Full recommendation pipeline: query -> constraints -> retrieval -> explanations.
    
    Args:
        query_text: Raw user query (English or Tanglish)
        top_k: Number of recommendations to return
        
    Returns:
        List of dicts: {title, language, genre, release_year, explanation, similarity_score, overview}
    """
    # Step 1: Normalize
    normalized = normalize_query(query_text)

    # Step 2: Extract constraints
    constraints = extract_constraints(query_text)

    # Step 3: Retrieve candidates
    candidates = retrieve_candidates(constraints, top_k=top_k * 2)  # Get extra for fallback

    # Step 4: Build results with explanations
    results = []
    for movie in candidates[:top_k]:
        explanation = _generate_explanation(movie, constraints)
        results.append({
            "title": movie["title"],
            "language": movie["language"],
            "genre": movie["genres"],
            "release_year": movie.get("release_year"),
            "explanation": explanation,
            "similarity_score": movie["similarity_score"],
            "overview": movie["overview"],
        })

    return results, constraints


# ---------------------------------------------------------------------------
# Test block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 70)
    print("  RECOMMENDATION PIPELINE TEST")
    print("=" * 70)

    test_queries = [
        "nalla comedy padam venum",
        "I want a dark thriller movie with suspense",
        "romba pazhaya Tamil action padam venum",
    ]

    for query in test_queries:
        print(f"\n{'-' * 60}")
        print(f"  Query: {query}")
        print(f"{'-' * 60}")

        results, constraints = recommend(query)

        print(f"  Normalized: {normalize_query(query)}")
        print(f"  Constraints: {constraints}")
        print()

        if not results:
            print("  [No recommendations found]")
            continue

        for i, rec in enumerate(results, 1):
            lang_label = "[Tamil]" if rec["language"] == "ta" else "[English]"
            print(f"  {i}. {rec['title']} {lang_label}")
            print(f"     Genre: {rec['genre']}")
            print(f"     Score: {rec['similarity_score']:.4f}")
            print(f"     {rec['explanation']}")
            print()
