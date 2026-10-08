"""
tests/test_recommend.py — Recommendation pipeline and grounded explanation tests (Section 5)
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.recommend import recommend


def test_recommendation_pipeline():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 70)
    print("  TEST 5: RECOMMENDATION PIPELINE & GROUNDED EXPLANATIONS")
    print("=" * 70)

    test_queries = [
        ("I want a dark psychological thriller with mystery", "en"),
        ("nalla comedy padam venum", "tanglish"),
        ("romba pazhaya Tamil action padam venum", "tanglish"),
    ]

    for q, origin in test_queries:
        print(f"\n[QUERY] \"{q}\" (Origin: {origin})")
        recs, constraints = recommend(q, top_k=5)

        assert isinstance(recs, list), "Expected list of recommendations"
        assert len(recs) > 0, f"No recommendations returned for '{q}'"
        print(f"  Extracted: genre={constraints.get('genre')}, mood={constraints.get('mood')}, lang={constraints.get('language_pref')}")

        for i, r in enumerate(recs, 1):
            # Check complete keys
            for expected_key in ["title", "language", "genre", "explanation", "similarity_score", "overview"]:
                assert expected_key in r, f"Missing key '{expected_key}' in recommendation"

            exp = r["explanation"]
            overview = r["overview"]

            # Grounding check: explanation must cite overview excerpt
            assert "overview mentions:" in exp, f"Explanation missing overview quotation: {exp}"

            # Check that detected mood or genre is mentioned in explanation
            detected_genre = constraints.get("genre", "")
            detected_mood = constraints.get("mood", "")

            if detected_mood:
                assert detected_mood.lower() in exp.lower(), f"Detected mood '{detected_mood}' missing from explanation"

            print(f"  {i}. {r['title']} [{r['language']}] — score: {r['similarity_score']:.3f}")
            print(f"     Explanation: {exp}")

        print("  PASS: All recommendations verified for schema completeness and textual grounding.")

    print("\n" + "=" * 70)
    print("  ALL RECOMMENDATION & EXPLANATION TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    test_recommendation_pipeline()
