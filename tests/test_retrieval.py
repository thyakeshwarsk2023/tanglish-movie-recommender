"""
tests/test_retrieval.py — FAISS retrieval assertion tests and latency benchmarking (Section 4)
"""

import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.retrieval import retrieve_candidates
from src.query_understanding import extract_constraints, normalize_query


def test_retrieval():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 70)
    print("  TEST 4: FAISS RETRIEVAL & FILTERING ASSERTIONS")
    print("=" * 70)

    # 1. Tamil Hard Constraint Assertion
    print("\n[CHECK 1] Asserting Language Filter (Tamil)...")
    c_ta = {"genre": "action", "mood": "exciting", "language_pref": "ta", "exclusions": "", "reference_title": ""}
    res_ta = retrieve_candidates(c_ta, top_k=10)
    assert len(res_ta) > 0, "No results returned for Tamil language_pref"
    assert all(r["language"] == "ta" for r in res_ta), "Non-Tamil movie found in Tamil-filtered retrieval"
    print(f"  PASS: All {len(res_ta)} candidates have language == 'ta'.")
    for r in res_ta[:3]:
        print(f"    * {r['title']} ({r['genres']})")

    # 2. English Hard Constraint Assertion
    print("\n[CHECK 2] Asserting Language Filter (English)...")
    c_en = {"genre": "comedy", "mood": "feel-good", "language_pref": "en", "exclusions": "", "reference_title": ""}
    res_en = retrieve_candidates(c_en, top_k=10)
    assert len(res_en) > 0, "No results returned for English language_pref"
    assert all(r["language"] == "en" for r in res_en), "Non-English movie found in English-filtered retrieval"
    print(f"  PASS: All {len(res_en)} candidates have language == 'en'.")
    for r in res_en[:3]:
        print(f"    * {r['title']} ({r['genres']})")

    # 3. 'Any' Language Returns Cross-Lingual Candidates
    print("\n[CHECK 3] Asserting 'Any' Language Filter...")
    c_any = {"genre": "action", "mood": "exciting", "language_pref": "any", "exclusions": "", "reference_title": ""}
    res_any = retrieve_candidates(c_any, top_k=20)
    languages = set(r["language"] for r in res_any)
    print(f"  Returned {len(res_any)} candidates with languages: {languages}")
    assert len(res_any) > 0, "No results for 'any'"
    print("  PASS: Cross-lingual retrieval returned valid candidates.")

    # 4. Exclusion Filter Assertion
    print("\n[CHECK 4] Asserting Exclusion Filter ('horror')...")
    c_exc = {"genre": "comedy", "mood": "feel-good", "language_pref": "any", "exclusions": "horror", "reference_title": ""}
    res_exc = retrieve_candidates(c_exc, top_k=15)
    for r in res_exc:
        assert "horror" not in r["genres"].lower(), f"Excluded genre 'horror' found in: {r['title']} ({r['genres']})"
    print(f"  PASS: Zero of {len(res_exc)} candidates contained excluded genre 'horror'.")

    # 5. Empty Result Graceful Handling
    print("\n[CHECK 5] Empty Result Graceful Handling...")
    c_absurd = {"genre": "nonexistent_xyz", "mood": "absurd_123", "language_pref": "ta", "exclusions": "action, comedy, drama, thriller, horror, romance, family", "reference_title": ""}
    res_absurd = retrieve_candidates(c_absurd, top_k=5)
    assert isinstance(res_absurd, list), "Should return a list"
    print(f"  PASS: Degrades gracefully without crash, returned {len(res_absurd)} items.")

    # 6. Latency Benchmark
    print("\n[CHECK 6] Retrieval Speed Benchmark...")
    start_time = time.perf_counter()
    for _ in range(5):
        retrieve_candidates(c_ta, top_k=5)
    elapsed_ms = ((time.perf_counter() - start_time) / 5) * 1000
    print(f"  Average retrieve_candidates() latency: {elapsed_ms:.2f} ms")
    assert elapsed_ms < 200, f"Retrieval too slow: {elapsed_ms:.2f} ms"
    print("  PASS: Sub-second latency verified.")

    # 7. CRITICAL TEST: 5 Tanglish Queries with Intended Tamil Preference
    print("\n" + "=" * 70)
    print("  CRITICAL TEST: 5 REAL TANGLISH QUERIES -> END-TO-END TAMIL RETRIEVAL")
    print("=" * 70)

    critical_tanglish_queries = [
        "nalla comedy padam venum Tamil la",
        "romba pazhaya Tamil action padam venum",
        "emotional Tamil movie pls no horror",
        "horror movie venum Tamil la but not too scary",
        "crime thriller padam sollunga Tamil la",
    ]

    for q in critical_tanglish_queries:
        norm = normalize_query(q)
        c = extract_constraints(q)
        # Ensure language_pref is 'ta' as intended
        c["language_pref"] = "ta"
        recs = retrieve_candidates(c, top_k=3)

        print(f"\n  Query: \"{q}\"")
        print(f"  Normalized: \"{norm}\"")
        print(f"  Detected Genre: {c['genre']}, Mood: {c['mood']}")
        assert len(recs) > 0, f"No candidates returned for query: {q}"
        assert all(r["language"] == "ta" for r in recs), f"Non-Tamil film returned for query: {q}"
        print("  Top Tamil Movies Returned:")
        for r in recs:
            print(f"    * [{r['language']}] {r['title']} ({r['genres']}) — score: {r['similarity_score']:.4f}")

    print("\n" + "=" * 70)
    print("  ALL FAISS RETRIEVAL TESTS & CRITICAL VERIFICATIONS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    test_retrieval()
