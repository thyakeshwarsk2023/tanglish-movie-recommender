"""
tests/test_normalization.py — Unit tests for query normalization (Section 2)
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.query_understanding import normalize_query


def test_normalization():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 70)
    print("  TEST 2: QUERY NORMALIZATION TESTS")
    print("=" * 70)

    # Test Case 1: Core Tanglish substitution
    input_1 = "nalla comedy padam venum"
    output_1 = normalize_query(input_1)
    print(f"\n[CASE 1] Input:  \"{input_1}\"")
    print(f"         Output: \"{output_1}\"")
    assert "good" in output_1 and "movie" in output_1 and "want" in output_1, f"Failed on case 1: {output_1}"
    print("  PASS: Core dictionary substitutions work.")

    # Test Case 2: Multi-word / intensifier substitutions
    input_2 = "romba pazhaya Tamil action padam"
    output_2 = normalize_query(input_2)
    print(f"\n[CASE 2] Input:  \"{input_2}\"")
    print(f"         Output: \"{output_2}\"")
    assert "very" in output_2 and "old" in output_2 and "movie" in output_2, f"Failed on case 2: {output_2}"
    print("  PASS: Multi-word substitutions handled.")

    # Test Case 3: Pure English query preserved
    input_3 = "recommend me a good thriller"
    output_3 = normalize_query(input_3)
    print(f"\n[CASE 3] Input:  \"{input_3}\"")
    print(f"         Output: \"{output_3}\"")
    assert output_3 == "recommend me a good thriller", f"Failed on case 3: {output_3}"
    print("  PASS: English input preserved without distortion.")

    # Test Case 4: Tanglish word not in dictionary passes through
    input_4 = "oru kalyana comedy cinema venum"
    output_4 = normalize_query(input_4)
    print(f"\n[CASE 4] Input:  \"{input_4}\"")
    print(f"         Output: \"{output_4}\"")
    assert "kalyana" in output_4 and "movie" in output_4 and "want" in output_4, f"Failed on case 4: {output_4}"
    print("  PASS: Out-of-vocabulary terms pass through gracefully.")

    # Test Case 5: Empty string
    input_5 = ""
    output_5 = normalize_query(input_5)
    print(f"\n[CASE 5] Input:  \"{input_5}\"")
    print(f"         Output: \"{output_5}\"")
    assert output_5 == "", f"Failed on case 5: expected empty string, got '{output_5}'"
    print("  PASS: Empty string handled without exception.")

    # Test Case 6: Mixed-script Unicode Tamil + English
    input_6 = "நல்ல comedy படம் வேணும்"
    output_6 = normalize_query(input_6)
    print(f"\n[CASE 6] Input:  \"{input_6}\"")
    print(f"         Output: \"{output_6}\"")
    assert isinstance(output_6, str) and len(output_6) > 0, "Failed on mixed script"
    print("  PASS: Unicode Tamil mixed-script handled without crash.")

    print("\n" + "=" * 70)
    print("  ALL QUERY NORMALIZATION TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    test_normalization()
