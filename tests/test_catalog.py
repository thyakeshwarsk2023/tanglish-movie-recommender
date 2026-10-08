"""
tests/test_catalog.py — Verification of movie catalog integrity (Section 1)
"""

import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATALOG_PATH = os.path.join(PROJECT_ROOT, "data", "catalog.csv")


def test_catalog():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 70)
    print("  TEST 1: CATALOG DATA INTEGRITY")
    print("=" * 70)

    assert os.path.exists(CATALOG_PATH), f"Catalog not found at {CATALOG_PATH}"
    df = pd.read_csv(CATALOG_PATH)

    # 1. Non-empty Tamil subset
    ta_count = len(df[df.language == "ta"])
    print(f"\n[CHECK 1] Tamil Movie Count: {ta_count}")
    assert ta_count >= 50, f"Expected at least 50 Tamil movies, found {ta_count}"
    print("  PASS: Tamil catalog subset is non-trivial and populated.")

    # 2. Non-empty English subset
    en_count = len(df[df.language == "en"])
    print(f"\n[CHECK 2] English Movie Count: {en_count}")
    assert en_count >= 4000, f"Expected at least 4000 English movies, found {en_count}"
    print("  PASS: English catalog subset matches expected TMDB scale.")

    # 3. Duplicate check
    dup_count = df.duplicated(subset=["title"]).sum()
    print(f"\n[CHECK 3] Duplicate Titles Count: {dup_count}")
    assert dup_count == 0, f"Expected 0 duplicates, found {dup_count}"
    print("  PASS: Zero duplicate movie titles.")

    # 4. Missing critical fields
    missing_title = df["title"].isnull().sum()
    missing_overview = df["overview"].isnull().sum()
    print(f"\n[CHECK 4] Missing Titles: {missing_title}, Missing Overviews: {missing_overview}")
    assert missing_title == 0 and missing_overview == 0, "Found missing title or overview fields"
    print("  PASS: No missing critical fields.")

    # 5. Schema consistency
    expected_cols = {"movie_id", "title", "overview", "genres", "language", "release_year"}
    actual_cols = set(df.columns)
    print(f"\n[CHECK 5] Columns Present: {list(df.columns)}")
    assert expected_cols.issubset(actual_cols), f"Missing columns: {expected_cols - actual_cols}"
    assert pd.api.types.is_numeric_dtype(df["movie_id"]), "movie_id should be numeric"
    assert pd.api.types.is_numeric_dtype(df["release_year"]), "release_year should be numeric"
    print("  PASS: Schema strictly matches specifications.")

    # 6. Spot-checks: 10 Tamil movies
    print("\n[SPOT-CHECK] 10 Random Tamil Movies:")
    ta_samples = df[df.language == "ta"].sample(10, random_state=42)
    for i, (_, row) in enumerate(ta_samples.iterrows(), 1):
        excerpt = str(row["overview"])[:80].replace("\n", " ")
        print(f"  {i:2d}. {row['title']} ({int(row['release_year'])}) | {row['genres']}")
        print(f"      Overview: \"{excerpt}...\"")

    # 7. Spot-checks: 10 English movies
    print("\n[SPOT-CHECK] 10 Random English Movies:")
    en_samples = df[df.language == "en"].sample(10, random_state=42)
    for i, (_, row) in enumerate(en_samples.iterrows(), 1):
        excerpt = str(row["overview"])[:80].replace("\n", " ")
        print(f"  {i:2d}. {row['title']} ({int(row['release_year']) if pd.notna(row['release_year']) else 'N/A'}) | {row['genres']}")
        print(f"      Overview: \"{excerpt}...\"")

    print("\n" + "=" * 70)
    print("  ALL CATALOG INTEGRITY TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    test_catalog()
