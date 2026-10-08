"""
tests/run_all_tests.py — Master test suite runner for Tanglish Movie Recommender
Executes all 6 testing stages and reports a consolidated scorecard.
"""

import sys
import subprocess
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_EXE = sys.executable

TEST_SUITES = [
    ("Catalog Data Integrity (Section 1)", "tests/test_catalog.py"),
    ("Query Normalization Unit Tests (Section 2)", "tests/test_normalization.py"),
    ("Constraint Extraction & Confusion Matrix (Section 3)", "tests/test_constraints.py"),
    ("FAISS Cross-Lingual Retrieval & Latency (Section 4)", "tests/test_retrieval.py"),
    ("Recommendation Pipeline & Grounding (Section 5)", "tests/test_recommend.py"),
    ("Benchmark Evaluation & Hit Rate Audit (Section 6)", "src/evaluate.py"),
]


def run_master_suite():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 80)
    print("  TANGLISH MOVIE RECOMMENDER — MASTER TEST RUNNER")
    print("=" * 80)
    print(f"  Python Environment: {PYTHON_EXE}")
    print(f"  Project Root:       {PROJECT_ROOT}")
    print("=" * 80)

    scorecard = []

    for name, script_relpath in TEST_SUITES:
        script_full = os.path.join(PROJECT_ROOT, script_relpath)
        print(f"\n>>> Running: {name} ...")
        cmd = [PYTHON_EXE, script_full]
        
        proc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace"
        )

        status = "PASSED" if proc.returncode == 0 else "FAILED"
        scorecard.append((name, status, proc.returncode))

        # Print last few lines of output
        lines = [line for line in proc.stdout.strip().split("\n") if line.strip()]
        tail = lines[-8:] if len(lines) >= 8 else lines
        for l in tail:
            print(f"    {l}")

        if proc.returncode != 0:
            print(f"    [STDERR]: {proc.stderr[:300]}")

    print("\n" + "=" * 80)
    print("  CONSOLIDATED TEST SCORECARD")
    print("=" * 80)
    print(f"  {'Component / Suite':<55} | {'Result':<10}")
    print(f"  {'-' * 55}-+-----------")
    all_passed = True
    for name, status, code in scorecard:
        print(f"  {name:<55} | {status}")
        if status != "PASSED":
            all_passed = False

    print(f"  {'-' * 55}-+-----------")
    overall = "ALL TESTS PASSED SUCCESSFULLY! (100%)" if all_passed else "SOME TESTS FAILED."
    print(f"  OVERALL: {overall}")
    print("=" * 80)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(run_master_suite())
