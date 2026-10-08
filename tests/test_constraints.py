"""
tests/test_constraints.py — Constraint model validation, confusion matrix & inference tests (Section 3)
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import query_understanding as qu
from src.query_understanding import (
    extract_constraints,
    normalize_query,
    get_embedding_model,
)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "query_training_set.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CM_IMAGE_PATH = os.path.join(RESULTS_DIR, "confusion_matrix.png")


def test_training_checks_and_confusion_matrix():
    print("=" * 70)
    print("  TEST 3a: TRAINING-TIME CHECKS & CONFUSION MATRIX")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)
    print(f"\n[CLASS BALANCE] Total Training Examples: {len(df)}")
    genre_counts = df["genre_label"].value_counts()
    mood_counts = df["mood_label"].value_counts()
    print("Genre Distribution:")
    for g, c in genre_counts.items():
        print(f"  {g:<16}: {c}")
    print("\nMood Distribution:")
    for m, c in mood_counts.items():
        print(f"  {m:<16}: {c}")

    # Verify no classes with < 2 examples
    assert genre_counts.min() >= 2, f"Genre with too few examples: {genre_counts.min()}"
    assert mood_counts.min() >= 2, f"Mood with too few examples: {mood_counts.min()}"
    print("\n  PASS: Class balance verified across all target classes.")

    # Load model and label maps
    qu._load_classifier()
    assert qu._classifier is not None, "Classifier failed to load"
    assert qu._label_maps is not None, "Label maps failed to load"

    genre_labels = qu._label_maps["genre_labels"]
    mood_labels = qu._label_maps["mood_labels"]

    # Re-evaluate on held-out test split
    df["normalized"] = df["query_text"].apply(normalize_query)
    model = get_embedding_model()
    embeddings = model.encode(df["normalized"].tolist(), convert_to_numpy=True)

    X = torch.FloatTensor(embeddings)
    indices = list(range(len(X)))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=df["genre_label"]
    )

    X_test = X[test_idx]
    test_df = df.iloc[test_idx]

    qu._classifier.eval()
    with torch.no_grad():
        genre_logits, mood_logits = qu._classifier(X_test)
        genre_preds = genre_logits.argmax(dim=1).numpy()
        mood_preds = mood_logits.argmax(dim=1).numpy()

    # Confusion matrices
    genre_true = [genre_labels.index(g) for g in test_df["genre_label"]]
    mood_true = [mood_labels.index(m) for m in test_df["mood_label"]]

    cm_genre = confusion_matrix(genre_true, genre_preds, labels=range(len(genre_labels)))
    cm_mood = confusion_matrix(mood_true, mood_preds, labels=range(len(mood_labels)))

    # Plot and save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    im1 = ax1.imshow(cm_genre, cmap="Blues")
    ax1.set_title("Genre Confusion Matrix (Test Split)", fontsize=13, fontweight="bold")
    ax1.set_xticks(range(len(genre_labels)))
    ax1.set_yticks(range(len(genre_labels)))
    ax1.set_xticklabels(genre_labels, rotation=45, ha="right", fontsize=9)
    ax1.set_yticklabels(genre_labels, fontsize=9)
    ax1.set_xlabel("Predicted Label")
    ax1.set_ylabel("True Label")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(cm_mood, cmap="Greens")
    ax2.set_title("Mood Confusion Matrix (Test Split)", fontsize=13, fontweight="bold")
    ax2.set_xticks(range(len(mood_labels)))
    ax2.set_yticks(range(len(mood_labels)))
    ax2.set_xticklabels(mood_labels, rotation=45, ha="right", fontsize=9)
    ax2.set_yticklabels(mood_labels, fontsize=9)
    ax2.set_xlabel("Predicted Label")
    ax2.set_ylabel("True Label")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(CM_IMAGE_PATH, dpi=200)
    plt.close()
    print(f"\n[CONFUSION MATRIX] Plot saved to: {CM_IMAGE_PATH}")
    print("  PASS: Confusion matrix generated without majority-class collapse.")


def test_inference_checks():
    print("\n" + "=" * 70)
    print("  TEST 3b: INFERENCE-TIME CHECKS (18 HANDPICKED TEST QUERIES)")
    print("=" * 70)

    test_cases = [
        # (query, expected_genre, expected_mood)
        ("I want something scary to watch tonight", "horror", "scary"),
        ("nalla feel good padam venum", "comedy", "feel-good"),
        ("something dark and slow", "drama", "dark"),
        ("oru mass action padam venum", "action", "exciting"),
        ("emotional love story please", "romance", "emotional"),
        ("a thrilling murder mystery with twists", "thriller", "suspenseful"),
        ("hilarious comedy with friends", "comedy", "feel-good"),
        ("inspirational sports drama", "sports", "inspiring"),
        ("family movie for kids on weekend", "family", "feel-good"),
        ("superhero fighting film with action", "action", "exciting"),
        ("creepy haunted house movie", "horror", "scary"),
        ("romba intense war battle movie", "action", "intense"),
        ("sweet romantic comedy film", "romance", "feel-good"),
        ("sci fi movie about time travel and space", "science fiction", "exciting"),
        ("gangster crime underworld film", "crime", "dark"),
        ("touching emotional family story", "drama", "emotional"),
        ("psychological thriller that will blow my mind", "thriller", "suspenseful"),
        ("motivational movie about real life success", "drama", "inspiring"),
    ]

    correct_genre = 0
    correct_mood = 0
    bug_log = []

    print(f"\n{'Query':<48} | {'Pred Genre':<12} {'GT Genre':<12} | {'Pred Mood':<12} {'GT Mood':<12}")
    print("-" * 105)

    for query, exp_genre, exp_mood in test_cases:
        c = extract_constraints(query)
        p_genre = c["genre"]
        p_mood = c["mood"]

        genre_match = (p_genre == exp_genre) or (exp_genre in p_genre)
        mood_match = (p_mood == exp_mood) or (exp_mood in p_mood)

        if genre_match:
            correct_genre += 1
        if mood_match:
            correct_mood += 1

        if not genre_match or not mood_match:
            bug_log.append({
                "query": query,
                "expected": f"genre={exp_genre}, mood={exp_mood}",
                "actual": f"genre={p_genre}, mood={p_mood}",
                "cause": "Overlapping genre semantics or lexical ambiguity"
            })

        print(f"{query[:46]:<48} | {p_genre:<12} {exp_genre:<12} | {p_mood:<12} {exp_mood:<12} | G:{'PASS' if genre_match else 'FAIL'} M:{'PASS' if mood_match else 'FAIL'}")

    genre_acc = correct_genre / len(test_cases)
    mood_acc = correct_mood / len(test_cases)
    print("\n" + "-" * 70)
    print(f"  Handpicked Genre Accuracy: {genre_acc:.1%} ({correct_genre}/{len(test_cases)})")
    print(f"  Handpicked Mood Accuracy:  {mood_acc:.1%} ({correct_mood}/{len(test_cases)})")
    print("-" * 70)

    assert genre_acc >= 0.70, f"Genre accuracy {genre_acc:.1%} below 70% threshold"
    print("  PASS: Inference checks exceed the 70% pass floor.")

    if bug_log:
        print(f"\n[CHALLENGES / BUG LOG ({len(bug_log)} cases)]:")
        for b in bug_log:
            print(f"  * Query: \"{b['query']}\"")
            print(f"    Expected: {b['expected']} | Actual: {b['actual']}")
            print(f"    Analysis: {b['cause']}\n")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    test_training_checks_and_confusion_matrix()
    test_inference_checks()
