"""
train_constraint_model.py — Train the genre/mood classifier on labeled query data

This script:
1. Loads data/query_training_set.csv (labeled queries with genre and mood)
2. Embeds each query using the multilingual sentence-transformer
3. Trains a feed-forward neural network to classify genre and mood
4. Saves the trained model to models/constraint_model.pt
5. Prints accuracy on a held-out test split

Usage:
    python train_constraint_model.py
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Setup paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "query_training_set.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "constraint_model.pt")
LABEL_MAPS_PATH = os.path.join(MODEL_DIR, "label_maps.pt")

from src.query_understanding import ConstraintClassifier, normalize_query, get_embedding_model


def main():
    # -------------------------------------------------------------------
    # 1. Load training data
    # -------------------------------------------------------------------
    print("[1/5] Loading training data...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded {len(df)} training examples")
    print(f"  Genre labels: {df['genre_label'].nunique()} unique — "
          f"{df['genre_label'].value_counts().to_dict()}")
    print(f"  Mood labels:  {df['mood_label'].nunique()} unique — "
          f"{df['mood_label'].value_counts().to_dict()}")
    print()

    # -------------------------------------------------------------------
    # 2. Normalize queries and encode labels
    # -------------------------------------------------------------------
    print("[2/5] Normalizing queries and encoding labels...")
    df["normalized"] = df["query_text"].apply(normalize_query)

    genre_encoder = LabelEncoder()
    mood_encoder = LabelEncoder()

    df["genre_idx"] = genre_encoder.fit_transform(df["genre_label"])
    df["mood_idx"] = mood_encoder.fit_transform(df["mood_label"])

    genre_labels = list(genre_encoder.classes_)
    mood_labels = list(mood_encoder.classes_)

    print(f"  Genre classes: {genre_labels}")
    print(f"  Mood classes:  {mood_labels}")
    print()

    # -------------------------------------------------------------------
    # 3. Embed with multilingual sentence-transformer
    # -------------------------------------------------------------------
    print("[3/5] Embedding queries with paraphrase-multilingual-MiniLM-L12-v2...")
    embedding_model = get_embedding_model()
    embeddings = embedding_model.encode(
        df["normalized"].tolist(), show_progress_bar=True, convert_to_numpy=True
    )
    print(f"  Embedding shape: {embeddings.shape}")
    print()

    # -------------------------------------------------------------------
    # 4. Train/test split and training
    # -------------------------------------------------------------------
    print("[4/5] Training classifier...")

    X = torch.FloatTensor(embeddings)
    y_genre = torch.LongTensor(df["genre_idx"].values)
    y_mood = torch.LongTensor(df["mood_idx"].values)

    # 80/20 split
    indices = list(range(len(X)))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=df["genre_label"]
    )

    X_train, X_test = X[train_idx], X[test_idx]
    y_genre_train, y_genre_test = y_genre[train_idx], y_genre[test_idx]
    y_mood_train, y_mood_test = y_mood[train_idx], y_mood[test_idx]

    print(f"  Train size: {len(train_idx)}, Test size: {len(test_idx)}")

    # Create model
    model = ConstraintClassifier(
        input_dim=embeddings.shape[1],
        hidden_dim=128,
        num_genres=len(genre_labels),
        num_moods=len(mood_labels),
    )

    # Training parameters
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    genre_loss_fn = nn.CrossEntropyLoss()
    mood_loss_fn = nn.CrossEntropyLoss()
    num_epochs = 200

    best_acc = 0.0
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        genre_logits, mood_logits = model(X_train)
        loss = genre_loss_fn(genre_logits, y_genre_train) + mood_loss_fn(
            mood_logits, y_mood_train
        )

        loss.backward()
        optimizer.step()

        # Evaluate every 20 epochs
        if (epoch + 1) % 20 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                g_logits, m_logits = model(X_test)
                g_preds = torch.argmax(g_logits, dim=1)
                m_preds = torch.argmax(m_logits, dim=1)
                g_acc = (g_preds == y_genre_test).float().mean().item()
                m_acc = (m_preds == y_mood_test).float().mean().item()
                combined = (g_acc + m_acc) / 2

            if combined > best_acc:
                best_acc = combined
                best_state = model.state_dict().copy()

            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(
                    f"  Epoch {epoch+1:4d} | Loss: {loss.item():.4f} | "
                    f"Genre Acc: {g_acc:.2%} | Mood Acc: {m_acc:.2%}"
                )

    # Restore best model
    if best_state:
        model.load_state_dict(best_state)

    print()

    # -------------------------------------------------------------------
    # 5. Final evaluation and save
    # -------------------------------------------------------------------
    print("[5/5] Final evaluation and saving model...")

    model.eval()
    with torch.no_grad():
        g_logits, m_logits = model(X_test)
        g_preds = torch.argmax(g_logits, dim=1)
        m_preds = torch.argmax(m_logits, dim=1)

    genre_acc = (g_preds == y_genre_test).float().mean().item()
    mood_acc = (m_preds == y_mood_test).float().mean().item()

    print(f"\n=== Final Test Accuracy ===")
    print(f"  Genre accuracy: {genre_acc:.2%}")
    print(f"  Mood accuracy:  {mood_acc:.2%}")
    print(f"  Combined:       {(genre_acc + mood_acc) / 2:.2%}")
    print(f"===========================\n")

    # Per-class genre accuracy
    print("  Genre per-class accuracy:")
    for i, label in enumerate(genre_labels):
        mask = y_genre_test == i
        if mask.sum() > 0:
            class_acc = (g_preds[mask] == y_genre_test[mask]).float().mean().item()
            print(f"    {label:20s}: {class_acc:.2%} ({mask.sum().item()} samples)")

    print()
    print("  Mood per-class accuracy:")
    for i, label in enumerate(mood_labels):
        mask = y_mood_test == i
        if mask.sum() > 0:
            class_acc = (m_preds[mask] == y_mood_test[mask]).float().mean().item()
            print(f"    {label:20s}: {class_acc:.2%} ({mask.sum().item()} samples)")

    # Save model and label maps
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    torch.save(
        {"genre_labels": genre_labels, "mood_labels": mood_labels}, LABEL_MAPS_PATH
    )

    print(f"\n  Model saved to: {MODEL_PATH}")
    print(f"  Label maps saved to: {LABEL_MAPS_PATH}")
    print(f"\n[DONE] Training complete.")


if __name__ == "__main__":
    main()
