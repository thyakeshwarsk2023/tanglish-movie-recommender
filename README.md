# Tanglish Movie Recommender

A voice/text movie recommender that understands Tanglish (code-mixed Tamil-English) queries and recommends both English and Tamil films, with structured constraint extraction and cross-lingual retrieval.

## Novel Contribution

Unlike standard content-based recommenders that treat the query as an unstructured embedding target, this system explicitly extracts structured constraints (genre, mood, language preference, exclusions) from natural-language movie requests — including code-mixed Tanglish queries — before retrieval. It further extends the recommendation catalog itself to be cross-lingual (English and Tamil films in one retrieval space), and evaluates constraint-extraction accuracy separately for English-origin and Tanglish-origin queries to measure the real performance gap introduced by code-mixed input.

## Architecture

```
User Query (English/Tanglish)
        │
        ▼
┌───────────────────────┐
│  normalize_query()    │  Tanglish word-level substitution
│  (dictionary-based)   │
└───────┬───────────────┘
        │
        ▼
┌───────────────────────┐
│  extract_constraints()│  Multilingual embeddings + trained classifier
│  (neural + rules)     │  → genre, mood, language_pref, exclusions
└───────┬───────────────┘
        │
        ▼
┌───────────────────────┐
│  retrieve_candidates()│  FAISS cross-lingual semantic search
│  (hard filter + FAISS)│  on English + Tamil catalog
└───────┬───────────────┘
        │
        ▼
┌───────────────────────┐
│  recommend()          │  Template-based grounded explanations
│  (pipeline + explain) │
└───────────────────────┘
```

## Setup Instructions

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

If you encounter issues with torch on Windows, install CPU-only:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 2. Prepare Raw Data

Place your TMDB movies CSV in `data/raw/`:
```bash
# Download tmdb_5000_movies.csv from Kaggle and place it at:
# data/raw/tmdb_5000_movies.csv
```

### 3. Build the Catalog

```bash
python src/data_prep.py
```

This creates `data/catalog.csv` with ~4,600 movies (English + Tamil).

### 4. Train the Constraint Model

```bash
python train_constraint_model.py
```

This trains the genre/mood classifier and saves it to `models/constraint_model.pt`.

### 5. Build the FAISS Index

```bash
python -c "from src.retrieval import build_index; build_index()"
```

### 6. Run the App

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### 7. Run Evaluation

```bash
python src/evaluate.py
```

This evaluates constraint extraction accuracy on the 30-query test set and reports English vs Tanglish accuracy gap.

## Project Structure

```
tanglish-movie-recommender/
├── data/
│   ├── raw/                          # Raw data files (not committed)
│   │   └── tmdb_5000_movies.csv
│   ├── catalog.csv                   # Unified movie catalog
│   ├── tanglish_test_queries.csv     # 30 evaluation queries
│   └── query_training_set.csv        # 100 training examples
├── src/
│   ├── __init__.py
│   ├── data_prep.py                  # Catalog builder
│   ├── query_understanding.py        # Normalization + constraint extraction
│   ├── retrieval.py                  # FAISS cross-lingual retrieval
│   ├── recommend.py                  # Full recommendation pipeline
│   └── evaluate.py                   # Evaluation script
├── models/
│   ├── constraint_model.pt           # Trained genre/mood classifier
│   ├── label_maps.pt                 # Label encodings
│   ├── faiss_index.bin               # FAISS index
│   └── id_map.pkl                    # Index→movie_id mapping
├── results/
│   └── evaluation_results.csv        # Detailed evaluation output
├── train_constraint_model.py         # Training script
├── app.py                            # Streamlit web UI
├── requirements.txt
├── .gitignore
└── README.md
```

## Tech Stack

- **Embeddings**: paraphrase-multilingual-MiniLM-L12-v2 (50+ languages)
- **Classification**: PyTorch feed-forward neural network
- **Retrieval**: FAISS (CPU) with cosine similarity
- **UI**: Streamlit
- **Data**: TMDB 5000 (English) + curated Tamil catalog (~100 films)

## Key Design Decisions

1. **Same embedding model for queries and catalog**: Both use `paraphrase-multilingual-MiniLM-L12-v2` so that English queries, Tanglish queries, and Tamil/English movie descriptions all live in the same 384-dimensional vector space.

2. **Hybrid constraint extraction**: Neural classifier for genre/mood (trained on labeled data) + rule-based for language preference and exclusions (more reliable for clear keyword patterns).

3. **Curated Tamil catalog**: The TMDB 5000 dataset contains only 2 Tamil films. We supplement with ~100 manually curated Tamil movies with real plot summaries to ensure meaningful cross-lingual retrieval.
