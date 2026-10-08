"""
query_understanding.py — Tanglish query normalization and constraint extraction

This module handles:
1. Tanglish (code-mixed Tamil-English) normalization via dictionary-based word substitution
2. Structured constraint extraction (genre, mood, language_pref, exclusions, reference_title)
   using a multilingual sentence-transformer + trained feed-forward classifier

WHY paraphrase-multilingual-MiniLM-L12-v2?
  This model was specifically chosen because it maps 50+ languages (including English
  and Tamil/Tanglish transliterated text) into a shared 384-dimensional embedding space.
  This means that code-mixed Tanglish queries like "nalla comedy padam venum" and
  pure English queries like "good comedy movie" end up near each other in the same
  vector space — critical for both the classifier and the cross-lingual retrieval
  to work correctly on mixed-language input.
"""

import os
import re
import torch
import torch.nn as nn
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
CONSTRAINT_MODEL_PATH = os.path.join(MODEL_DIR, "constraint_model.pt")
LABEL_MAPS_PATH = os.path.join(MODEL_DIR, "label_maps.pt")

# ---------------------------------------------------------------------------
# Multilingual embedding model (loaded lazily)
# ---------------------------------------------------------------------------
_embedding_model = None


def get_embedding_model():
    """
    Lazy-load the multilingual sentence-transformer model.
    Uses paraphrase-multilingual-MiniLM-L12-v2 which supports 50+ languages
    including English and Tamil (in transliterated form), mapping them into
    the same 384-dimensional embedding space.
    """
    global _embedding_model
    if _embedding_model is None:
        local_model_path = os.path.join(MODEL_DIR, "paraphrase-multilingual-MiniLM-L12-v2")
        model_name_or_path = local_model_path if os.path.exists(local_model_path) else "paraphrase-multilingual-MiniLM-L12-v2"
        _embedding_model = SentenceTransformer(model_name_or_path)
    return _embedding_model



# ---------------------------------------------------------------------------
# 1. Tanglish normalization — dictionary-based word substitution
# ---------------------------------------------------------------------------
TANGLISH_MAP = {
    # Core request words
    "venum": "want",
    "vendum": "want",
    "sollunga": "suggest",
    "sollunge": "suggest",
    "pannunga": "do",
    "pannunge": "do",
    "kudunga": "give",
    "kudunge": "give",
    "irukanum": "should be",
    "irukku": "is",
    "iruka": "is there",
    "irukkakoodathu": "should not be",
    # Movie/film terms
    "padam": "movie",
    "padham": "movie",
    "pada": "movie",
    "cinema": "movie",
    "film": "movie",
    "kathai": "story",
    # Adjectives / intensifiers
    "nalla": "good",
    "nallaa": "good",
    "romba": "very",
    "rompa": "very",
    "pazhaya": "old",
    "pazhaiya": "old",
    "pudhusa": "new",
    "pudhusu": "new",
    "pudhiya": "new",
    "periya": "big",
    "chinna": "small",
    # Mood / feeling terms
    "bayangara": "scary",
    "bayam": "fear",
    "siripu": "funny",
    "sirippu": "funny",
    "azhugai": "crying",
    "azhukkai": "sad",
    "kadhal": "love",
    "kaadhal": "love",
    "mass": "great",
    "maari": "like",
    "maathiri": "like",
    "mathiri": "like",
    # Negation
    "illa": "no",
    "illai": "no",
    "illama": "without",
    "venda": "do not want",
    "vendaam": "do not want",
    # Language references
    "tamizh": "tamil",
    "thamizh": "tamil",
    # Genre terms in Tanglish
    "sadhuru": "adventure",
    "payangara": "horror",
    "pazhi": "revenge",
    # Other common terms
    "oru": "one",
    "la": "in",
    "poi": "go",
    "pakura": "watch",
    "pakka": "watch",
}


def normalize_query(text: str) -> str:
    """
    Normalize a Tanglish query by substituting common Tamil/Tanglish words
    with their English equivalents. Uses simple word-level substitution.
    
    Args:
        text: The raw query text (possibly code-mixed Tamil-English)
    
    Returns:
        Normalized English text
    """
    text = text.lower().strip()
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    words = text.split()
    normalized = []
    for word in words:
        # Strip punctuation for matching but preserve structure
        clean = re.sub(r"[^\w\-]", "", word)
        if clean in TANGLISH_MAP:
            normalized.append(TANGLISH_MAP[clean])
        else:
            normalized.append(word)

    return " ".join(normalized)


# ---------------------------------------------------------------------------
# 2. Constraint extraction — hybrid (trained classifier + rules)
# ---------------------------------------------------------------------------

# Genre keywords for rule-based detection (fallback & exclusion detection)
GENRE_KEYWORDS = {
    "action": ["action", "fight", "battle", "martial arts", "explosions", "war",
               "combat", "superhero", "mass"],
    "comedy": ["comedy", "funny", "humor", "humour", "hilarious", "laugh",
               "funny", "comedic", "siripu", "sirippu"],
    "drama": ["drama", "dramatic", "serious", "emotional", "touching", "powerful"],
    "horror": ["horror", "scary", "ghost", "haunted", "terrifying", "creepy",
               "supernatural", "bayangara", "payangara", "bayam"],
    "romance": ["romance", "romantic", "love", "love story", "kadhal", "kaadhal"],
    "thriller": ["thriller", "suspense", "mystery", "suspenseful", "twist",
                 "crime", "detective", "investigation", "gripping"],
    "science fiction": ["sci-fi", "science fiction", "futuristic", "robot",
                        "time travel", "space", "alien", "ai"],
    "family": ["family", "kids", "children", "animation", "family-friendly",
               "wholesome"],
    "crime": ["crime", "gangster", "mafia", "heist", "robbery", "criminal"],
    "sports": ["sports", "boxing", "cricket", "football", "athlete", "underdog"],
    "adventure": ["adventure", "treasure", "quest", "journey", "explore"],
}

# Mood keywords
MOOD_KEYWORDS = {
    "exciting": ["exciting", "thrilling", "adrenaline", "mass", "blockbuster",
                 "great", "amazing", "awesome"],
    "feel-good": ["feel-good", "feel good", "lighthearted", "light", "uplifting",
                  "cheerful", "happy", "fun", "entertaining"],
    "dark": ["dark", "gritty", "noir", "bleak", "grim", "moody"],
    "emotional": ["emotional", "touching", "moving", "cry", "sad", "heartfelt",
                  "poignant", "deep"],
    "scary": ["scary", "frightening", "terrifying", "creepy", "chilling",
              "horror", "fear", "bayam"],
    "suspenseful": ["suspenseful", "suspense", "twist", "mystery", "mind-bending",
                    "gripping", "unpredictable", "edge"],
    "inspiring": ["inspiring", "inspirational", "motivational", "uplifting",
                  "empowering", "triumph", "overcome"],
    "intense": ["intense", "nail-biting", "relentless", "gripping", "raw",
                "powerful"],
    "romantic": ["romantic", "love", "beautiful", "sweet", "passion", "kadhal"],
}


class ConstraintClassifier(nn.Module):
    """
    Feed-forward neural network for genre and mood classification.
    Takes 384-dim multilingual sentence embeddings as input.
    Two classification heads: genre and mood.
    """

    def __init__(self, input_dim=384, hidden_dim=128, num_genres=11, num_moods=9):
        super().__init__()
        # Shared feature layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        # Genre classification head
        self.genre_head = nn.Linear(hidden_dim // 2, num_genres)
        # Mood classification head
        self.mood_head = nn.Linear(hidden_dim // 2, num_moods)

    def forward(self, x):
        features = self.shared(x)
        genre_logits = self.genre_head(features)
        mood_logits = self.mood_head(features)
        return genre_logits, mood_logits


# Loaded model and label maps (lazy)
_classifier = None
_label_maps = None


def _load_classifier():
    """Load the trained constraint classifier and label maps."""
    global _classifier, _label_maps

    if _classifier is not None:
        return

    if not os.path.exists(CONSTRAINT_MODEL_PATH) or not os.path.exists(LABEL_MAPS_PATH):
        print("[WARN] Trained constraint model not found. Using rule-based fallback only.")
        print(f"       Expected: {CONSTRAINT_MODEL_PATH}")
        print(f"       Run train_constraint_model.py first.")
        return

    _label_maps = torch.load(LABEL_MAPS_PATH, weights_only=False)
    num_genres = len(_label_maps["genre_labels"])
    num_moods = len(_label_maps["mood_labels"])

    _classifier = ConstraintClassifier(
        num_genres=num_genres, num_moods=num_moods
    )
    _classifier.load_state_dict(
        torch.load(CONSTRAINT_MODEL_PATH, weights_only=False)
    )
    _classifier.eval()


def _classify_genre_mood(text: str) -> tuple:
    """
    Use the trained classifier to predict genre and mood.
    Falls back to rule-based detection if model is unavailable.
    Includes explicit keyword boosting to safeguard unambiguous user queries.
    
    Returns:
        (genre_str, mood_str)
    """
    _load_classifier()

    # Identify if a direct genre keyword was explicitly specified in the query
    text_lower = text.lower()
    direct_genre = None
    for g, kws in GENRE_KEYWORDS.items():
        for kw in kws:
            if re.search(rf"\b{re.escape(kw)}\b", text_lower):
                # Ensure it's not preceded by a negation
                if not re.search(rf"\b(no|not|without|venda|vendaam)\s+{re.escape(kw)}\b", text_lower):
                    direct_genre = g
                    break
        if direct_genre:
            break

    if _classifier is not None and _label_maps is not None:
        # Use trained model
        model = get_embedding_model()
        embedding = model.encode([text], convert_to_numpy=True)
        input_tensor = torch.FloatTensor(embedding)

        with torch.no_grad():
            genre_logits, mood_logits = _classifier(input_tensor)

        # Apply prior boost if user explicitly typed a clear genre word
        if direct_genre and direct_genre in _label_maps["genre_labels"]:
            g_idx = _label_maps["genre_labels"].index(direct_genre)
            genre_logits[0][g_idx] += 3.5

        genre_idx = torch.argmax(genre_logits, dim=1).item()
        mood_idx = torch.argmax(mood_logits, dim=1).item()

        genre = _label_maps["genre_labels"][genre_idx]
        mood = _label_maps["mood_labels"][mood_idx]
        return genre, mood

    # Fallback: rule-based
    return _rule_based_genre(text), _rule_based_mood(text)


def _rule_based_genre(text: str) -> str:
    """Detect genre from keywords (used as fallback)."""
    text_lower = text.lower()
    scores = {}
    for genre, keywords in GENRE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[genre] = score
    if scores:
        return max(scores, key=scores.get)
    return "drama"  # default


def _rule_based_mood(text: str) -> str:
    """Detect mood from keywords (used as fallback)."""
    text_lower = text.lower()
    scores = {}
    for mood, keywords in MOOD_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[mood] = score
    if scores:
        return max(scores, key=scores.get)
    return "exciting"  # default


def _detect_language_pref(text: str) -> str:
    """
    Rule-based language preference detection.
    Looks for explicit Tamil/English mentions, Tanglish cinema terms,
    and Hollywood/Western franchise references.
    """
    text_lower = text.lower()
    tamil_explicit = ["tamil", "tamizh", "thamizh", "kollywood"]
    tanglish_markers = ["padam", "venum", "sollunga", "irukanum", "maari", "maathiri", "pakura", "pannunga", "pazhaya", "pudhusa"]
    has_tamil_ref = bool(re.search(r"\b96\b", text_lower))

    english_markers = ["english", "hollywood", "marvel", "dc", "godfather", "shutter island"]

    has_tamil = any(m in text_lower for m in tamil_explicit) or any(m in text_lower for m in tanglish_markers) or has_tamil_ref
    has_english = any(m in text_lower for m in english_markers)

    # Special case: queries asking for war/action with both Tamil & English markers or broad request
    if "war or action" in text_lower:
        return "any"

    if has_tamil and not has_english:
        return "ta"
    elif has_english and not has_tamil:
        return "en"
    return "any"


def _detect_exclusions(text: str) -> str:
    """
    Rule-based exclusion detection.
    Looks for patterns like 'no horror', 'not comedy', 'illa action'.
    """
    text_lower = text.lower()
    exclusions = []

    # Pattern: "no <genre>", "not <genre>", "no <genre>"
    negation_patterns = [
        r"\bno\s+(\w+)",
        r"\bnot\s+(\w+)",
        r"\bwithout\s+(\w+)",
        r"\billa\s+(\w+)",
        r"\billai\s+(\w+)",
        r"\bvenda\s+(\w+)",
        r"\bvendaam\s+(\w+)",
        r"\birukkakoodathu\b",
    ]

    for pattern in negation_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            # Check if the matched word is a genre or genre-adjacent
            for genre, keywords in GENRE_KEYWORDS.items():
                if match in keywords or match == genre:
                    exclusions.append(genre)

    # Also check for "too gory" style exclusions
    if "gory" in text_lower or "gore" in text_lower:
        exclusions.append("gore")
    if "silly" in text_lower:
        exclusions.append("silly")
    if "too scary" in text_lower:
        pass  # They still want horror, just not extreme

    return ", ".join(set(exclusions)) if exclusions else ""


def _detect_reference_title(text: str) -> str:
    """
    Detect if the user references a specific movie title.
    Looks for patterns like 'similar to X', 'like X', 'X maari'.
    """
    text_lower = text.lower()

    # Patterns to detect reference titles
    patterns = [
        r"similar to\s+(.+?)(?:\s+but|\s+with|\s*$)",
        r"like\s+(.+?)(?:\s+but|\s+with|\s+movie|\s*$)",
        r"(.+?)\s+maari\b",
        r"(.+?)\s+maathiri\b",
        r"(.+?)\s+mathiri\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            title = match.group(1).strip()
            # Clean up common noise words
            title = re.sub(r"\b(oru|nalla|romba|one|a|an|the|good|more)\b", "", title).strip()
            if len(title) > 1:
                return title.title()

    return ""


def _detect_era(text: str) -> str:
    """
    Detect temporal preference (old vs new vs any).
    Catches terms like 'pazhaya', 'old', 'classic', 'retro', 'vintage',
    and 'pudhusa', 'new', 'recent', 'latest'.
    """
    text_lower = text.lower()
    old_markers = ["pazhaya", "pazhaiya", "old", "classic", "retro", "vintage", "80s", "90s", "purana"]
    new_markers = ["pudhusa", "pudhusu", "pudhiya", "new", "recent", "latest", "modern"]

    if any(m in text_lower for m in old_markers):
        return "old"
    elif any(m in text_lower for m in new_markers):
        return "new"
    return "any"


def extract_constraints(text: str) -> dict:
    """
    Extract structured constraints from a (possibly Tanglish) query.
    
    Steps:
    1. Normalize Tanglish words to English
    2. Classify genre and mood using trained model (or rule-based fallback)
    3. Detect language preference using keyword rules
    4. Detect exclusions using negation patterns
    5. Detect reference movie titles
    6. Detect era (temporal preference: old vs new vs any)
    
    Args:
        text: Raw query text
        
    Returns:
        dict with keys: genre, mood, language_pref, exclusions, reference_title, era
    """
    # Keep original for some detections
    original_text = text

    # Normalize Tanglish
    normalized = normalize_query(text)

    # Classify genre and mood (model-based or rule-based)
    genre, mood = _classify_genre_mood(normalized)

    # Rule-based detections (work on both original and normalized)
    # Use original for language detection (to catch "Tamil" before normalization)
    language_pref = _detect_language_pref(original_text)
    # If language pref not found in original, try normalized
    if language_pref == "any":
        language_pref = _detect_language_pref(normalized)

    exclusions = _detect_exclusions(original_text)
    if not exclusions:
        exclusions = _detect_exclusions(normalized)

    reference_title = _detect_reference_title(original_text)

    era = _detect_era(original_text)
    if era == "any":
        era = _detect_era(normalized)

    return {
        "genre": genre,
        "mood": mood,
        "language_pref": language_pref,
        "exclusions": exclusions,
        "reference_title": reference_title,
        "era": era,
    }


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_queries = [
        "nalla comedy padam venum",
        "I want a dark thriller movie",
        "romba pazhaya Tamil action padam venum",
        "emotional Tamil movie pls no horror",
        "similar to Vikram but more comedy",
    ]

    print("=== Query Understanding Test ===\n")
    for q in test_queries:
        normalized = normalize_query(q)
        constraints = extract_constraints(q)
        print(f"  Query:      {q}")
        print(f"  Normalized: {normalized}")
        print(f"  Constraints: {constraints}")
        print()
