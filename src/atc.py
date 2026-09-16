"""
src/atc.py
----------
Stage 2: Clause extraction and aspect-term categorization (ATC).

Produces:
  - clause_dataset.csv     (reviews with clause lists)
  - exploded_clauses.csv   (one row per clause with predicted topic)
  - ATExtracted_reviews.csv (clause list for each category + density metrics per review)
"""

import ast
import csv
import re

import numpy as np
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from constants import (
    REVIEWS_TEXT,
    CLAUSE_DATASET,
    EXPLODED_CLAUSES,
    ATC_EXTRACTED,
    TOPIC_DESCRIPTIONS,
    TOPICS,
    SIMILARITY_THRESHOLD,
    EMBEDDING_MODEL,
)

# ==============================================================================
# CLAUSE EXTRACTION
# ==============================================================================



nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])


def _adaptive_clause_splitting(doc) -> list[str]:
    """
    Three-stage strategy:
      1. Proper punctuation (. ! ?) → use spaCy sentence segmentation.
      2. No punctuation but commas every ~15 words → split on commas + conjunctions.
      3. No punctuation at all → split on transition/stop words.
    """
    text = doc.text
    words = [t.text for t in doc if not t.is_punct and not t.is_space]
    word_count = len(words)
    if word_count == 0:
        return []

    terminals = sum(1 for t in doc if t.text in {".", "!", "?"})
    commas    = sum(1 for t in doc if t.text == ",")

    if terminals > 0:
        clauses = [sent.text.strip() for sent in doc.sents]

    elif commas / word_count > 0.06:
        delimiters = r"[,]| \bbut\b | \bso\b | \byet\b "
        clauses = re.split(delimiters, text, flags=re.IGNORECASE)

    else:
        delimiters = r" \bbut\b | \bso\b | \byet\b | \band\b | \balso\b "
        clauses = re.split(delimiters, text, flags=re.IGNORECASE)

    return [c.strip() for c in clauses if c.strip()]


def extract_clauses(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `review_clauses` column (list of clause strings) to the dataframe."""
    df["review"] = df["review"].str.lower()

    docs = list(
        tqdm(
            nlp.pipe(df["review"].astype(str), batch_size=500),
            total=len(df),
            desc="  Extracting clauses",
        )
    )
    df["review_clauses"] = [_adaptive_clause_splitting(doc) for doc in docs]
    return df


# ==============================================================================
# TOPIC CATEGORIZATION
# ==============================================================================

def _build_topic_embeddings(embedder: SentenceTransformer) -> np.ndarray:
    """Return (n_topics, embedding_dim) matrix of averaged topic embeddings."""
    print("  Precomputing topic embeddings...")
    vectors = []
    for info in TOPIC_DESCRIPTIONS.values():
        embs = embedder.encode(
            info["descriptions"],
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        vectors.append(np.mean(embs, axis=0))
    return np.vstack(vectors)


def categorize_clauses(df_exploded: pd.DataFrame, embedder: SentenceTransformer) -> pd.DataFrame:
    """Assign each clause to a topic or 'misc' based on cosine similarity."""
    topic_matrix = _build_topic_embeddings(embedder)

    all_clauses = df_exploded["review_clauses"].tolist()
    print("  Computing clause embeddings...")
    clause_matrix = embedder.encode(all_clauses, show_progress_bar=True, convert_to_numpy=True)

    print("  Categorizing clauses...")
    scores_matrix   = cosine_similarity(clause_matrix, topic_matrix)
    max_scores      = np.max(scores_matrix, axis=1)
    winning_indices = np.argmax(scores_matrix, axis=1)
    above_threshold = max_scores > SIMILARITY_THRESHOLD

    df_exploded["predicted_topic"] = [
        TOPICS[winning_indices[i]] if above_threshold[i] else "misc"
        for i in range(len(winning_indices))
    ]

    df_exploded["similarity"] = [
        f"[{' '.join(f'{v:.2f}' for v in row)}]"
        for row in scores_matrix
    ]

    return df_exploded


# ==============================================================================
# DENSITY METRICS
# ==============================================================================

def _word_count(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .str.replace("|", " ", regex=False)
        .str.split()
        .str.len()
        .fillna(0)
    )


def compute_density(df: pd.DataFrame) -> pd.DataFrame:
    eff_wc  = _word_count(df["instructional_effectiveness"])
    fair_wc = _word_count(df["fairness"])
    work_wc = _word_count(df["workload"])
    misc_wc = _word_count(df["misc"])
    total   = eff_wc + fair_wc + work_wc + misc_wc

    df["eff_d"]  = (eff_wc  / (1 + misc_wc + eff_wc )).round(2)
    df["fair_d"] = (fair_wc / (1 + misc_wc + fair_wc)).round(2)
    df["work_d"] = (work_wc / (1 + misc_wc + work_wc)).round(2)
    df["misc_d"] = (misc_wc / total).round(2)
    return df


# ==============================================================================
# PIVOT: one row per review
# ==============================================================================

def pivot_to_reviews(df_exploded: pd.DataFrame) -> pd.DataFrame:
    """Recombine per-clause rows back into one row per review."""
    pivot = df_exploded.pivot_table(
        index=df_exploded.index,
        columns="predicted_topic",
        values="review_clauses",
        aggfunc=lambda x: " | ".join(x.astype(str)),
    )
    pivot = pivot.reindex(columns=TOPICS)

    metadata = (
        df_exploded
        .drop(columns=["review_clauses", "predicted_topic", "similarity"])
        .groupby(df_exploded.index)
        .first()
    )
    return metadata.join(pivot)


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def run() -> None:
    print("\n=== Stage 2: ATC ===")

    # --- Clause extraction ---
    print("Loading cleaned reviews...")
    df_raw = pd.read_csv(REVIEWS_TEXT)
    df_raw = extract_clauses(df_raw)

    CLAUSE_DATASET.parent.mkdir(parents=True, exist_ok=True)
    df_raw.to_csv(CLAUSE_DATASET, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"  Saved: {CLAUSE_DATASET}")

    # ------------------------ ATC --------------------

    # --- Clause Segmentation
    print("Loading clause dataset...")
    df_clauses = pd.read_csv(CLAUSE_DATASET, on_bad_lines="skip")
    df_clauses = df_clauses.drop(columns=["review"], errors="ignore")
    df_clauses["review_clauses"] = df_clauses["review_clauses"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    df_exploded = df_clauses.explode("review_clauses").copy()


    # ----- Categorization ---

    

    embedder    = SentenceTransformer(EMBEDDING_MODEL)
    df_exploded = categorize_clauses(df_exploded, embedder)

    df_exploded.reindex(
        columns=["review_id", "prof_ID", "rating", "predicted_topic", "review_clauses", "similarity"]
    ).to_csv(EXPLODED_CLAUSES, index=False)
    print(f"  Saved: {EXPLODED_CLAUSES}")

    # --- Pivot + attach density ---
    df_final = pivot_to_reviews(df_exploded)
    df_final = compute_density(df_final)
    df_final.to_csv(ATC_EXTRACTED, index=False)
    print(f"  Saved: {ATC_EXTRACTED}")


if __name__ == "__main__":
    run()