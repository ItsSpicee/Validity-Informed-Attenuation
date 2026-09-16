"""
src/preprocessing.py
--------------------
Stage 1: Raw dataset cleaning.

Produces:
  - professors_cleaned.csv
  - reviews_cleaned.csv
  - reviews_text_cleaned.csv
"""

import re
import ftfy
import pandas as pd

from constants import (
    RAW_DATASET,
    PROFESSORS_CLEANED,
    REVIEWS_CLEANED,
    REVIEWS_TEXT,
    MIN_REVIEWS_PER_PROF,
)


# ==============================================================================
# STAGE 1A — Structural filtering 
# ==============================================================================

def load_raw(path=RAW_DATASET) -> pd.DataFrame:
    return pd.read_csv(path, on_bad_lines="skip")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename new dataset columns to the internal names used throughout the pipeline.

    """
    df = df.rename(columns={
        "professor_name":  "professor",
        "school_name":     "school",
        "department_name": "department",
        "state_name":      "state",
        "student_star":     "rating",
        "comments":        "review",
    })

    # Generate a sequential review_id 
    if "review_id" not in df.columns:
        df = df.reset_index(drop=True)
        df["review_id"] = df.index + 1

    return df


def clean_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("state", "professor", "school", "department"):
        df[col] = df[col].str.replace(r"\s+", " ", regex=True).str.strip()
    return df


def drop_french(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[df["state"] != "QC"]
    print(f"  French university reviews excluded: {before - len(df)}")
    return df


def drop_no_comment(df: pd.DataFrame) -> pd.DataFrame:
    """Remove reviews lacking either a usable comment or a numerical rating."""
    before = len(df)

    # CSV blank cells are read as NaN.  They do not match the string-based
    # filters below, so remove them explicitly before continuing downstream.
    has_review = df["review"].notna() & df["review"].astype(str).str.strip().ne("")
    has_rating = df["rating"].notna() & df["rating"].astype(str).str.strip().ne("")
    df = df[has_review & has_rating].copy()

    df = df[df["review"] != "No Comments"]

    pattern = r"^[nN][oO]\s+[cC][oO][mM][mM][eE][nN][tT][sS]?[\s.!]*$"
    df = df[~df["review"].str.contains(pattern, na=False, regex=True)]

    no_letters = r"^[^a-zA-Z]*$"
    df = df[~df["review"].str.contains(no_letters, na=False, regex=True)]

    print(f"  Empty, no-comment, or unrated reviews excluded: {before - len(df)}")
    return df


def drop_profanity_tagged(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[~df["review"].str.contains(r"\*\*+", na=False, regex=True)]
    print(f"  Auto-tagged profanity reviews excluded: {before - len(df)}")
    return df


def assign_prof_ids(df: pd.DataFrame) -> pd.DataFrame:
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["prof_ID"] = df.groupby(["professor", "school"]).ngroup() + 1
    df["review_count"] = df.groupby("prof_ID")["rating"].transform("count")
    return df


def drop_low_review_profs(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[df["review_count"] >= MIN_REVIEWS_PER_PROF]
    print(f"  Reviews dropped (professor < {MIN_REVIEWS_PER_PROF} reviews): {before - len(df)}")
    return df


def build_professor_df(df: pd.DataFrame) -> pd.DataFrame:
    prof_df = (
        df.groupby("prof_ID")
        .agg(
            professor=("professor", "first"),
            school=("school", "first"),
            department=("department", "first"),
            state=("state", "first"),
            average_rating=("rating", "mean"),
            review_count=("review_count", "first"),
        )
        .reset_index()
    )
    prof_df["prof_ID"] = prof_df["prof_ID"].astype(int)
    prof_df["review_count"] = prof_df["review_count"].astype(int)
    return prof_df


def build_review_df(df: pd.DataFrame) -> pd.DataFrame:
    review_df = df[["review_id", "prof_ID", "rating", "review"]].copy()
    review_df["prof_ID"] = review_df["prof_ID"].astype(int)
    return review_df


# ==============================================================================
# STAGE 1B — Text normalization 
# ==============================================================================

_ENCODING_REPLACEMENTS = {
    r"x?c3x?82x?c2x?b4": "'",
    r"xc2xb4": "'",
    r"xe2x80x9[34]": "-",
    r"xe2x80x9[89]": "'",
    r"xe2x80xa6": "...",
    r"xe2x80x9[cd]": '"',
    r'(?<=[a-zA-Z])\?(?=[tmsd]\b)': "'",
    r"(\w)\\\'(\w)": r"\1'\2",
}

_ARTIFACT_REPLACEMENTS = {
    r"([:;][\^\-]?[:\)DPp\(\/\\\|]|d\(.[^a-zA-Z0-9\s]+\.[^a-zA-Z0-9\s]*\)[a-zA-Z]?|\([ \^_\- ]+\))": " emoji ",
    r"(?<!\w)(?=(?:.*[!@#$%^&*]){2,})[!@#$%^&*]{3,}": " profanity ",
}

_SEGMENTATION_REGEX = {
    r"\bvs\.?(?![a-z])": "versus",
    r"\bapprox\.?\b": "approximately",
    r"\bdef\.?\b": "definitely",
    r"\b(eng)\.?": r"\1",
    r"\b(englit)\.?": r"\1",
    r"\b(soc)\.": r"\1",
    r"\b(lit)\.": r"\1",
    r"\b(mr)\.": r"\1",
    r"\b(ms)\.": r"\1",
    r"\b(mrs)\.": r"\1",
}

_SEGMENTATION_NORM = {
    "prof.": " professor ",
    "prof ": " professor ",
    "dr.":   " doctor ",
    "dr ":   " doctor ",
}

_SHORTFORM_NORM = {
    "%": " percent ",
    "a$$": "ass",
    "&": " and ",
    "$": " dollars ",
    "@": " at ",
}

_SHORTFORM_REGEX = {
    r"\bta\b": "teaching assistant",
    r"\btas\b|\bta\'s": "teaching assistants",
    r"\bh\.s\.\b": "highschool",
    r"\bhs\b": "highschool",
    r"\bm/c\b": "multiple choice",
    r"\bt/f\b": "true or false",
    r"\bq\s*(&|and)\s*a\b": "question and answer",
    r"\bq\'s\b": "questions",
    r"\bhws\b|\bhmwks\b|\bhwks\b": "homework",
    r"\bhwk\b|\bhmwk\b|\bhw\b": "homework",
    r"\bpgs\b": "pages",
    r"\bpg\b": "page",
    r"\bmt\b": "midterm",
    r"\bdb\b|\bDb\b": "discussion board",
    r"\bdb\'?s\b|\bDb\'?s\b|\bDB\'?[sS]\b": "discussion boards",
    r"\bhr\b": "hour",
    r"\bhrs\b": "hours",
    r"\bwk\b": "week",
    r"\bwks\b": "weeks",
    r"\bn\b": "and",
    r">\s*(\d+)": r"greater than \1",
    r"#\s*(\d+)": r"number \1",
    r"(?<!C)#\'?s\b": "numbers",
    r"\b(\d+)x\b": r"\1 times",
    r"\bw/\s*": "with ",
    r"\bw/o\s*": "without ",
}

_PUNCTUATION_REGEX = {
    r"\b(can|don|isn|wasn|won|shouldn|couldn|aren|didn|wouldn)[^a-zA-Z]{1,2}t\b": r"\1't",
    r"-{2,}": " ",
    r"[^a-zA-Z0-9\s.,!?\']": " ",
    r"([.!?,])([a-z])": r"\1 \2",
    r"\s+([.,!?])": r"\1",
}


def _apply_replacements(series: pd.Series, mapping: dict, regex: bool) -> pd.Series:
    for pattern, replacement in mapping.items():
        series = series.str.replace(pattern, replacement, regex=regex)
    return series


def fix_encoding(df: pd.DataFrame) -> pd.DataFrame:
    df["review"] = df["review"].astype(str).apply(ftfy.fix_text)
    df["review"] = (
        df["review"]
        .str.normalize("NFKD")
        .str.encode("ascii", "ignore")
        .str.decode("utf-8")
    )
    df["review"] = _apply_replacements(df["review"], _ENCODING_REPLACEMENTS, regex=True)
    return df


def tag_artifacts(df: pd.DataFrame) -> pd.DataFrame:
    df["review"] = df["review"].str.replace("<3", "love", regex=False)
    df["review"] = _apply_replacements(df["review"], _ARTIFACT_REPLACEMENTS, regex=True)
    return df


def fix_segmentation_risks(df: pd.DataFrame) -> pd.DataFrame:
    acronym_pattern = r"\b(?:[a-zA-Z]\.){2,}"
    df["review"] = df["review"].str.replace(
        acronym_pattern,
        lambda m: m.group(0).replace(".", "").lower(),
        regex=True,
    )
    df["review"] = _apply_replacements(df["review"], _SEGMENTATION_NORM, regex=False)
    df["review"] = _apply_replacements(df["review"], _SEGMENTATION_REGEX, regex=True)
    return df


def expand_short_forms(df: pd.DataFrame) -> pd.DataFrame:
    df["review"] = _apply_replacements(df["review"], _SHORTFORM_NORM, regex=False)
    df["review"] = _apply_replacements(df["review"], _SHORTFORM_REGEX, regex=True)
    return df


def fix_punctuation(df: pd.DataFrame) -> pd.DataFrame:
    df["review"] = _apply_replacements(df["review"], _PUNCTUATION_REGEX, regex=True)
    return df


def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Full text normalization pipeline applied to the review column."""
    print("  Fixing encoding errors...")
    df = fix_encoding(df)

    print("  Converting to lowercase...")
    df["review"] = df["review"].str.lower()

    print("  Tagging emojis and self-censored profanity...")
    df = tag_artifacts(df)

    print("  Fixing segmentation risks...")
    df = fix_segmentation_risks(df)

    print("  Expanding short-forms...")
    df = expand_short_forms(df)

    print("  Fixing punctuation...")
    df = fix_punctuation(df)

    print("  Normalizing whitespace...")
    df["review"] = df["review"].str.replace(r"\s+", " ", regex=True).str.strip()

    return df


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def run() -> None:
    print("\n=== Stage 1: Preprocessing ===")

    # --- 1A: Structural filtering ---
    print("Loading raw dataset...")
    df = load_raw()

    print("Normalizing column names...")
    df = normalize_columns(df)

    print("Applying structural filters...")
    df = clean_whitespace(df)
    df = drop_french(df)
    df = drop_no_comment(df)
    df = drop_profanity_tagged(df)
    df = assign_prof_ids(df)
    df = drop_low_review_profs(df)

    print(f"  Reviews remaining: {len(df)}")

    PROFESSORS_CLEANED.parent.mkdir(parents=True, exist_ok=True)
    build_professor_df(df).to_csv(PROFESSORS_CLEANED, index=False)
    build_review_df(df).to_csv(REVIEWS_CLEANED, index=False)
    print(f"  Saved: {PROFESSORS_CLEANED}, {REVIEWS_CLEANED}")

    # --- 1B: Text normalization ---
    print("Normalizing review text...")
    review_df = pd.read_csv(REVIEWS_CLEANED)
    review_df = normalize_text(review_df)
    review_df.to_csv(REVIEWS_TEXT, index=False)
    print(f"  Saved: {REVIEWS_TEXT}")


if __name__ == "__main__":
    run()
