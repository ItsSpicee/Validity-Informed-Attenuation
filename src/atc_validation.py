"""
src/atc_validation.py
---------------------
ATC validation: inter-rater LLM agreement and expert accuracy scoring.

Produces:
  - Cohen's Kappa between two LLM raters (Grok + GPT)
  - Set-based accuracy of ATC predictions vs expert labels
  - Confusion matrix
  - Classification report for single-label clauses
"""

import pandas as pd
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)

from constants import (
    ATC_EXPERT_LABELS,
    ATC_PREDICTIONS,
    GROK_LABELS,
    GPT_LABELS,
    TAU_SELECTION,
    TOPICS,
)

# ==============================================================================
# HELPERS
# ==============================================================================

def _convert_numeric_labels(numbers) -> list[str]:
    """
    Convert a numeric string like '13' into ['instructional_effectiveness', 'workload']
    using the index of the imported TOPICS list (1-based index).
    """
    result = []
    for char in set(str(numbers)):
        if char.isdigit():
            idx = int(char) - 1
            if 0 <= idx < len(TOPICS):
                result.append(TOPICS[idx])
    return result


def _load_merged(
    pred_path=ATC_PREDICTIONS,
    expert_path=ATC_EXPERT_LABELS,
) -> pd.DataFrame:
    preds_df  = pd.read_csv(pred_path)
    expert_df = pd.read_csv(expert_path)

    expert_df["expert_labels"] = expert_df["expert_labels"].apply(_convert_numeric_labels)


    merged = pd.merge(
        preds_df[["clause_id", "predicted_topic"]],
        expert_df,
        on="clause_id",
        how="inner",
    )
    merged = merged[merged['review_id'] != 13964]
    return merged


# ==============================================================================
# METRICS
# ==============================================================================

def set_based_accuracy(df: pd.DataFrame) -> float:
    """
    Containment accuracy — prediction is correct if it appears anywhere
    in the expert's allowed label set.
    """
    return df.apply(
        lambda row: row["predicted_topic"] in row["expert_labels"], axis=1
    ).mean()


def build_confusion_labels(df: pd.DataFrame) -> tuple[list, list]:
    """
    Projects multi-label expert annotations to a single label for
    confusion matrix computation.

    If prediction is in expert labels → y_true = prediction (correct).
    Otherwise → y_true = first expert label (disagreement).
    """
    y_true, y_pred = [], df["predicted_topic"].tolist()

    for _, row in df.iterrows():
        pred       = row["predicted_topic"]
        expert_set = row["expert_labels"]
        y_true.append(pred if pred in expert_set else expert_set[0])

    return y_true, y_pred


def cohens_kappa(
    grok_path=GROK_LABELS,
    gpt_path=GPT_LABELS,
) -> tuple[int, float]:
    grok_df = pd.read_csv(grok_path)
    gpt_df  = pd.read_csv(gpt_path)

    merged = pd.merge(
        grok_df[["clause_id", "silver_label"]],
        gpt_df[["clause_id",  "silver_label"]],
        on="clause_id",
        suffixes=("_grok", "_gpt"),
    )
    
    agreed = (merged["silver_label_grok"] == merged["silver_label_gpt"]).sum()
    kappa  = cohen_kappa_score(merged["silver_label_grok"], merged["silver_label_gpt"])
    return agreed, kappa


# ==============================================================================
# REPORTING
# ==============================================================================

def _report_kappa() -> None:
    agreed, kappa = cohens_kappa()
    print("\n" + "-" * 50)
    print("  Inter-rater agreement (ChatGPT vs Grok)")
    print("-" * 50)
    print(f"  Agreements : {agreed}")
    print(f"  Cohen's κ  : {kappa:.4f}")
    print("-" * 50)


def _report_accuracy(df: pd.DataFrame) -> None:
    acc = set_based_accuracy(df)
    print(f"\n  Set-based accuracy (all clauses): {acc:.4f}")


def _report_confusion(df: pd.DataFrame) -> None:
    y_true, y_pred = build_confusion_labels(df)
    cm    = confusion_matrix(y_true, y_pred, labels=TOPICS)
    cm_df = pd.DataFrame(cm, index=TOPICS, columns=TOPICS)
    print("\n  Confusion matrix (all clauses, single-label projection):")
    print(cm_df.to_string())


def _report_classification(df: pd.DataFrame) -> None:
    single_df = df[df["expert_labels"].apply(len) == 1].copy()
    n = len(single_df)

    if n == 0:
        print("\n  No single-label clauses found.")
        return

    y_true = single_df["expert_labels"].apply(lambda x: x[0]).tolist()
    y_pred = single_df["predicted_topic"].tolist()

    print(f"\n  Classification report — single-label clauses (n={n}):")
    print(
        classification_report(y_true, y_pred, labels=TOPICS, zero_division=0)
    )


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def _report_tau_overlap() -> None:
    tau = pd.read_csv(TAU_SELECTION)
    validation = pd.read_csv(ATC_PREDICTIONS)
    tau_ids = set(tau["review_id"].astype(int))
    validation_ids = set(validation["review_id"].astype(int))
    overlap = tau_ids & validation_ids
    print(f"\n  Tau development reviews:  {len(tau_ids)}")
    print(f"  ATC validation reviews:  {len(validation_ids)}")
    print(f"  Shared review IDs:       {len(overlap)}")


def run() -> None:
    print("\n=== ATC Validation ===")

    print("  Checking tau/validation set overlap...")
    _report_tau_overlap()

    print("  Computing inter-rater agreement...")
    _report_kappa()

    print("  Loading predictions and expert labels...")
    df = _load_merged()

    _report_accuracy(df)
    _report_confusion(df)
    _report_classification(df)


if __name__ == "__main__":
    run()