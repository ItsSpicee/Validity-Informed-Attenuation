"""
src/regression.py
-----------------
Stage 4: Emotion feature engineering (aggregation per review from clauses) + CatBoost regression.

Produces:
  - final_emotions.csv           (per-review feature matrix)
  - cat_boost_final.cbm          (trained model)
  - final_feature_importance.csv (trained model feature importance)
"""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

from constants import (
    CLAUSE_VECTORS,
    ATC_EXTRACTED,
    FINAL_EMOTIONS,
    CATBOOST_FINAL_MODEL,
    FEATURE_IMPORTANCE,
    METADATA_COLS,
    EMOTION_LABELS_NO_NEUTRAL,
    CATBOOST_PARAMS,
    CV_N_SPLITS,
)
from src.splits import heldout_rows, professor_split, train_rows


# ==============================================================================
# FEATURE ENGINEERING
# ==============================================================================

def build_feature_matrix(
    clause_vectors_path=CLAUSE_VECTORS,
    atc_path=ATC_EXTRACTED,
    output_path=FINAL_EMOTIONS,
) -> pd.DataFrame:
    """
    Pivot per-clause emotion vectors into per-review feature matrix and
    attach pedagogical density metrics.

    Column naming: {emotion}_{topic}  e.g. 'anger_fairness'
    """
    print("  Loading clause vectors...")
    df = pd.read_parquet(clause_vectors_path)

    # Drop neutral and similarity vector columns
    df = df.drop(columns=["neutral"], errors="ignore")
    df = df.drop(columns=["similarity"], errors="ignore")

    print("  Pivoting to per-review emotion features...")
    pivoted = (
        df.pivot_table(
            index="review_id",
            columns="predicted_topic",
            values=EMOTION_LABELS_NO_NEUTRAL,
            aggfunc="mean",
        )
        .fillna(0)
    )
    # Column naming: {emotion}_{topic}  e.g. 'anger_fairness'
    pivoted.columns = [f"{emotion}_{topic}" for emotion, topic in pivoted.columns]
    pivoted = pivoted.reset_index()

    # Merge with review metadata
    meta = df[["review_id", "prof_ID", "rating"]].drop_duplicates()
    result = meta.merge(pivoted, on="review_id", how="left")

    # Attach density metrics from ATC stage
    density_cols = ["review_id", "misc_d", "eff_d", "fair_d", "work_d"]
    ped_df = pd.read_csv(atc_path)
    result = result.merge(ped_df[density_cols], on="review_id", how="left").fillna(0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")
    return result


# ==============================================================================
# TRAINING
# ==============================================================================

def _evaluate(y_true, preds) -> tuple[float, float, float]:
    mae = mean_absolute_error(y_true, preds)
    rho = spearmanr(y_true, preds).statistic if len(np.unique(preds)) >= 2 else np.nan
    r2  = r2_score(y_true, preds)
    return mae, rho, r2


def cross_validate(df: pd.DataFrame) -> None:
    """5-fold professor-level cross-validation on training split.

    Each fold is fitted for a fixed number of iterations and scored on its
    validation rows. No `eval_set` is passed: early stopping would select the
    stopping iteration on the same rows the fold is then scored on, biasing the
    reported fold metrics downward.
    """
    train_profs, _ = professor_split(df)
    train_df = train_rows(df, train_profs)

    groups        = train_df["prof_ID"]
    X_train_full  = train_df.drop(columns=METADATA_COLS)
    y_train_full  = train_df["rating"]

    cv = GroupKFold(n_splits=CV_N_SPLITS)
    mae_scores, spearman_rhos, r2_scores = [], [], []

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X_train_full, y_train_full, groups=groups)
    ):
        X_tr, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
        y_tr, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

        model = CatBoostRegressor(**CATBOOST_PARAMS)
        model.fit(X_tr, y_tr)

        mae, rho, r2 = _evaluate(y_val, model.predict(X_val))
        mae_scores.append(mae)
        spearman_rhos.append(rho)
        r2_scores.append(r2)
        print(f"  Fold {fold + 1}: MAE={mae:.4f}  Spearman={rho:.4f}  R²={r2:.4f}")

    print(f"\n  CV Mean MAE:      {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
    print(f"  CV Mean Spearman: {np.mean(spearman_rhos):.4f} ± {np.std(spearman_rhos):.4f}")
    print(f"  CV Mean R²:       {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")


def train_final_model(df: pd.DataFrame) -> CatBoostRegressor:
    """Train on all training professors, evaluate on held-out test professors.

    Fitting takes no `eval_set`. Passing the test professors as one would let
    them select the stopping iteration, so the model size would be chosen on the
    same rows the model is then reported on — and, since this model produces the
    attenuation deltas, on the same rows Section 4.3 reports. The iteration
    count is fixed by CATBOOST_PARAMS instead; the loss curve is flat over the
    final iterations, so the cap is not doing selection work.
    """
    train_profs, test_profs = professor_split(df)

    train_df = train_rows(df, train_profs)
    test_df  = heldout_rows(df, test_profs)

    X_train = train_df.drop(columns=METADATA_COLS)
    y_train = train_df["rating"]
    X_test  = test_df.drop(columns=METADATA_COLS)
    y_test  = test_df["rating"]

    model = CatBoostRegressor(**CATBOOST_PARAMS)
    model.fit(X_train, y_train)

    mae, rho, r2 = _evaluate(y_test, model.predict(X_test))
    print(f"\n  Final test (unseen professors):")
    print(f"  MAE={mae:.4f}  Spearman={rho:.4f}  Pseudo R²={r2:.4f}")

    return model, X_train


def save_model_and_importance(
    model: CatBoostRegressor,
    feature_names: pd.Index,
) -> None:
    CATBOOST_FINAL_MODEL.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(CATBOOST_FINAL_MODEL))
    print(f"  Saved: {CATBOOST_FINAL_MODEL}")

    feat_imp = pd.DataFrame({
        "feature":    feature_names,
        "importance": model.get_feature_importance(),
    }).sort_values("importance", ascending=False)
    feat_imp.to_csv(FEATURE_IMPORTANCE, index=False)
    print(f"  Saved: {FEATURE_IMPORTANCE}")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def run() -> None:
    print("\n=== Stage 4: Regression ===")

    print("Building feature matrix...")
    df = build_feature_matrix()

    print("Running 5-fold cross-validation...")
    cross_validate(df)

    print("Training final model...")
    model, X_train = train_final_model(df)

    save_model_and_importance(model, X_train.columns)


if __name__ == "__main__":
    run()
