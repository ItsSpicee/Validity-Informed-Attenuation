"""
src/attenuation.py
------------------
Stage 5: Validity-weighted attenuation and rating adjustment.

Produces:
  - weighted_emotions.csv      (down-weighted feature matrix)
  - attuned_ratings.csv        (reviews with misc_d > 0, adjusted ratings + deltas)
  - attuned_ratings_full.csv   (all reviews including those with misc_d == 0)

The down-weighting exponent `s` is fixed a priori (constants.S_VALUE), not
estimated. At s = 1 the misc emotion channel is retained in exact proportion to
the review's on-topic fraction, so the attenuation is proportional by
construction and there is no parameter fitted to the reported metrics.

Both attuned_* files carry an `is_heldout` flag marking professors excluded from
CatBoost training. Reporting stages use it to keep validation metrics free of
the rows the model was fit on.
"""

import pandas as pd
from catboost import CatBoostRegressor

from constants import (
    CATBOOST_FINAL_MODEL,
    FINAL_EMOTIONS,
    WEIGHTED_EMOTIONS,
    ATTUNED_RATINGS,
    ATTUNED_RATINGS_FULL,
    METADATA_COLS,
    MISC_D_COL,
    POS_EMOTIONS,
    NEG_EMOTIONS,
    TOPICS,
    S_VALUE,
)
from src.splits import heldout_mask, professor_split


# ==============================================================================
# HELPERS
# ==============================================================================

def _sentiment_sum(df: pd.DataFrame, topic: str, polarity: str) -> pd.Series:
    base = POS_EMOTIONS if polarity == "pos" else NEG_EMOTIONS
    cols = [f"{e}_{topic}" for e in base if f"{e}_{topic}" in df.columns]
    return df[cols].sum(axis=1)


def _misc_intensity(df: pd.DataFrame, polarity: str) -> pd.Series:
    return _sentiment_sum(df, "misc", polarity)


def _apply_down_weighting(df: pd.DataFrame, s: float) -> pd.DataFrame:
    """Down-weight all *_misc emotion columns by (1 - misc_d^s)."""
    df_w = df.copy()
    misc_cols = [c for c in df_w.columns if "_misc" in c and c != MISC_D_COL]
    down_weighting = (1 - df_w[MISC_D_COL] ** s).values[:, None]
    df_w[misc_cols] *= down_weighting
    return df_w


# ==============================================================================
# ATTENUATION
# ==============================================================================

def attenuate(
    model: CatBoostRegressor,
    df: pd.DataFrame,
    s: float,
) -> pd.DataFrame:
    """
    Apply validity-weighted attenuation and return a DataFrame with adjusted ratings.

    Adjusted rating = raw_rating + (attenuated_pred - baseline_pred)
    """
    meta_df  = df[METADATA_COLS].copy()
    df_data  = df.drop(columns=METADATA_COLS)

    # Baseline predictions
    raw_preds = model.predict(df_data)

    # Weighted predictions
    df_weighted = _apply_down_weighting(df, s)
    weighted_preds = model.predict(df_weighted.drop(columns=METADATA_COLS))

    delta = weighted_preds - raw_preds

    full_df = pd.concat(
        [
            meta_df,
            pd.Series(raw_preds,    name="original_pred"),
            pd.Series(weighted_preds, name="weighted_pred"),
            pd.Series(delta,          name="weighting_delta"),
            df[MISC_D_COL],
        ],
        axis=1,
    )
    return full_df


def _print_summary(full_df: pd.DataFrame, final_df: pd.DataFrame, total_n: int) -> None:
    inc = (final_df["weighting_delta"] > 0).sum()
    dec = (final_df["weighting_delta"] < 0).sum()
    abs_delta = final_df["weighting_delta"].abs()

    print(f"\n  Total ratings attuned:        {((inc + dec) / total_n) * 100:.2f}%")
    print(f"  Ratings increased:            {(inc / total_n) * 100:.2f}%")
    print(f"  Ratings decreased:            {(dec / total_n) * 100:.2f}%")
    print(f"  Max |Δ|:                      {abs_delta.max():.4f}")
    print(f"  Mean |Δ| (adjusted reviews):  {abs_delta.mean():.4f}")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def run() -> None:
    print("\n=== Stage 5: Attenuation ===")

    print("Loading model and data...")
    model = CatBoostRegressor()
    model.load_model(str(CATBOOST_FINAL_MODEL), format="cbm")

    df = pd.read_csv(FINAL_EMOTIONS)

    # Subset: reviews with miscellaneous content
    df_misc = df[df[MISC_D_COL] > 0].reset_index(drop=True)

    # The model is still fit on training professors only, so reported metrics
    # stay on held-out professors. s itself is no longer estimated from data.
    train_profs, test_profs = professor_split(df)

    s = S_VALUE
    print(f"  Using s = {s} (fixed a priori, not tuned)")

    # The down-weighted feature matrix is written with the s this run actually
    # used, not with S_VALUE. It feeds the SHAP figures while attuned_ratings*
    # feeds everything else, so writing it at a different s would let the two
    # halves of the reporting describe different exponents without erroring.
    df_weighted = _apply_down_weighting(df, s)
    WEIGHTED_EMOTIONS.parent.mkdir(parents=True, exist_ok=True)
    df_weighted.to_csv(WEIGHTED_EMOTIONS, index=False)
    print(f"  Saved: {WEIGHTED_EMOTIONS}")

    # Run attenuation on full dataset
    full_df = attenuate(model, df, s)
    final_df = full_df[full_df[MISC_D_COL] > 0].reset_index(drop=True)

    # Attach misc emotion intensities
    X_misc_feat = df_misc.drop(columns=METADATA_COLS)
    final_df["misc_pos_intensity"] = _misc_intensity(X_misc_feat, "pos").values
    final_df["misc_neg_intensity"] = _misc_intensity(X_misc_feat, "neg").values
    final_df["total_misc_intensity"] = (
        final_df["misc_pos_intensity"] + final_df["misc_neg_intensity"]
    )

    # Tag split membership so reporting stages can restrict to held-out
    # professors without re-deriving (and possibly diverging from) the split.
    full_df["is_heldout"]  = heldout_mask(full_df, test_profs).values
    final_df["is_heldout"] = heldout_mask(final_df, test_profs).values
    print(f"  Held-out reviews (misc subset): {int(final_df['is_heldout'].sum()):,} "
          f"of {len(final_df):,}")

    _print_summary(full_df, final_df, total_n=len(df))

    final_df.to_csv(ATTUNED_RATINGS, index=False)
    full_df.to_csv(ATTUNED_RATINGS_FULL, index=False)
    print(f"  Saved: {ATTUNED_RATINGS}")
    print(f"  Saved: {ATTUNED_RATINGS_FULL}")


if __name__ == "__main__":
    run()
