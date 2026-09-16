"""
src/visualizations/correlation_plots.py
----------------------------------------
Correlation change visualizations before and after attenuation.

Produces:
  1. Pearson and Spearman correlation bar charts (before vs after attenuation)
  2. Average relative % change in correlation magnitude per topic

Plot 2 reports a true relative change, (|attuned| - |raw|) / |raw| * 100,
matching the convention used by the SHAP importance figure in
attenuation_plots.py so the two results figures are commensurable. It
previously computed (|attuned| - |raw|) * 100 -- an absolute difference in
correlation units labelled as a percentage -- and was titled "weighted"
although the two polarity groups were combined by an unweighted mean.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from catboost import CatBoostRegressor

from constants import (
    ATTUNED_RATINGS,
    FINAL_EMOTIONS,
    METADATA_COLS,
    MISC_D_COL,
    POS_EMOTIONS,
    NEG_EMOTIONS,
    TOPICS,
    TOPIC_DISPLAY_NAMES,
)

POLARITIES  = ["pos", "neg"]



# ==============================================================================
# HELPERS
# ==============================================================================

def _get_emotion_columns(df: pd.DataFrame, topic: str, polarity: str) -> list[str]:
    base = POS_EMOTIONS if polarity == "pos" else NEG_EMOTIONS
    return [f"{e}_{topic}" for e in base if f"{e}_{topic}" in df.columns]


def _compute_corr(x: pd.Series, y: pd.Series) -> tuple[float, float, float, float]:
    p, p_val = pearsonr(x, y)
    s, s_val = spearmanr(x, y)
    return p, p_val, s, s_val


def _print_corr(p, p_val, s, s_val, label: str) -> None:
    print(
        f"  {label}:\n"
        f"    Pearson: {p:.4f} (p={p_val:.4f}) | "
        f"Spearman: {s:.4f} (p={s_val:.4f})"
    )



# ==============================================================================
# CORRELATION RECORDS
# ==============================================================================

def build_corr_records(
    df_data: pd.DataFrame,
    final_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute raw and attuned correlations for every topic/polarity combination."""
    corr_records = []
    att_rating = final_df["rating"] + final_df["weighting_delta"]

    for topic in TOPICS:
        for polarity in POLARITIES:
            cols    = _get_emotion_columns(df_data, topic, polarity)
            summed  = df_data[cols].sum(axis=1)

            p_r, p_val_r, s_r, s_val_r = _compute_corr(final_df["rating"], summed)
            p_a, p_val_a, s_a, s_val_a = _compute_corr(att_rating, summed)

            _print_corr(p_r, p_val_r, s_r, s_val_r,
                        f"Raw Rating vs {polarity.capitalize()} {topic}")
            _print_corr(p_a, p_val_a, s_a, s_val_a,
                        f"Adjusted Rating vs {polarity.capitalize()} {topic}")

            corr_records.extend([
                {"topic": TOPIC_DISPLAY_NAMES[topic], "polarity": polarity, "type": "Pearson",
                 "raw": p_r, "attuned": p_a},
                {"topic": TOPIC_DISPLAY_NAMES[topic], "polarity": polarity, "type": "Spearman",
                 "raw": s_r, "attuned": s_a},
            ])

        print("  " + "-" * 30)

    return pd.DataFrame(corr_records)


# ==============================================================================
# PLOT 1: Pearson + Spearman before/after bars
# ==============================================================================

def plot_correlation_changes(corr_df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    x_labels = corr_df["topic"].unique()
    x        = np.arange(len(x_labels))
    width    = 0.18

    fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

    for ax, corr_type in zip(axes, ["Pearson", "Spearman"]):
        df_t = corr_df[corr_df["type"] == corr_type]

        pos_raw = df_t[df_t["polarity"] == "pos"]["raw"].values
        pos_att = df_t[df_t["polarity"] == "pos"]["attuned"].values
        neg_raw = df_t[df_t["polarity"] == "neg"]["raw"].values
        neg_att = df_t[df_t["polarity"] == "neg"]["attuned"].values

        ax.bar(x - 1.5 * width, pos_raw, width, label="Pos Raw",       color="palegreen")
        ax.bar(x - 0.5 * width, pos_att, width, label="Pos Adjusted", color="forestgreen")
        ax.bar(x + 0.5 * width, neg_raw, width, label="Neg Raw",       color="lightcoral")
        ax.bar(x + 1.5 * width, neg_att, width, label="Neg Adjusted", color="red")

        for xi, bars in enumerate(zip(pos_raw, pos_att, neg_raw, neg_att)):
            for j, val in enumerate(bars):
                va     = "bottom" if val >= 0 else "top"
                offset = 0.01    if val >= 0 else -0.01
                ax.text(
                    x[xi] + (j - 1.5) * width, val + offset,
                    f"{val:.2f}", ha="center", va=va, fontsize=8,
                )

        ax.set_title(f"{corr_type} Correlations Before and After Adjustment")
        ax.set_ylim(-0.6, 0.6)
        ax.legend()

    axes[0].set_ylabel("Pearson r")
    axes[1].set_ylabel("Spearman ρ")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(x_labels, rotation=0)
    axes[1].set_xlabel("Topics")
    plt.tight_layout()
    


# ==============================================================================
# PLOT 2: Average relative % change in correlation magnitude
# ==============================================================================

# Baseline correlations below this magnitude are treated as too small to give a
# meaningful relative change: dividing by them produces arbitrarily large
# percentages driven by noise in the denominator rather than by attenuation.
MIN_BASELINE_CORR = 0.01


def plot_relative_pct_change(corr_df: pd.DataFrame) -> None:
    def _pct_change(raw, attuned):
        """Relative change in correlation magnitude, in percent.

        Uses the same convention as the SHAP importance figure in
        attenuation_plots.py -- (attenuated - baseline) / baseline * 100 -- so
        that the two results figures report commensurable quantities. The
        previous form, (abs(attuned) - abs(raw)) * 100, was an absolute
        difference in correlation units mislabelled as a percentage.

        Magnitudes are compared because the sign of a correlation carries the
        direction of the emotion-rating relationship, not its strength; a
        negative correlation becoming more negative is a strengthening.
        """
        raw_mag = abs(raw)
        if raw_mag < MIN_BASELINE_CORR:
            return float("nan")
        return (abs(attuned) - raw_mag) / raw_mag * 100

    x_labels = corr_df["topic"].unique()
    records  = []

    def _topic_avg(df_a: pd.DataFrame, corr_type: str) -> float:
        """Mean relative change across the two polarity groups for one topic.

        nanmean, so that a polarity group whose baseline correlation falls below
        MIN_BASELINE_CORR is skipped rather than voiding the whole topic.
        """
        changes = [
            _pct_change(
                df_a[(df_a["type"] == corr_type) & (df_a["polarity"] == polarity)]["raw"].values[0],
                df_a[(df_a["type"] == corr_type) & (df_a["polarity"] == polarity)]["attuned"].values[0],
            )
            for polarity in POLARITIES
        ]
        if np.all(np.isnan(changes)):
            return float("nan")
        return float(np.nanmean(changes))

    for topic in x_labels:
        df_a = corr_df[corr_df["topic"] == topic]
        records.append({
            "topic":      topic,
            "pearson_d":  _topic_avg(df_a, "Pearson"),
            "spearman_d": _topic_avg(df_a, "Spearman"),
        })

    df_pct = pd.DataFrame(records)

    print("\n  --- Relative change in correlation magnitude (%) ---")
    for r in records:
        print(f"  {r['topic'].replace(chr(10), ' '):<28} "
              f"Pearson: {r['pearson_d']:+7.2f}%  |  Spearman: {r['spearman_d']:+7.2f}%")
    x      = np.arange(len(df_pct))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - width / 2, df_pct["pearson_d"],  width, label="Pearson Change",  color="dodgerblue")
    ax.bar(x + width / 2, df_pct["spearman_d"], width, label="Spearman Change", color="orange")

    for xi, (p, s) in enumerate(zip(df_pct["pearson_d"], df_pct["spearman_d"])):
        for val, xpos in [(p, x[xi] - width / 2), (s, x[xi] + width / 2)]:
            if np.isnan(val):
                continue
            va     = "bottom" if val >= 0 else "top"
            offset = 0.1     if val >= 0 else -0.1
            ax.text(xpos, val + offset, f"{val:+.1f}%", ha="center", va=va, fontsize=13)

    ax.set_xticks(x)
    ax.set_xticklabels(df_pct["topic"], rotation=0, fontsize=14.5)
    ax.set_ylabel("Avg. % Change in Correlation Magnitude", fontsize = 14.5)
    ax.set_title("Relative Change in Correlation Magnitude After Adjustment", fontsize = 14.5)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin * 1.1, ymax * 1.1)
    ax.legend()
    plt.tight_layout()


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def run() -> None:
    
    print("\n=== Correlation Change Visualizations ===")
    print("  Loading data...")
    df     = pd.read_csv(FINAL_EMOTIONS)
    df_att = pd.read_csv(ATTUNED_RATINGS)


    df_data   = df.drop(columns=METADATA_COLS)
    df_data   = df_data[df_data[MISC_D_COL] > 0].reset_index(drop=True)
    final_df  = df_att.copy()

    # Report correlations on professors excluded from model fitting.
    # df_data and final_df are row-aligned (both are the misc subset in order),
    # so the same positional mask applies to each.
    if "is_heldout" not in final_df.columns:
        raise KeyError(
            "attuned_ratings.csv has no 'is_heldout' column — re-run stage 5 "
            "(attenuation) to regenerate it."
        )
    mask     = final_df["is_heldout"].to_numpy(dtype=bool)
    df_data  = df_data[mask].reset_index(drop=True)
    final_df = final_df[mask].reset_index(drop=True)
    print(f"  Held-out professors only: {len(final_df):,} reviews")

    print("  Computing correlations...")
    corr_df = build_corr_records(df_data, final_df)

    print("  Plotting correlation changes...")
    plot_correlation_changes(corr_df)

    print("  Plotting relative % change...")
    plot_relative_pct_change(corr_df)


    plt.show()

    
if __name__ == "__main__":
    run()
