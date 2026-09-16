"""
src/visualizations/descriptive_plots.py
----------------------------------------
Descriptive visualizations of topic structure and emotion distributions.

Produces:
  1. Rating frequency bar chart
  2. Clause-level and review-level topic frequency bar charts
  3. Topic combination frequency bar chart
  4. Clause topics vs rating heatmap
  5. Emotion heatmaps (all clauses, 5-star clauses, 1-star clauses)
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from constants import (
    EXPLODED_CLAUSES,
    REVIEWS_TEXT,
    CLAUSE_VECTORS,
    EMOTION_LABELS_NO_NEUTRAL,
    POS_EMOTIONS,
    NEG_EMOTIONS,
    TOPIC_DISPLAY_NAMES,
    TOPIC_ABBREVIATIONS,
    ATC_EXTRACTED,
    PROFESSORS_CLEANED
)

# ==============================================================================
# HELPERS
# ==============================================================================

def _color_xticklabels(ax) -> None:
    for label in ax.get_xticklabels():
        text = label.get_text()
        if text in POS_EMOTIONS:
            label.set_color("green")
        elif text in NEG_EMOTIONS:
            label.set_color("red")


def _load_exploded() -> pd.DataFrame:
    df = pd.read_csv(EXPLODED_CLAUSES)
    df["predicted_topic"] = df["predicted_topic"].map(TOPIC_DISPLAY_NAMES)
    return df


# ==============================================================================
# FINAL VISUALIZATIONS (topic structure)
# ==============================================================================

def plot_rating_frequencies(reviews_df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    ax = reviews_df["rating"].value_counts().sort_index().plot(kind="bar", color="navy", rot=0)
    ax.bar_label(ax.containers[0], padding=3)
    plt.title("Rating Frequencies")
    plt.xlabel("Rating (1-5)")
    plt.ylabel("Number of Reviews")
    plt.tight_layout()


def plot_topic_frequencies(df: pd.DataFrame) -> None:
    review_level = (
        df.pivot_table(
            index="review_id",
            columns="predicted_topic",
            values="review_clauses",
            aggfunc="count",
            fill_value=0,
        )
        > 0
    ).astype(int)

    review_freq = review_level.sum().sort_values(ascending=False)
    clause_freq = df["predicted_topic"].value_counts().sort_values(ascending=False)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 10))

    clause_freq.plot(kind="bar", color="lightgreen", ax=axes[0], rot=0)
    axes[0].bar_label(axes[0].containers[0], padding=3)
    axes[0].set_title("Clause-Level Topic Frequency", fontsize=14)
    axes[0].set_ylabel("Number of Clauses")
    axes[0].set_xlabel("Topic")
    axes[0].set_ylim(top=clause_freq.max() * 1.15)

    review_freq.plot(kind="bar", color="skyblue", ax=axes[1], rot=0)
    axes[1].bar_label(axes[1].containers[0], padding=3)
    axes[1].set_title("Review-Level Topic Frequency", fontsize=14)
    axes[1].set_ylabel("Number of Reviews")
    axes[1].set_xlabel("Topic")
    axes[1].set_ylim(top=review_freq.max() * 1.15)

    plt.subplots_adjust(hspace=0.6)


def plot_topic_combinations(df: pd.DataFrame) -> None:
    # predicted_topic is already mapped to display names by _load_exploded, so accept
    # either the raw key or the display name when abbreviating.
    to_abbrev = dict(TOPIC_ABBREVIATIONS)
    to_abbrev.update(
        {TOPIC_DISPLAY_NAMES[k]: v for k, v in TOPIC_ABBREVIATIONS.items() if k in TOPIC_DISPLAY_NAMES}
    )

    def _to_short(topics) -> str:
        codes = {to_abbrev.get(t, t) for t in topics}
        order = list(TOPIC_ABBREVIATIONS.values())
        return " + ".join(sorted(codes, key=lambda c: order.index(c) if c in order else len(order)))

    combination_counts = (
        df.groupby("review_id")["predicted_topic"].apply(_to_short).value_counts().sort_values()
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    combination_counts.plot(kind="barh", color="goldenrod", ax=ax)
    ax.bar_label(ax.containers[0], padding=3, fontsize=9)
    ax.set_xlim(right=combination_counts.max() * 1.12)
    ax.set_title("Frequency of Topic Combinations (per Review)")
    ax.set_xlabel("Number of Reviews")
    ax.set_ylabel("Topic Combination")

    legend_labels = [
        mpatches.Patch(color="none", label=f"{v} = {TOPIC_DISPLAY_NAMES[k].replace(chr(10), ' ')}")
        for k, v in TOPIC_ABBREVIATIONS.items()
    ]
    ax.legend(
        handles=legend_labels,
        title="Topic Key",
        loc="lower right",
        frameon=True,
        fontsize=10,
        handlelength=0,
        handletextpad=0,
    )
    plt.tight_layout()


def plot_topic_rating_heatmap(df: pd.DataFrame) -> None:
    heatmap_data = df.pivot_table(
        index="predicted_topic",
        columns="rating",
        values="review_id",
        aggfunc="count",
    ).fillna(0)

    robust_vmax = pd.Series(heatmap_data.values.flatten()).quantile(0.95)

    plt.figure(figsize=(20, 6))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt="g",
        cmap="YlGnBu",
        vmax=robust_vmax,
        cbar_kws={"label": "Number of Clauses", "extend": "max"},
    )
    plt.title("Clause Topics vs Rating: Heatmap", fontsize=16, pad=10)
    plt.xlabel("Rating", fontsize=12)
    plt.ylabel("Topic", fontsize=12)
    plt.subplots_adjust(left=0.1, right=1.05)
    plt.yticks(rotation=0)


# ==============================================================================
# EMOTION VISUALIZATIONS
# ==============================================================================

def _emotion_heatmap(
    df: pd.DataFrame,
    emotions: list[str],
    title: str,
    cmap: str,
    vmax: float | None = None,
) -> None:
    category_emotions = df.groupby("predicted_topic")[emotions].mean()

    if vmax is None:
        vmax = pd.Series(category_emotions.values.flatten()).quantile(0.95)

    plt.figure(figsize=(20, 3))
    ax = sns.heatmap(
        category_emotions,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmax=vmax,
        cbar_kws={"extend": "max", "label": "Average Probability"},
    )
    ax.set_aspect(0.4) 
    cbar = ax.collections[0].colorbar
    cbar.set_label("Average Probability", fontsize=12)
    cbar.ax.tick_params(labelsize=12)
    _color_xticklabels(ax)
    ax.tick_params(axis="x", labelsize=12, rotation = 5)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right", fontsize=12)
    plt.ylabel("")
    plt.title(title, fontsize=19, pad=10)
    plt.subplots_adjust(left=0.1, right=1.05)


def plot_emotion_heatmaps(df_clauses: pd.DataFrame) -> None:
    df_clauses["predicted_topic"] = df_clauses["predicted_topic"].map(TOPIC_DISPLAY_NAMES)

    # All clauses — top 10 emotions by mean probability
    top_10_all = df_clauses[EMOTION_LABELS_NO_NEUTRAL].mean().nlargest(10).index.tolist()
    _emotion_heatmap(df_clauses, top_10_all, "All Clauses: Emotion Heatmap", "YlGnBu")

    # 5-star clauses
    df_5 = df_clauses[df_clauses["rating"] == 5.0]
    top_10_5 = df_5[EMOTION_LABELS_NO_NEUTRAL].mean().nlargest(10).index.tolist()
    _emotion_heatmap(df_5, top_10_5, "5-Star Clauses: Emotion Heatmap", "Greens")

    # 1-star clauses
    df_1 = df_clauses[df_clauses["rating"] == 1.0]
    top_10_1 = df_1[EMOTION_LABELS_NO_NEUTRAL].mean().nlargest(10).index.tolist()
    _emotion_heatmap(df_1, top_10_1, "1-Star Clauses: Emotion Heatmap", "Reds", vmax=0.14)


def plot_reviews_per_professor(professors_df: pd.DataFrame) -> None:
    from scipy import stats

    mean_count   = professors_df["review_count"].mean()
    median_count = professors_df["review_count"].median()
    mode_count   = stats.mode(professors_df["review_count"], keepdims=True).mode[0]

    plt.figure(figsize=(10, 6))
    ax = sns.kdeplot(
        professors_df["review_count"],
        color="navy",
        fill=True,
        alpha=0.3,
    )
    ax.axvline(mean_count,   color="navy",   linestyle="--", linewidth=1.5, label=f"Mean: {mean_count:.1f}")
    ax.axvline(median_count, color="green",  linestyle="--", linewidth=1.5, label=f"Median: {median_count:.1f}")
    ax.axvline(mode_count,   color="red",    linestyle="--", linewidth=1.5, label=f"Mode: {mode_count:.1f}")
    ax.set_xlim(left=8)
    plt.title("Distribution of Review Count per Professor")
    plt.xlabel("Number of Reviews")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

def plot_text_length_statistics(reviews_df: pd.DataFrame) -> None:
    from scipy import stats

    reviews_df = reviews_df.copy()
    reviews_df["text_length"] = reviews_df["review"].str.split().str.len()

    mean_len   = reviews_df["text_length"].mean()
    median_len = reviews_df["text_length"].median()
    mode_len   = stats.mode(reviews_df["text_length"], keepdims=True).mode[0]

    plt.figure(figsize=(10, 6))
    ax = sns.histplot(
        reviews_df["text_length"],
        bins=50,
        color="darkorange",
        kde=False,
    )
    ax.axvline(mean_len,   color="navy",  linestyle="--", linewidth=1.5, label=f"Mean: {mean_len:.1f}")
    ax.axvline(median_len, color="green", linestyle="--", linewidth=1.5, label=f"Median: {median_len:.1f}")
    ax.axvline(mode_len,   color="red",   linestyle="--", linewidth=1.5, label=f"Mode: {mode_len:.1f}")
    plt.title("Distribution of Review Text Length (Word Count)")
    plt.xlabel("Word Count")
    plt.ylabel("Number of Reviews")
    plt.legend()
    plt.tight_layout()


def plot_dmisc_distribution(atc_df: pd.DataFrame) -> None:
    mean_dmisc   = atc_df["misc_d"].mean()
    median_dmisc = atc_df["misc_d"].median()

    plt.figure(figsize=(10, 6))
    ax = sns.histplot(
        atc_df["misc_d"],
        bins=40,
        color="steelblue",
        kde=False,
    )
    ax.axvline(mean_dmisc,   color="navy",  linestyle="--", linewidth=1.5, label=f"Mean: {mean_dmisc:.2f}")
    ax.axvline(median_dmisc, color="green", linestyle="--", linewidth=1.5, label=f"Median: {median_dmisc:.2f}")
    ax.set_xlim(left=0)
    plt.title("Distribution of Miscellaneous Content Density (D$_{misc}$)", fontsize=16)
    plt.xlabel("D$_{misc}$", fontsize=16, labelpad=1)
    plt.ylabel("Number of Reviews", fontsize=16)
    plt.legend(fontsize=16)
    plt.tight_layout()

# ==============================================================================
# ENTRY POINT
# ==============================================================================



def run() -> None:
    print("\n=== Descriptive Visualizations ===")

    print("  Loading data...")
    df         = _load_exploded()
    reviews_df = pd.read_csv(REVIEWS_TEXT)
    df_clauses = pd.read_parquet(CLAUSE_VECTORS)
    atc_df = pd.read_csv(ATC_EXTRACTED)
    professors_df = pd.read_csv(PROFESSORS_CLEANED)

    print("  Plotting rating frequencies...")
    plot_rating_frequencies(reviews_df)

    print("  Plotting topic frequencies...")
    plot_topic_frequencies(df)

    print("  Plotting topic combinations...")
    plot_topic_combinations(df)

    print("  Plotting topic-rating heatmap...")
    plot_topic_rating_heatmap(df)

    print("  Plotting emotion heatmaps...")
    plot_emotion_heatmaps(df_clauses)

    print("  Plotting Dmisc distribution...")
    plot_dmisc_distribution(atc_df)

    print("  Plotting reviews per professor...")
    plot_reviews_per_professor(professors_df)

    print("  Plotting text length statistics...")
    plot_text_length_statistics(reviews_df)

    plt.show()


if __name__ == "__main__":
    run()
