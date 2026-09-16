"""Three-expert attenuation validation, Fleiss' kappa and Wilson intervals.

Labels 1/2 select a review; 3 is non-directional (both valid). Blank ratings
are missing. Kappa uses complete pairs before joining model deltas. Accuracy
counts a matching directional majority as correct and no majority as incorrect.
A label-3 majority also counts as incorrect; only unavailable deltas are excluded.
Wilson intervals assume
independent review pairs and condition on these three fixed experts.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from constants import (
    ATTUNED_RATINGS, DATA_DIR, EXPERT_LABELS_PATH, MISC_D_COL,
    COARSE_DELTA_THRESHOLD, FINE_DELTA_MIN, FINE_DELTA_MAX,
    REVIEWS_CLEANED, ATC_EXTRACTED, FINAL_EMOTIONS,
    NEG_EMOTIONS, POS_EMOTIONS,
)

PAIR_KEYS = ["review_id_1", "review_id_2"]
RATER_COLUMNS = ["expert_1", "expert_2", "expert_3"]


def load_expert_consensus(expert_paths=EXPERT_LABELS_PATH):
    """Align unordered pairs by IDs; preserve expert 1's displayed orientation.

Use the union of pairs so missing raters remain visible. Require three files;
two matching votes are required even when the third rating is missing.
"""
    if isinstance(expert_paths, (str, Path)) or len(expert_paths) != 3:
        raise ValueError("Provide exactly three expert CSV paths")
    merged = None
    for path, column in zip(expert_paths, RATER_COLUMNS):
        frame = pd.read_csv(path)[PAIR_KEYS + ["Expert Label:"]].copy()
        for key in PAIR_KEYS:
            values = pd.to_numeric(frame[key], errors="raise")
            if values.isna().any() or not np.isfinite(values).all() or (values % 1 != 0).any():
                raise ValueError(f"{path}: review IDs must be nonmissing integers")
            frame[key] = values.astype("int64")
        labels = pd.to_numeric(frame.pop("Expert Label:"), errors="raise")
        if (~labels.isna() & ~labels.isin([1, 2, 3])).any():
            raise ValueError(f"{path}: allowed labels are 1, 2, 3 or blank")
        reverse = frame.review_id_1 > frame.review_id_2
        frame[column] = labels.where(~(reverse & labels.isin([1, 2])), 3 - labels)
        frame["display_reversed"] = reverse
        frame[PAIR_KEYS] = np.sort(frame[PAIR_KEYS].to_numpy(), axis=1)
        if (frame.review_id_1 == frame.review_id_2).any() or frame.duplicated(PAIR_KEYS).any():
            raise ValueError(f"{path}: duplicate or self-comparison pair")
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame.drop(columns="display_reversed"), on=PAIR_KEYS,
                                  how="outer", validate="one_to_one")
    reverse = merged.pop("display_reversed").eq(True)
    merged.loc[reverse, PAIR_KEYS] = merged.loc[reverse, PAIR_KEYS[::-1]].to_numpy()
    for column in RATER_COLUMNS:
        labels = merged[column]
        merged[column] = labels.where(~(reverse & labels.isin([1, 2])), 3 - labels)
    for label in (1, 2, 3):
        merged[f"votes_{label}"] = merged[RATER_COLUMNS].eq(label).sum(axis=1)
    merged["n_ratings"] = merged[RATER_COLUMNS].notna().sum(axis=1)
    merged["consensus_label"] = np.select(
        [merged[f"votes_{label}"] >= 2 for label in (1, 2, 3)], [1, 2, 3], default=np.nan)
    merged["has_majority"] = merged.consensus_label.notna()
    merged["Expert Label:"] = merged.consensus_label
    merged["unsure"] = ~merged.consensus_label.isin([1, 2])
    return merged.sort_values(PAIR_KEYS).reset_index(drop=True)


def fleiss_kappa(ratings):
    """Fixed-rater Fleiss kappa with pooled category proportions.

Use complete pairs and labels 1, 2, 3. Return NaN when undefined.
"""
    complete = ratings[RATER_COLUMNS].dropna()
    counts = np.column_stack([complete.eq(k).sum(axis=1) for k in (1, 2, 3)])
    n = len(counts)
    observed = float(np.mean((counts * (counts - 1)).sum(axis=1) / 6)) if n else np.nan
    expected = float(np.sum((counts.sum(axis=0) / (3 * n)) ** 2)) if n else np.nan
    kappa = (observed - expected) / (1 - expected) if n and expected < 1 else np.nan
    return dict(n_total=len(ratings), n_complete=n, n_missing=len(ratings) - n,
                observed_agreement=observed, expected_agreement=expected, fleiss_kappa=kappa)


def fleiss_kappa_excluding_3(ratings):
    """Generalized Fleiss kappa after excluding individual 3/blank votes.

    Uses the unweighted irrCAC formulation for variable rater counts:
    observed agreement is the mean of each pair's agreement proportion among
    pairs with >=2 directional votes. Category proportions are the mean of
    within-pair proportions among pairs with >=1 directional vote. Thus a
    singleton contributes only to chance agreement; a zero-vote pair to neither.
    With three directional votes everywhere this equals ordinary Fleiss kappa.
    Reference: https://github.com/kgwet/irrCAC/blob/master/R/agree.coeff3.raw.r
    """
    votes = ratings[RATER_COLUMNS]
    counts = np.column_stack([votes.eq(k).sum(axis=1) for k in (1, 2)])
    totals = counts.sum(axis=1)
    comparable = totals >= 2
    nonempty = totals >= 1
    observed = expected = kappa = np.nan
    if comparable.any():
        numerator = (counts[comparable] * (counts[comparable] - 1)).sum(axis=1)
        denominator = totals[comparable] * (totals[comparable] - 1)
        observed = float(np.mean(numerator / denominator))
    if nonempty.any():
        proportions = (counts[nonempty] / totals[nonempty, None]).mean(axis=0)
        expected = float(np.sum(proportions ** 2))
    if comparable.any() and expected < 1:
        kappa = (observed - expected) / (1 - expected)
    return dict(
        n_total=len(votes), n_agreement_pairs=int(comparable.sum()),
        n_marginal_pairs=int(nonempty.sum()),
        n_two_raters=int((totals == 2).sum()), n_three_raters=int((totals == 3).sum()),
        n_one_rater=int((totals == 1).sum()), n_zero_raters=int((totals == 0).sum()),
        n_excluded_3=int(votes.eq(3).sum().sum()),
        n_blank_votes=int(votes.isna().sum().sum()),
        observed_agreement=observed, expected_agreement=expected,
        generalized_fleiss_kappa=kappa,
    )


def merge_expert_deltas(attuned, expert_path=EXPERT_LABELS_PATH, verbose=True):
    """Shared validation/robustness join, retaining exclusions for audit."""
    expert = load_expert_consensus(expert_path)
    # Bootstrap callers can contain repeated identical review rows.
    deltas = attuned[["review_id", "weighting_delta"]].drop_duplicates()
    if deltas.review_id.isna().any() or deltas.review_id.duplicated().any():
        raise ValueError("Model data contains missing IDs or conflicting deltas")
    delta_map = deltas.set_index("review_id").weighting_delta
    for number in (1, 2):
        expert[f"delta_{number}"] = expert[f"review_id_{number}"].map(delta_map)
    expert["missing_delta"] = ~np.isfinite(expert[["delta_1", "delta_2"]]).all(axis=1)
    expert["evaluable"] = ~expert.missing_delta
    if verbose:
        print(f"Pairs: {len(expert)}; no majority: {(~expert.has_majority).sum()}; "
              f"non-directional majority: {expert.consensus_label.eq(3).sum()}; "
              f"missing model deltas: {expert.missing_delta.sum()}; "
              f"evaluable: {expert.evaluable.sum()}")
    return expert


def load_and_merge(attuned_path=ATTUNED_RATINGS, expert_path=EXPERT_LABELS_PATH):
    return merge_expert_deltas(pd.read_csv(attuned_path), expert_path)


def score_predictions(expert):
    expert = expert.copy()
    expert["abs_delta_1"] = expert.delta_1.abs()
    expert["abs_delta_2"] = expert.delta_2.abs()
    expert["delta_diff"] = (expert.abs_delta_1 - expert.abs_delta_2).abs()
    # Preserve the existing review-1 tie rule.
    expert["model_pred"] = np.where(expert.abs_delta_1 <= expert.abs_delta_2, 1., 2.)
    expert.loc[expert.missing_delta, "model_pred"] = np.nan
    expert["model_tie"] = expert.abs_delta_1.eq(expert.abs_delta_2)
    expert["expert_label"] = expert.consensus_label
    expert["correct"] = (
        expert.model_pred.eq(expert.expert_label).fillna(False)
        & expert.consensus_label.isin([1, 2])
    ).astype("boolean")
    expert.loc[~expert.evaluable, "correct"] = pd.NA
    return expert


def wilson_interval(correct, n, confidence=0.95):
    """Two-sided Wilson confidence interval for accuracy."""
    if not 0 < confidence < 1 or not 0 <= correct <= n:
        raise ValueError("Invalid confidence level or success count")
    if n == 0:
        return np.nan, np.nan
    interval = stats.binomtest(correct, n).proportion_ci(confidence, method="wilson")
    return interval.low, interval.high


def condition_masks(expert):
    """Shared disjoint delta bins; pairs without model deltas are unbinned."""
    available = ~expert.missing_delta
    return {
        "Overall": pd.Series(True, index=expert.index),
        "Coarse": available & (expert.delta_diff >= COARSE_DELTA_THRESHOLD),
        "Fine": available & expert.delta_diff.between(FINE_DELTA_MIN, FINE_DELTA_MAX, inclusive="left"),
    }


def kappa_summary(expert, exclude_3=False):
    """Recompute category proportions independently within each condition.

    Overall includes all annotated pairs. Coarse/fine require model deltas to
    assign bins, but never require a majority vote or model correctness.
    exclude_3 adds the variable-rater sensitivity analysis on directional votes.
    """
    calculate = fleiss_kappa_excluding_3 if exclude_3 else fleiss_kappa
    return pd.DataFrame([
        dict(condition=condition, **calculate(expert.loc[mask]))
        for condition, mask in condition_masks(expert).items()
    ])


def accuracy_summary(expert, confidence=0.95):
    """Consensus and individual accuracy; label 3 counts as incorrect.

    Individual scores exclude missing ratings and unavailable model deltas.
    """
    conditions = condition_masks(expert)
    rows = []
    for target in ["consensus_label"] + RATER_COLUMNS:
        valid = (expert.evaluable if target == "consensus_label"
                 else expert[target].notna() & ~expert.missing_delta)
        for condition, mask in conditions.items():
            subset = expert.loc[valid & mask]
            n = len(subset)
            correct = (int(subset.correct.sum()) if target == "consensus_label"
                       else int(subset.model_pred.eq(subset[target]).sum()))
            lo, hi = wilson_interval(correct, n, confidence)
            rows.append(dict(target=target, condition=condition, n=n, correct=correct,
                             accuracy=correct / n if n else np.nan,
                             confidence=confidence, ci_low=lo, ci_high=hi))
    return pd.DataFrame(rows)


def fine_disagreement_diagnostic(expert):
    """Cross-tabulate expert unanimity with model accuracy in the fine condition."""
    masks = condition_masks(expert)
    fine = expert.loc[masks["Fine"] & expert.evaluable].copy()
    votes = fine[RATER_COLUMNS].apply(
        lambda row: row.dropna().nunique() == 1 and row.dropna().iloc[0] in (1, 2),
        axis=1,
    )
    fine["unanimous"] = votes
    rows = []
    for label, mask in [("Unanimous", fine.unanimous), ("Split", ~fine.unanimous)]:
        subset = fine.loc[mask]
        agrees = int(subset.correct.sum())
        disagrees = len(subset) - agrees
        rows.append({"Expert voting pattern": label,
                     "Model agrees": agrees, "Model disagrees": disagrees})
    return pd.DataFrame(rows)


def add_density_scoring(expert, attuned):
    """Add density-only predictions alongside attenuation predictions."""
    density_map = attuned.set_index("review_id")[MISC_D_COL]
    for n in (1, 2):
        expert[f"density_{n}"] = expert[f"review_id_{n}"].map(density_map)
    expert["missing_density"] = expert[["density_1", "density_2"]].isna().any(axis=1)
    expert["density_tie"] = expert["density_1"].eq(expert["density_2"])
    expert["density_pred"] = np.where(expert["density_1"] <= expert["density_2"], 1., 2.)
    expert.loc[expert.missing_density, "density_pred"] = np.nan
    expert["density_correct"] = (
        expert.density_pred.eq(expert.consensus_label)
        & expert.consensus_label.isin([1, 2])
        & ~expert.density_tie
    )
    expert["common_evaluable"] = expert.evaluable & ~expert.missing_density
    return expert


def density_accuracy_summary(expert, confidence=0.95):
    """Density-only agreement alongside attenuation, with paired comparison."""
    rows = []
    for condition, mask in condition_masks(expert).items():
        subset = expert.loc[mask & expert.common_evaluable]
        n = len(subset)
        model_ok = subset.correct.fillna(False).astype(bool)
        density_ok = subset.density_correct.astype(bool)
        for method, correct, ties in [
            ("Attenuation", model_ok, subset.model_tie),
            ("Density only", density_ok, subset.density_tie),
        ]:
            count = int(correct.sum())
            lo, hi = wilson_interval(count, n, confidence)
            rows.append(dict(condition=condition, method=method, n=n, correct=count,
                             agreement=count / n if n else np.nan,
                             ci_low=lo, ci_high=hi, ties=int(ties.sum())))
    return pd.DataFrame(rows)


def density_paired_comparison(expert):
    """Count concordant/discordant pairs between attenuation and density."""
    rows = []
    for condition, mask in condition_masks(expert).items():
        subset = expert.loc[mask & expert.common_evaluable]
        model_ok = subset.correct.fillna(False).astype(bool)
        density_ok = subset.density_correct.astype(bool)
        rows.append(dict(
            condition=condition, n=len(subset),
            both_correct=int((model_ok & density_ok).sum()),
            attenuation_only=int((model_ok & ~density_ok).sum()),
            density_only=int((density_ok & ~model_ok).sum()),
            both_incorrect=int((~model_ok & ~density_ok).sum()),
        ))
    return pd.DataFrame(rows)


def _miscellaneous_emotion_summaries(emotions):
    output = emotions[["review_id"]].copy()
    for name, labels in (("misc_positive", POS_EMOTIONS), ("misc_negative", NEG_EMOTIONS)):
        columns = [f"{label}_misc" for label in labels if f"{label}_misc" in emotions]
        output[name] = emotions[columns].sum(axis=1) if columns else 0.0
    output["misc_affect_total"] = output.misc_positive + output.misc_negative
    return output


def _review_metadata(attuned):
    base = attuned[["review_id", "prof_ID", "rating", "original_pred", "weighted_pred",
                     "weighting_delta", "misc_d", "misc_pos_intensity", "misc_neg_intensity",
                     "total_misc_intensity", "is_heldout"]].copy()
    text = pd.read_csv(REVIEWS_CLEANED)[["review_id", "review"]]
    topics = pd.read_csv(ATC_EXTRACTED)[["review_id", "instructional_effectiveness", "fairness", "workload", "misc"]]
    emotions = _miscellaneous_emotion_summaries(pd.read_csv(FINAL_EMOTIONS))
    return (base.merge(text, on="review_id", how="left", validate="one_to_one")
                .merge(topics, on="review_id", how="left", validate="one_to_one")
                .merge(emotions, on="review_id", how="left", validate="one_to_one"))


def _attach_reviews(pairs, metadata):
    output = pairs.copy()
    for number in (1, 2):
        renamed = metadata.rename(columns={col: f"review_{number}_{col}"
                                           for col in metadata.columns if col != "review_id"})
        output = output.merge(renamed, left_on=f"review_id_{number}", right_on="review_id",
                              how="left", validate="many_to_one").drop(columns="review_id")
    return output


def build_discordant_audit(expert, attuned):
    """Identify the six discordant pairs and attach review-level detail."""
    common = expert.loc[expert.common_evaluable].copy()
    attenuation_ok = common.correct.fillna(False).astype(bool)
    density_ok = common.density_correct.fillna(False).astype(bool)
    common["divergence_method"] = np.select(
        [attenuation_ok & ~density_ok, density_ok & ~attenuation_ok],
        ["attenuation_only", "density_only"], default="none",
    )
    divergent = common.loc[common.divergence_method.ne("none")].copy()
    divergent = _attach_reviews(divergent, _review_metadata(attuned))
    divergent["abs_delta_gap"] = divergent.abs_delta_1 - divergent.abs_delta_2
    divergent["density_gap"] = divergent.density_1 - divergent.density_2
    divergent["methods_select_different_reviews"] = divergent.model_pred.ne(divergent.density_pred)
    divergent["expert_selected_review_id"] = np.where(
        divergent.consensus_label.eq(1), divergent.review_id_1, divergent.review_id_2
    )
    front = ["divergence_method", "condition", "review_id_1", "review_id_2",
             "consensus_label", "expert_selected_review_id", "model_pred", "density_pred",
             "abs_delta_1", "abs_delta_2", "abs_delta_gap", "density_1", "density_2",
             "density_gap", "methods_select_different_reviews"]
    remaining = [col for col in divergent.columns if col not in front]
    return divergent[front + remaining].sort_values(["divergence_method", "condition", "review_id_1"])


def instructor_metadata(expert, attuned):
    """Report instructor recurrence across evaluable expert pairs."""
    instructor_map = attuned.set_index("review_id")["prof_ID"]
    evaluable = expert.loc[expert.evaluable].copy()
    for n in (1, 2):
        evaluable[f"prof_ID_{n}"] = evaluable[f"review_id_{n}"].map(instructor_map)
    missing = evaluable[[f"prof_ID_1", f"prof_ID_2"]].isna().any(axis=1).sum()
    if missing:
        print(f"  Pairs missing instructor metadata: {missing}")
        evaluable = evaluable.dropna(subset=["prof_ID_1", "prof_ID_2"])
    same = evaluable["prof_ID_1"].eq(evaluable["prof_ID_2"]).sum()
    long = pd.DataFrame({
        "prof_ID": pd.concat([evaluable["prof_ID_1"], evaluable["prof_ID_2"]]),
        "pair_idx": list(evaluable.index) * 2,
    }).drop_duplicates()
    pair_counts = long.groupby("prof_ID")["pair_idx"].nunique()
    repeated = pair_counts[pair_counts > 1]
    print(f"\n  Distinct instructors across evaluable pairs: {pair_counts.nunique()}")
    print(f"  Instructors in more than one pair: {len(repeated)}"
          f" (max {int(pair_counts.max())})")
    print(f"  Pairs with same instructor in both reviews: {same}")


def report(expert, attuned=None):
    print("\n=== Stage 6: Validation ===")
    print(kappa_summary(expert).to_string(index=False))
    print("\nGeneralized Fleiss' kappa excluding individual label-3 votes:")
    print(kappa_summary(expert, exclude_3=True).to_string(index=False))
    print("Pairs with one retained vote contribute only to chance agreement; "
          "pairs with zero retained votes are excluded.")
    print(f"Pairs without model deltas (overall kappa only): {expert.missing_delta.sum()}")
    print("\nAccuracy and two-sided 95% Wilson intervals:")
    print("No-majority and label-3-majority pairs with model deltas count as incorrect.")
    print("Individual label-3 votes count as incorrect; blank votes are excluded.")
    print(accuracy_summary(expert).to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print(f"\nModel ties (review 1 selected): {expert.model_tie.sum()}")
    print("\nFine-condition disagreement diagnostic:")
    print(fine_disagreement_diagnostic(expert).to_string(index=False))

    if attuned is not None:
        expert = add_density_scoring(expert, attuned)
        print("\nDensity-only baseline comparison:")
        print(density_accuracy_summary(expert).to_string(
            index=False, float_format=lambda x: f"{x:.4f}"))
        print("\nPaired comparison (attenuation vs density):")
        print(density_paired_comparison(expert).to_string(index=False))

        audit = build_discordant_audit(expert, attuned)
        counts = audit.divergence_method.value_counts()
        print(f"\nDiscordant pairs: "
              f"{counts.get('attenuation_only', 0)} attenuation-only, "
              f"{counts.get('density_only', 0)} density-only")
        output_path = DATA_DIR / "processed" / "discordant_pairs.csv"
        try:
            audit.to_csv(output_path, index=False, mode="x")
            print(f"  Saved: {output_path}")
        except FileExistsError:
            print(f"  Skipped (already exists): {output_path}")

        instructor_metadata(expert, attuned)


def run():
    attuned = pd.read_csv(ATTUNED_RATINGS)
    expert = score_predictions(merge_expert_deltas(attuned))
    report(expert, attuned)


if __name__ == "__main__":
    run()
