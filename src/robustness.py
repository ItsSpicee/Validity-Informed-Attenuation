"""
src/robustness.py
-----------------
Stage 7: robustness analyses supporting Sections 4.3 and 4.4. Opt-in — this
stage is not part of a default pipeline run, because the permutation control
re-attenuates the corpus N_PERMUTATIONS times.

Three independent analyses:

  Permutation control
      Every result in Section 4.3 shows that attenuation behaves as designed,
      but none of them establish that the behaviour depends on a review's
      *actual* off-topic content. A mechanism that adjusted ratings by an
      arbitrary review-specific amount would also produce structured-looking
      correlations. Permuting D_misc among the reviews that have off-topic
      content breaks the correspondence between measured density and applied
      attenuation while leaving the marginal distribution and the analytic
      population intact, so it is the control condition that separates the two.
      Reports subgroup size variation when bins are recomputed, plus a second
      comparison with bins fixed at the observed operating point. Both use the
      same shuffles and predictions.
      The null is therefore "the amount of off-topic content is unrelated to
      which review it is", not "off-topic content is absent" -- the graded
      claim Sections 4.3.3 and 4.4 actually make.

  Sensitivity across s
      Table A — agreement between the deltas produced at different s.
      Table B — the Section 4.3.3 modulator correlations recomputed at each s.
      Table C — expert paired-comparison accuracy at each s.
      Together these show the framework's behaviour is not an artifact of the
      particular exponent the grid search happened to select, which matters
      because the s objective has several near-equivalent minima.

  Bootstrap professor stability
      Resamples the 206 held-out professors with replacement across N iterations,
      recomputing modulator correlations and expert accuracy on each bootstrap
      sample. Reports means, SDs, and 95% CIs, showing that the results are not
      an artifact of which professors happen to fall on the held-out side.

Loads the trained model without refitting and prints all summary reports to the console.

Usage:
    python pipeline.py --robustness
    python -m src.robustness --skip-permutation
    python -m src.robustness --n-permutations 20 --s-values 0.4 0.83
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.stats import pearsonr, spearmanr

from constants import (
    CATBOOST_FINAL_MODEL,
    EXPERT_LABELS_PATH,
    FINAL_EMOTIONS,
    MISC_D_COL,
    N_PERMUTATIONS,
    PERMUTATION_SEED,
    SENSITIVITY_S_VALUES,
    S_VALUE,
)
from src.attenuation import attenuate
from src.splits import heldout_mask, professor_split
from src.validation import PAIR_KEYS, condition_masks, merge_expert_deltas, score_predictions


# ==============================================================================
# SHARED HELPERS
# ==============================================================================

def deltas_for(model: CatBoostRegressor, df: pd.DataFrame, s: float) -> pd.DataFrame:
    """Per-review deltas at one `s`, mirroring attenuation.run()."""
    full = attenuate(model, df, s)
    return full[full[MISC_D_COL] > 0].reset_index(drop=True)


def _expert_counts(expert, reference=None):
    """Score dynamic bins, or bins from an observed pair-indexed reference."""
    if reference is None:
        expert = expert.loc[expert.evaluable]
        masks = condition_masks(expert)
    else:
        expert = expert.reindex(reference.index)
        if not expert.evaluable.fillna(False).all():
            raise ValueError("Model deltas unavailable for an observed reference pair")
        masks = condition_masks(reference)
    return {
        name.lower(): (int(expert.loc[mask, "correct"].sum()), int(mask.sum()))
        for name, mask in masks.items()
    }


def expert_accuracy(attuned, verbose=False):
    """Per-condition (correct, n), recomputing bins from supplied deltas."""
    expert = score_predictions(merge_expert_deltas(attuned, verbose=verbose))
    return _expert_counts(expert.set_index(PAIR_KEYS))


def _modulator_corrs(delta: np.ndarray, misc_d: np.ndarray) -> dict[str, float]:
    """Delta-vs-D_misc correlations, split by direction of adjustment."""
    out: dict[str, float] = {}
    for name, mask in (("pos", delta > 0), ("neg", delta < 0)):
        if mask.sum() >= 2:
            out[f"{name}_pearson"]  = pearsonr(delta[mask], misc_d[mask])[0]
            out[f"{name}_spearman"] = spearmanr(delta[mask], misc_d[mask])[0]
        else:
            out[f"{name}_pearson"] = out[f"{name}_spearman"] = np.nan
    return out


# ==============================================================================
# PERMUTATION CONTROL
# ==============================================================================

def _permutation_measure(
    model: CatBoostRegressor,
    df_variant: pd.DataFrame,
    test_profs,
    true_misc_d: pd.Series,
    verbose: bool = False,
    reference: pd.DataFrame | None = None,
) -> tuple[dict, pd.DataFrame | None]:
    """Attenuate once; return metrics and scored pairs for both bin definitions.

    Correlations are measured against the TRUE densities, joined on `review_id`,
    not against whatever densities this frame carries. That is the whole point of
    the control: the paper claims the adjustment a review receives tracks its
    *real* off-topic content, so the null has to ask whether a mechanism driven
    by shuffled densities still reproduces that relationship. Correlating the
    permuted deltas against the permuted densities would instead re-measure the
    mechanism against its own input and return roughly the observed value at
    every permutation, testing nothing.

    The join is on values, not on population: the permutation is restricted to
    non-zero densities (see `permutation_control`), so the `D_misc > 0` subset is
    already identical to the observed one in every permutation. What the join
    supplies is each review's TRUE density as the correlate.
    """
    full = attenuate(model, df_variant, S_VALUE)

    # Expert accuracy mirrors the pipeline, which scores against the misc subset
    # of this frame. Because the permutation preserves the zero pattern, that
    # subset is the same set of reviews every iteration and n holds at the
    # observed value; it is still returned so a drift would be visible rather
    # than assumed away.
    metrics = {}
    expert = None
    if all(path.exists() for path in EXPERT_LABELS_PATH):
        misc_subset = full[full[MISC_D_COL] > 0].reset_index(drop=True)
        expert = score_predictions(
            merge_expert_deltas(misc_subset, verbose=verbose)
        ).set_index(PAIR_KEYS)
        if reference is not None:
            if set(expert.index[expert.evaluable]) != set(reference.index):
                raise ValueError("Permutation changed the evaluable expert-pair population")
        for suffix, bins in (("", None), ("_fixed", reference)):
            for name, (correct, n) in _expert_counts(expert, bins).items():
                key = "expert_accuracy" if name == "overall" else f"expert_{name}"
                metrics[key + suffix] = correct / n if n else np.nan
                metrics[key + suffix + "_n"] = n

    true_d = full["review_id"].map(true_misc_d)
    mask = heldout_mask(full, test_profs).to_numpy(dtype=bool) & (true_d > 0).to_numpy()
    metrics.update(_modulator_corrs(
        full.loc[mask, "weighting_delta"].to_numpy(), true_d[mask].to_numpy()
    ))
    return metrics, expert


def _summarise_permutation(
    perm: pd.DataFrame, col: str, observed: float, direction: str
) -> dict:
    vals = perm[col].dropna().to_numpy()
    if not np.isfinite(observed) or len(vals) == 0:
        print(f"\n  {col}: undefined (empty observed group or no valid permutations)")
        return dict(metric=col, observed=observed, perm_mean=np.nan,
                    perm_sd=np.nan, p_value=np.nan, n_valid=len(vals))
    # One-sided: how often does the permuted null reach what we observed?
    hits = int((vals >= observed).sum()) if direction == "greater" else int((vals <= observed).sum())
    # +1 correction: a permutation p-value can never legitimately be zero.
    p = (hits + 1) / (len(vals) + 1)

    print(f"\n  {col}")
    print(f"    observed        : {observed:.4f}")
    print(f"    permuted mean   : {vals.mean():.4f}  (SD {vals.std(ddof=1):.4f})")
    print(f"    permuted range  : [{vals.min():.4f}, {vals.max():.4f}]")
    print(f"    valid permutations: {len(vals)}/{len(perm)}")
    print(f"    permutations reaching observed: {hits}/{len(vals)}   p = {p:.4f}")

    return {
        "metric": col, "observed": observed, "perm_mean": vals.mean(),
        "perm_sd": vals.std(ddof=1), "p_value": p, "n_valid": len(vals),
    }


def permutation_control(
    model: CatBoostRegressor,
    df: pd.DataFrame,
    test_profs,
    n_permutations: int = N_PERMUTATIONS,
    seed: int = PERMUTATION_SEED,
) -> pd.DataFrame:
    """Permute D_misc, re-attenuate, and compare against the observed result."""
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    true_misc_d = df.set_index("review_id")[MISC_D_COL]

    print(f"\n  Observed (true D_misc), s = {S_VALUE}...")
    observed, reference = _permutation_measure(
        model, df, test_profs, true_misc_d, verbose=True
    )
    if reference is not None:
        reference = reference.loc[reference.evaluable].copy()
        for key in ("expert_accuracy", "expert_coarse", "expert_fine"):
            print(f"    {key}: {observed[key]:.4f} (n = {observed[key + '_n']})")
    for key in ("pos_pearson", "neg_pearson"):
        print(f"    {key}: {observed[key]:.4f}")

    print(f"\n  Running {n_permutations} permutations of D_misc...")
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_permutations):
        df_perm = df.copy()
        # Permuting the column permutes it everywhere it is consumed: as a model
        # feature and as the zeta driver inside _apply_down_weighting. Permuting
        # only one would leave a back-channel through which true density still
        # influences the adjustment.
        #
        # The shuffle is restricted to reviews whose density is already non-zero,
        # so zeros stay zero. That subset IS the analytic population: Section 4.3
        # reports on `D_misc > 0` and all expert pairs fall inside it, so
        # exchangeability only has to hold there. Permuting the full column
        # instead relocates the zeros, which silently resamples the reporting
        # population every iteration -- the expert comparison then scores only
        # those pairs whose BOTH reviews happened to draw a positive density
        # (~0.606^2 x 80 = ~29 of 77), inflating the null's variance and making
        # the resulting p-value a small-sample artifact.
        vals = df_perm[MISC_D_COL].to_numpy().copy()
        nz = np.flatnonzero(vals > 0)
        vals[nz] = rng.permutation(vals[nz])
        df_perm[MISC_D_COL] = vals

        metrics, _ = _permutation_measure(
            model, df_perm, test_profs, true_misc_d, reference=reference
        )
        rows.append({"permutation": i, **metrics})

        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{n_permutations}")

    perm = pd.DataFrame(rows)

    print("\n" + "=" * 66)
    print("  PERMUTATION CONTROL")
    print("=" * 66)
    summary_rows = []
    if reference is not None:
        print("  Test 1: bins recomputed after each permutation")
        for key in ("expert_accuracy", "expert_coarse", "expert_fine"):
            sizes = perm[key + "_n"]
            print(f"\n  {key} subgroup size: observed={observed[key + '_n']}; "
                  f"mean={sizes.mean():.2f}, SD={sizes.std(ddof=1):.2f}, "
                  f"min={sizes.min()}, max={sizes.max()}, empty={(sizes == 0).sum()}")
            summary_rows.append(_summarise_permutation(perm, key, observed[key], "greater"))
        print(f"\n  Test 2: bins fixed at observed s={S_VALUE} (same shuffles)")
        print("  Overall accuracy is identical to Test 1; only subgroup tests are repeated.")
        for key in ("expert_coarse_fixed", "expert_fine_fixed"):
            print(f"\n  {key} fixed subgroup size: {observed[key + '_n']}")
            summary_rows.append(_summarise_permutation(perm, key, observed[key], "greater"))
    for key, direction in (("pos_pearson", "greater"), ("neg_pearson", "less")):
        summary_rows.append(_summarise_permutation(perm, key, observed[key], direction))
    summary = pd.DataFrame(summary_rows)
    return summary


# ==============================================================================
# SENSITIVITY ACROSS S
# ==============================================================================

def table_delta_agreement(delta_by_s: dict[float, np.ndarray]) -> pd.DataFrame:
    """Table A — do different s values produce the same adjustments?"""
    rows = []
    for s_a, s_b in itertools.combinations(delta_by_s, 2):
        a, b = delta_by_s[s_a], delta_by_s[s_b]
        rows.append({
            "$s_a$": s_a,
            "$s_b$": s_b,
            "Pearson $r$": pearsonr(a, b)[0],
            "Spearman $\\rho$": spearmanr(a, b)[0],
            "Mean $|\\Delta_a - \\Delta_b|$": np.abs(a - b).mean(),
            "Max $|\\Delta_a - \\Delta_b|$": np.abs(a - b).max(),
        })
    return pd.DataFrame(rows)


def table_modulator_corrs(
    delta_by_s: dict[float, np.ndarray], misc_d: np.ndarray
) -> pd.DataFrame:
    """Table B — is the Section 4.3.3 result an artifact of the tuned s?"""
    rows = []
    for s, d in delta_by_s.items():
        row: dict = {"$s$": s}
        for name, mask, sym in (("pos", d > 0, "^+"), ("neg", d < 0, "^-")):
            if mask.sum() >= 2:
                row[f"$\\Delta{sym}$ Pearson $r$"]      = pearsonr(d[mask], misc_d[mask])[0]
                row[f"$\\Delta{sym}$ Spearman $\\rho$"] = spearmanr(d[mask], misc_d[mask])[0]
            else:
                row[f"$\\Delta{sym}$ Pearson $r$"]      = np.nan
                row[f"$\\Delta{sym}$ Spearman $\\rho$"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def table_expert_accuracy(
    model: CatBoostRegressor, df: pd.DataFrame, s_values: list[float]
) -> pd.DataFrame:
    """Accuracy across s with evaluable pairs and bins fixed at s=1.0.

    Match pairs by review IDs. Fail if a setting cannot score a reference pair,
    rather than silently changing the denominator.
    """
    def scored_pairs(s):
        return score_predictions(
            merge_expert_deltas(deltas_for(model, df, s), verbose=False)
        ).set_index(PAIR_KEYS)

    reference = scored_pairs(1.0)
    reference = reference.loc[reference.evaluable]
    masks = condition_masks(reference)
    counts = {name: int(mask.sum()) for name, mask in masks.items()}
    rows = []
    for s in s_values:
        expert = reference if s == 1.0 else scored_pairs(s).reindex(reference.index)
        if not expert.evaluable.fillna(False).all():
            raise ValueError(f"s={s}: model deltas unavailable for an s=1.0 reference pair")
        accuracies = {
            name: int(expert.loc[mask, "correct"].sum()) / counts[name]
            if counts[name] else np.nan
            for name, mask in masks.items()
        }
        rows.append({
            "$s$": s,
            "Pairs": counts["Overall"],
            "Overall": accuracies["Overall"],
            "Coarse pairs": counts["Coarse"],
            "Coarse": accuracies["Coarse"],
            "Fine pairs": counts["Fine"],
            "Fine": accuracies["Fine"],
        })
    return pd.DataFrame(rows)



def sensitivity_across_s(
    model: CatBoostRegressor,
    df: pd.DataFrame,
    test_profs,
    s_values: list[float],
) -> list[tuple[pd.DataFrame, str]]:
    """Build Tables A, B and C. Returns (table, title) tuples for console reporting."""
    print(f"\n  s values: {s_values}   (pipeline uses S_VALUE = {S_VALUE})")

    # Tables A and B are validation metrics, so they use held-out professors,
    # matching the reporting convention in Section 4.3.
    delta_by_s: dict[float, np.ndarray] = {}
    misc_d: np.ndarray | None = None
    for s in s_values:
        attuned = deltas_for(model, df, s)
        mask = heldout_mask(attuned, test_profs).to_numpy(dtype=bool)
        delta_by_s[s] = attuned.loc[mask, "weighting_delta"].to_numpy()
        if misc_d is None:
            misc_d = attuned.loc[mask, MISC_D_COL].to_numpy()
            print(f"  Held-out misc reviews: {int(mask.sum()):,}")

    tables = [
        (table_delta_agreement(delta_by_s), "Delta agreement across s (held-out professors)"),
        (table_modulator_corrs(delta_by_s, misc_d), "Modulator correlations across s (held-out professors)"),
    ]

    # Expert pairs span all professors; report accuracy on the full sample.
    if all(path.exists() for path in EXPERT_LABELS_PATH):
        tables.append((
            table_expert_accuracy(model, df, s_values),
            "Expert paired-comparison accuracy across s (full sample; bins fixed at s=1.0)",
        ))
    else:
        missing = [str(path) for path in EXPERT_LABELS_PATH if not path.exists()]
        print(f"\n  Skipping Table C — expert labels not found: {', '.join(missing)}")

    return tables


# ==============================================================================
# BOOTSTRAP PROFESSOR STABILITY
# ==============================================================================

N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42


def bootstrap_professor_stability(
    model: CatBoostRegressor,
    df: pd.DataFrame,
    test_profs,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Resample held-out professors with replacement; show metric stability.

    The held-out set has 206 professors. Each iteration draws 206 professors
    with replacement, recomputes modulator correlations and expert accuracy
    on the resampled set, and records the results. No re-fitting or
    re-attenuating — the attenuated data is computed once from the true model.
    """
    print(f"\n  Bootstrap professor stability ({n_bootstrap} iterations)...")

    full = attenuate(model, df, S_VALUE)
    misc = full[full[MISC_D_COL] > 0].reset_index(drop=True)
    held_mask = heldout_mask(misc, test_profs).to_numpy(dtype=bool)
    held = misc[held_mask].copy()

    test_prof_list = list(test_profs)
    rng = np.random.default_rng(seed)

    rows = []
    for i in range(n_bootstrap):
        sampled_profs = rng.choice(test_prof_list, size=len(test_prof_list), replace=True)
        # Build the bootstrap sample: all reviews belonging to sampled professors.
        # A professor drawn k times contributes k copies of their reviews.
        parts = []
        for p in sampled_profs:
            parts.append(held[held["prof_ID"] == p])
        boot = pd.concat(parts, ignore_index=True)

        delta = boot["weighting_delta"].to_numpy()
        misc_d = boot[MISC_D_COL].to_numpy()
        corrs = _modulator_corrs(delta, misc_d)

        rows.append({
            "iteration": i,
            "pos_pearson": corrs["pos_pearson"],
            "neg_pearson": corrs["neg_pearson"],
            "pos_spearman": corrs["pos_spearman"],
            "neg_spearman": corrs["neg_spearman"],
        })

        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{n_bootstrap}")

    boot_df = pd.DataFrame(rows)

    # Report
    print("\n" + "=" * 66)
    print("  BOOTSTRAP PROFESSOR STABILITY (held-out professors)")
    print("=" * 66)

    summary_rows = []
    for col, label in [
        ("pos_pearson",   "Δ+ vs D_misc (Pearson r)"),
        ("neg_pearson",   "Δ- vs D_misc (Pearson r)"),
        ("pos_spearman",  "Δ+ vs D_misc (Spearman ρ)"),
        ("neg_spearman",  "Δ- vs D_misc (Spearman ρ)"),
    ]:
        vals = boot_df[col].dropna().to_numpy()
        if len(vals) == 0:
            continue
        lo, hi = np.percentile(vals, [2.5, 97.5])
        print(f"\n  {label}")
        print(f"    mean: {vals.mean():.4f}   SD: {vals.std(ddof=1):.4f}")
        print(f"    95% CI: [{lo:.4f}, {hi:.4f}]")
        summary_rows.append({
            "metric": label,
            "mean": vals.mean(),
            "sd": vals.std(ddof=1),
            "ci_lo": lo,
            "ci_hi": hi,
        })

    summary = pd.DataFrame(summary_rows)
    return summary


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def run(
    skip_permutation: bool = False,
    skip_sensitivity: bool = False,
    skip_bootstrap: bool = False,
    n_permutations: int = N_PERMUTATIONS,
    n_bootstrap: int = N_BOOTSTRAP,
    s_values: list[float] | None = None,
) -> None:
    print("\n=== Stage 7: Robustness ===")

    print("Loading model and data...")
    model = CatBoostRegressor()
    model.load_model(str(CATBOOST_FINAL_MODEL), format="cbm")
    df = pd.read_csv(FINAL_EMOTIONS)
    _, test_profs = professor_split(df)

    if not skip_sensitivity:
        resolved = sorted(set(s_values if s_values else SENSITIVITY_S_VALUES + [S_VALUE]))
        for table, title in sensitivity_across_s(model, df, test_profs, resolved):
            print(f"\n{'=' * 66}\n  {title}\n{'=' * 66}\n")
            print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    else:
        print("  Skipping sensitivity tables.")

    if not skip_permutation:
        permutation_control(model, df, test_profs, n_permutations=n_permutations)
    else:
        print("\n  Skipping permutation control.")

    if not skip_bootstrap:
        bootstrap_professor_stability(
            model, df, test_profs, n_bootstrap=n_bootstrap
        )
    else:
        print("\n  Skipping bootstrap professor stability.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stage 7: robustness analyses")
    parser.add_argument("--skip-permutation", action="store_true")
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--s-values", type=float, nargs="+", default=None)
    args = parser.parse_args()

    run(
        skip_permutation=args.skip_permutation,
        skip_sensitivity=args.skip_sensitivity,
        skip_bootstrap=args.skip_bootstrap,
        n_permutations=args.n_permutations,
        n_bootstrap=args.n_bootstrap,
        s_values=args.s_values,
    )
