"""
pipeline.py
-----------
Main entry point. Runs all pipeline stages in sequence.

Usage:
    python pipeline.py
    python pipeline.py --skip-validation
    python pipeline.py --skip-atc-validation
    python pipeline.py --robustness
    python pipeline.py --visualize
    python pipeline.py --visualize-only
"""

import argparse
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from constants import DATA_DIR
from src import (
    preprocessing, atc, sentiment, regression, attenuation, validation,
    atc_validation, robustness,
)
from src.visualizations import attenuation_plots, descriptive_plots, correlation_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ABSA SET pipeline")
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip the attenuation expert validation stage",
    )
    parser.add_argument(
        "--skip-atc-validation",
        action="store_true",
        help="Skip the ATC validation stage (requires grok.csv, gpt.csv, atc_expert_labels.csv)",
    )
    parser.add_argument(
        "--robustness",
        action="store_true",
        help=(
            "Run the stage 7 robustness analyses (permutation control and "
            "s-sensitivity tables). Off by default: the permutation control "
            "re-attenuates the corpus once per permutation."
        ),
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate all plots after pipeline completes",
    )
    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help="Skip pipeline stages and only generate plots (requires existing outputs)",
    )
    return parser.parse_args()


def _timed(fn, *args, **kwargs):
    t0 = time.time()
    fn(*args, **kwargs)
    print(f"  Completed in {time.time() - t0:.1f}s\n")


def main() -> None:
    args = parse_args()

    #Generate folder for processed data if does not exist
    (DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("ABSA Validity-Weighted SET Pipeline")
    print("=" * 60)

    if not args.visualize_only:
        _timed(preprocessing.run)
        _timed(atc.run)
        _timed(sentiment.run)
        _timed(regression.run)
        _timed(attenuation.run)

        if not args.skip_atc_validation:
            _timed(atc_validation.run)
        else:
            print(" Skipping ATC validation stage.\n")

        if not args.skip_validation:
            _timed(validation.run)
        else:
            print(" Skipping attenuation validation stage.\n")

        if args.robustness:
            _timed(robustness.run)

    if args.visualize or args.visualize_only:
        _timed(descriptive_plots.run)
        _timed(attenuation_plots.run)
        _timed(correlation_plots.run)

    print("=" * 60)
    print(" Pipeline complete.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()