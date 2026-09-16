"""
src/splits.py
-------------
Single source of truth for the professor-level data split.

The split is defined once here and reused by regression (model training),
attenuation, and the reporting visualizations, so the three stages cannot drift
apart. Drift would reintroduce circularity: results are only clean if the
professors used to fit the model are the same ones excluded from the reported
metrics.

Splitting is on professor rather than review because reviews of the same
professor are correlated; a review-level split would leak across the boundary.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from constants import RANDOM_STATE, TEST_SIZE

PROF_COL = "prof_ID"


def professor_split(df: pd.DataFrame) -> tuple[list, list]:
    """Return (train_profs, test_profs).

    Professor IDs are drawn from the full frame, not from any filtered subset,
    so that every caller sees the same partition regardless of the rows it holds.
    """
    prof_ids = df[PROF_COL].unique()
    train_profs, test_profs = train_test_split(
        prof_ids, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return train_profs, test_profs


def train_rows(df: pd.DataFrame, train_profs) -> pd.DataFrame:
    """Rows belonging to training professors."""
    return df[df[PROF_COL].isin(train_profs)]


def heldout_rows(df: pd.DataFrame, test_profs) -> pd.DataFrame:
    """Rows belonging to held-out professors."""
    return df[df[PROF_COL].isin(test_profs)]


def heldout_mask(df: pd.DataFrame, test_profs) -> pd.Series:
    """Boolean mask marking held-out professors."""
    return df[PROF_COL].isin(test_profs)
