"""
tests/test_cleaner.py
----------------------
Pytest suite validating the output of clean_firm_panel()
against the saved data/interim/firm_panel_clean.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

INTERIM_DIR = Path(__file__).resolve().parents[1] / "data" / "interim"


@pytest.fixture(scope="module")
def firm_panel():
    """Load the cleaned firm panel once per module."""
    return pd.read_csv(
        INTERIM_DIR / "firm_panel_clean.csv",
        parse_dates=["date"],
    )


# ── 1. No duplicate (ticker, date) pairs ─────────────────────────────────────

def test_no_duplicate_ticker_date_pairs(firm_panel):
    n_dupes = firm_panel.duplicated(subset=["ticker", "date"]).sum()
    assert n_dupes == 0, (
        f"Found {n_dupes} duplicate (ticker, date) pairs after cleaning."
    )


# ── 2. price_op within ±200 ──────────────────────────────────────────────────

def test_price_op_within_bounds(firm_panel):
    if "price_op" not in firm_panel.columns:
        pytest.skip("'price_op' column not present in cleaned panel.")
    valid = firm_panel["price_op"].dropna()
    violators = valid[valid.abs() > 200]
    assert len(violators) == 0, (
        f"{len(violators)} rows have |price_op| > 200 after cleaning:\n"
        f"{violators.values}"
    )


# ── 3. pe within ±500 ────────────────────────────────────────────────────────

def test_pe_within_bounds(firm_panel):
    if "pe" not in firm_panel.columns:
        pytest.skip("'pe' column not present in cleaned panel.")
    valid = firm_panel["pe"].dropna()
    violators = valid[valid.abs() > 500]
    assert len(violators) == 0, (
        f"{len(violators)} rows have |pe| > 500 after cleaning:\n"
        f"{violators.values}"
    )


# ── 4. No rows where roe, int_coverage, and ev_multiple are ALL null ──────────

def test_no_all_null_ratio_rows(firm_panel):
    core = [c for c in ("roe", "int_coverage", "ev_multiple")
            if c in firm_panel.columns]
    if len(core) < 3:
        pytest.skip(f"Not all core columns present; found: {core}")
    all_null = firm_panel[core].isna().all(axis=1)
    n_all_null = all_null.sum()
    assert n_all_null == 0, (
        f"{n_all_null} rows have roe, int_coverage, AND ev_multiple all null — "
        "expected these to be dropped by clean_firm_panel()."
    )


# ── 5. has_beta column exists and contains only boolean values ────────────────

def test_has_beta_column_exists_and_is_bool(firm_panel):
    assert "has_beta" in firm_panel.columns, \
        "'has_beta' column missing from cleaned panel."
    assert firm_panel["has_beta"].isin([True, False]).all(), (
        "'has_beta' contains values other than True/False:\n"
        f"{firm_panel['has_beta'].unique()}"
    )


# ── 6. quarter column exists and has no nulls ─────────────────────────────────

def test_quarter_column_exists(firm_panel):
    assert "quarter" in firm_panel.columns, \
        "'quarter' column missing from cleaned panel."
    n_null = firm_panel["quarter"].isna().sum()
    assert n_null == 0, (
        f"'quarter' column has {n_null} null values — expected none."
    )


# ── 7. Row count is above minimum threshold ───────────────────────────────────

def test_row_count_above_minimum(firm_panel):
    assert len(firm_panel) >= 2000, (
        f"Cleaned firm panel has only {len(firm_panel)} rows — "
        "expected >= 2000. Possible unexpected drop during cleaning."
    )
