"""
tests/test_quality.py
----------------------
Pytest suite for src/data/quality.py.
Uses check_data_quality() — the single public gate function.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.quality import check_data_quality, FIRM_PANEL_REQUIRED

INTERIM_DIR = Path(__file__).resolve().parents[1] / "data" / "interim"

# Minimal set of columns that make a synthetic panel plausible
_ALL_REQUIRED = FIRM_PANEL_REQUIRED   # imported from quality.py directly


def _make_clean_df(n: int = 120) -> pd.DataFrame:
    """Return a synthetic DataFrame that satisfies all quality requirements."""
    tickers = ["NWL", "KO", "PEP", "PG", "CL"]
    rows_each = n // len(tickers)
    pieces = []
    for t in tickers:
        dates = pd.date_range("2018-01-31", periods=rows_each, freq="ME")
        piece = pd.DataFrame({"ticker": t, "date": dates})
        pieces.append(piece)
    df = pd.concat(pieces, ignore_index=True)

    rng = np.random.default_rng(42)
    # Fill every required numeric column with plausible values
    numeric_cols = [c for c in _ALL_REQUIRED if c not in ("ticker", "date")]
    for col in numeric_cols:
        df[col] = rng.uniform(0.01, 0.5, size=len(df))

    return df


# ── 1. Cleaned firm panel passes the quality gate ────────────────────────────

def test_quality_gate_passes_on_clean_data():
    path = INTERIM_DIR / "firm_panel_clean.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    result = check_data_quality(df, "firm_panel")
    assert result["success"], (
        f"Quality gate failed on cleaned firm panel.\n"
        f"Failures: {result['failures']}"
    )


# ── 2. Missing required columns cause a critical failure ─────────────────────

def test_missing_required_column_causes_failure():
    df = pd.DataFrame({
        "ticker": ["NWL"] * 100,
        "date":   pd.date_range("2017-01-01", periods=100, freq="ME"),
        "roe":    range(100),
    })
    result = check_data_quality(df, "firm_panel")
    # Must not pass — either failures list is non-empty or success is False
    assert not result["success"] or len(result["failures"]) > 0, (
        "Expected gate to detect missing required columns, but it passed."
    )


# ── 3. Too few rows triggers a critical failure ───────────────────────────────

def test_too_few_rows_causes_failure():
    # 10 rows — well below the 50-row CRITICAL threshold
    df = _make_clean_df(n=10)
    result = check_data_quality(df, "firm_panel")

    row_issues = [
        m for m in result["failures"] + result["warnings"]
        if "ROW" in m or "row" in m.lower() or "rows" in m.lower()
        or str(len(df)) in m
    ]
    assert not result["success"] or row_issues, (
        f"Expected row-count failure/warning for {len(df)}-row DataFrame, "
        f"but gate returned success with no row-count message."
    )


# ── 4. High null rate on 'roe' causes a warning or critical failure ───────────

def test_high_null_rate_causes_warning_or_failure():
    df = _make_clean_df(n=100)
    # Set 75% of 'roe' to NaN — above both the 60% CRITICAL threshold
    null_idx = df.sample(frac=0.75, random_state=0).index
    df.loc[null_idx, "roe"] = np.nan

    result = check_data_quality(df, "firm_panel")

    roe_issues = [
        m for m in result["failures"] + result["warnings"]
        if "roe" in m.lower()
    ]
    assert roe_issues, (
        "Expected a null-rate failure or warning for 'roe' at 75% null, "
        f"but got none.\nFailures: {result['failures']}\n"
        f"Warnings: {result['warnings']}"
    )


# ── 5. Fully valid synthetic DataFrame passes with no critical failures ────────

def test_clean_dataframe_passes_all_checks():
    df = _make_clean_df(n=120)
    result = check_data_quality(df, "firm_panel")
    assert result["success"], (
        f"Fully valid synthetic DataFrame should pass.\n"
        f"Failures: {result['failures']}"
    )
