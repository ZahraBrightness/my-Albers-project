"""
tests/test_normalize.py
------------------------
Pytest suite for src/features/normalize.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from features.normalize import assign_industry, compute_gaps, compute_rolling_z

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


@pytest.fixture(scope="module")
def norm_panel():
    """Load normalized panel once per module."""
    return pd.read_csv(
        PROCESSED_DIR / "normalized_panel.csv",
        parse_dates=["date"],
    )


# ── 1. assign_industry — NODUR firms ─────────────────────────────────────────

def test_assign_industry_nodur_firms():
    assert assign_industry("KO")  == "NODUR"
    assert assign_industry("PEP") == "NODUR"
    assert assign_industry("PG")  == "NODUR"
    assert assign_industry("CL")  == "NODUR"


# ── 2. assign_industry — DUR firms ───────────────────────────────────────────

def test_assign_industry_dur_firms():
    assert assign_industry("NWL") == "DUR"
    assert assign_industry("WHR") == "DUR"
    assert assign_industry("MAT") == "DUR"


# ── 3. assign_industry — unknown ticker raises ValueError ────────────────────

def test_assign_industry_unknown_raises_error():
    with pytest.raises(ValueError):
        assign_industry("AAPL")


# ── 4. Z-score columns are present in the panel ──────────────────────────────

def test_z_score_columns_present_for_all_firms(norm_panel):
    required_z = [
        "roe_z", "int_coverage_z", "leverage_z",
        "ev_multiple_z", "pb_z", "price_sales_z",
    ]
    missing = [col for col in required_z if col not in norm_panel.columns]
    assert not missing, (
        f"Z-score columns missing from normalized panel: {missing}"
    )


# ── 5. All z-scores are within ±3 (winsorize bounds) ─────────────────────────

def test_z_scores_within_winsorize_bounds(norm_panel):
    z_cols = [c for c in norm_panel.columns if c.endswith("_z")]
    assert z_cols, "No _z columns found in normalized panel."

    violations = {}
    for col in z_cols:
        vals = norm_panel[col].dropna()
        out = vals[(vals < -3.01) | (vals > 3.01)]
        if not out.empty:
            violations[col] = len(out)

    assert not violations, (
        f"Z-score values outside ±3.01 found — winsorizing may have failed:\n"
        f"{violations}"
    )


# ── 6. NWL outperforms DUR peers on interest coverage ────────────────────────

def test_nwl_int_coverage_z_mostly_positive(norm_panel):
    """
    NWL's interest coverage gap vs the DUR benchmark is consistently positive
    — confirmed in fig1 (panel 2 green fill) and by the raw data.
    The z-score should be > 50% positive.
    """
    nwl_z = norm_panel.loc[
        norm_panel["ticker"] == "NWL", "int_coverage_z"
    ].dropna()
    assert len(nwl_z) > 0, "No int_coverage_z values found for NWL."
    pct_positive = (nwl_z > 0).mean()
    assert pct_positive > 0.50, (
        f"Expected > 50% of NWL int_coverage_z to be positive "
        f"(NWL outperforms DUR benchmark), got {pct_positive:.2%}."
    )


# ── 7. Gap is null when either firm or benchmark value is null ────────────────

def test_gap_is_null_when_either_input_is_null(norm_panel):
    # Rows where roe_firm is null → roe_gap must also be null
    if "roe_firm" in norm_panel.columns and "roe_gap" in norm_panel.columns:
        firm_null_mask = norm_panel["roe_firm"].isna()
        if firm_null_mask.any():
            leaking = norm_panel.loc[firm_null_mask, "roe_gap"].notna()
            assert not leaking.any(), (
                f"{leaking.sum()} rows have roe_firm=null but roe_gap is non-null — "
                "null propagation not working."
            )

    # Rows where roe_bench is null → roe_gap must also be null
    if "roe_bench" in norm_panel.columns and "roe_gap" in norm_panel.columns:
        bench_null_mask = norm_panel["roe_bench"].isna()
        if bench_null_mask.any():
            leaking = norm_panel.loc[bench_null_mask, "roe_gap"].notna()
            assert not leaking.any(), (
                f"{leaking.sum()} rows have roe_bench=null but roe_gap is non-null — "
                "null propagation not working."
            )
