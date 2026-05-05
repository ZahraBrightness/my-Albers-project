"""
tests/test_loader.py
---------------------
Pytest suite for src/data/loader.py.
"""

import pandas as pd
import pytest

from data.loader import (
    load_all,
    load_firm_panel,
    load_nodur_benchmark,
    load_dur_benchmark,
    load_sec_curated,
)

_EXPECTED_KEYS = [
    "primary_panel",
    "extended_panel",
    "nwl_extended_firm",
    "nwl_extended_benchmark",
    "nodur_bench",
    "dur_bench",
    "nwl_vs_nodur_firm",
    "nwl_vs_nodur_benchmark",
    "nwl_vs_dur_firm",
    "nwl_vs_dur_benchmark",
    "sec_curated",
]


@pytest.fixture(scope="module")
def all_datasets():
    """Load all datasets once per module to avoid repeated I/O."""
    return load_all()


# ── 1. load_all returns a dict with the expected keys ─────────────────────────

def test_load_all_returns_dict(all_datasets):
    assert isinstance(all_datasets, dict), "load_all() must return a dict"
    missing = [k for k in _EXPECTED_KEYS if k not in all_datasets]
    assert not missing, f"Missing keys in load_all() result: {missing}"


# ── 2. Rename map has been applied ────────────────────────────────────────────

def test_rename_map_applied(all_datasets):
    df = all_datasets["primary_panel"]
    assert "roe"    in df.columns, "'roe' column missing — rename not applied"
    assert "ticker" in df.columns, "'ticker' column missing — rename not applied"
    assert "date"   in df.columns, "'date' column missing — rename not applied"
    assert "ROE Return on Equity"  not in df.columns, \
        "Original 'ROE Return on Equity' still present — rename not applied"
    assert "Firm name" not in df.columns, \
        "Original 'Firm name' still present — rename not applied"


# ── 3. Date column is datetime ────────────────────────────────────────────────

def test_date_column_is_datetime(all_datasets):
    df = all_datasets["primary_panel"]
    assert pd.api.types.is_datetime64_any_dtype(df["date"]), (
        f"'date' column dtype is {df['date'].dtype}, expected datetime64"
    )


# ── 4. No required column is entirely null ────────────────────────────────────

def test_no_required_column_all_null(all_datasets):
    df = all_datasets["primary_panel"]
    cols = ["ticker", "date", "roe", "leverage", "ev_multiple"]
    for col in cols:
        assert col in df.columns, f"Required column '{col}' not found"
        null_rate = df[col].isna().mean()
        assert null_rate < 1.0, (
            f"Column '{col}' is 100% null (null rate = {null_rate:.2f})"
        )


# ── 5. Row counts within expected ranges ──────────────────────────────────────

def test_row_counts_within_expected_range(all_datasets):
    primary    = all_datasets["primary_panel"]
    nodur      = all_datasets["nodur_bench"]
    dur        = all_datasets["dur_bench"]
    sec_cur    = all_datasets["sec_curated"]

    assert 2000 <= len(primary) <= 2500, (
        f"primary_panel has {len(primary)} rows, expected 2000–2500"
    )
    assert 80 <= len(nodur) <= 90, (
        f"nodur_bench has {len(nodur)} rows, expected 80–90"
    )
    assert 80 <= len(dur) <= 90, (
        f"dur_bench has {len(dur)} rows, expected 80–90"
    )
    assert len(sec_cur) >= 50, (
        f"sec_curated has {len(sec_cur)} rows, expected >= 50"
    )
