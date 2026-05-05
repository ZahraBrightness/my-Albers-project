"""
src/data/cleaner.py
-------------------
Column rename map, drop list, TICKER_MAP, and dataset-specific cleaning
functions.  The low-level `clean()` function is applied inside every loader;
the higher-level `clean_firm_panel()`, `clean_benchmark()`, and
`clean_sec_bulk()` are called by `run_full_cleaning_pipeline()`.

Run as a script to execute the full cleaning pipeline end-to-end:
    python src/data/cleaner.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

INTERIM_DIR = Path(__file__).resolve().parents[2] / "data" / "interim"

# ---------------------------------------------------------------------------
# Rename map  (applied via df.rename; missing source columns are silently skipped)
# ---------------------------------------------------------------------------

RENAME: dict[str, str] = {
    "Firm name":                                                    "firm_name",
    "Firm Name (Use Firms & Tickers SS)":                          "firm_name",
    "Ticker Symbol":                                               "ticker",
    "Date (by month)":                                             "date",
    "Date":                                                        "date",
    "Total Volatiliyt":                                            "total_vol",       # intentional typo in source
    "Operating Profit Margin ":                                    "op_margin",       # trailing space
    "Operating Profit Margin":                                     "op_margin",
    "ROE Return on Equity":                                        "roe",
    "ROE Return on  Equity":                                       "roe",             # double-space variant
    "ROIC Return on Capital Employed":                             "roic_ce",         # no trailing space (primary panel)
    "ROIC Return on Capital Employed ":                            "roic_ce",         # trailing-space variant
    "Return on Capital Employed":                                  "roic_ce",
    " Return on  Common Stock":                                    "ret_common",      # leading space
    "ROIC Return on Invested capital":                             "roic",
    "Pre-tax return on Net Operating Assets":                      "pretax_noa",
    "Pre-tax Return on Total Earning Assets":                      "pretax_ea",
    "Interest/Average Long-term Debt":                             "int_ltdebt",
    "LEVERAGE Tot LT Debt / Assets ":                              "leverage",        # trailing space
    "LEVERAGE Tot LT Debt / Assets":                               "leverage",
    "Long-term Debt/Book Equity":                                  "lt_debt_eq",
    "Interest Coverage Ratio":                                     "int_coverage",
    "Cash Conversion Cycle (Days)":                                "ccc",
    "Enterprise Value Multiple":                                   "ev_multiple",
    "Price/Operating Earnings (Diluted, Excl. EI)":               "price_op",
    "P/E (Diluted, Excl. EI)":                                    "pe",
    "Price/Sales":                                                 "price_sales",
    "Price/Book":                                                  "pb",
    "Shillers Cyclically Adjusted P/E Ratio":                      "cape",
    "Trailing P/E to Growth (PEG) ratio":                          "peg",
    "Market Beta":                                                 "mkt_beta",
    "Alpha":                                                       "alpha",
    "IVOL":                                                        "ivol",
    "Returns":                                                     "returns",
    " Returns":                                                    "returns",         # leading space
    # Benchmark median variants — renamed so benchmarks share column names with firms
    "After-tax Return on Average Common Equity_Median":            "roe",
    "After-tax Return on Invested Capital_Median":                 "roic",
    "Operating Profit Margin After Depreciation_Median":           "op_margin",
    "Total Long Term Debt Plus Debt in Current Liabilities/Total Assets_Median": "leverage",
    "RETURNS (Value Weighted) Industry Return":                    "industry_return",
}

# ---------------------------------------------------------------------------
# TICKER_MAP — uppercase substring of SEC 'Company Name' → exchange ticker
# Longer patterns are preferred (matched first) to avoid false positives.
# ---------------------------------------------------------------------------

TICKER_MAP: dict[str, str] = {
    "NEWELL BRANDS":  "NWL",
    "COCA COLA":      "KO",
    "PEPSICO":        "PEP",
    "PROCTER":        "PG",
    "COLGATE":        "CL",
    "KIMBERLY":       "KMB",
    "GENERAL MILLS":  "GIS",
    "HERSHEY":        "HSY",
    "KRAFT HEINZ":    "KHC",
    "CAMPBELL":       "CPB",
    "CONAGRA":        "CAG",
    "CLOROX":         "CLX",
    "MONDELEZ":       "MDLZ",
    "CHURCH":         "CHD",
    "WHIRLPOOL":      "WHR",
    "MATTEL":         "MAT",
    "SHARKNINJA":     "SN",
    "WAYFAIR":        "W",
    "RESTORATION":    "RH",
    "WILLIAMS SONOMA":"WSM",
    "BRADY":          "BRC",
    "JOHNSON":        "JNJ",
    "3M CO":          "MMM",
    "HELEN OF TROY":  "HELE",
    "REYNOLDS":       "REYN",
    "KELLANOVA":      "K",
}

# ---------------------------------------------------------------------------
# Columns to drop unconditionally (original names, pre-rename)
# ---------------------------------------------------------------------------

COLS_TO_DROP: list[str] = [
    "Ticker Symbol_1",
    "Historical CRSP PERMNO Link to COMPUSTAT Record",
    "Unnamed: 20",
    "Column21",
    "Column22",
]

# Columns forward-filled within ticker (fundamentals, max 3 monthly periods)
_FFILL_COLS = ["roe", "roic_ce", "roic", "pretax_noa", "int_coverage", "leverage", "ccc"]


# ---------------------------------------------------------------------------
# low-level helpers used by clean() and the high-level cleaners
# ---------------------------------------------------------------------------

def _deduplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the less-null copy when renaming creates duplicate column names."""
    keep: list[int] = []
    best: dict[str, int] = {}

    for i, col in enumerate(df.columns):
        if col not in best:
            best[col] = len(keep)
            keep.append(i)
        else:
            j = best[col]
            if df.iloc[:, i].isna().mean() < df.iloc[:, keep[j]].isna().mean():
                keep[j] = i

    return df.iloc[:, keep].copy()


def _assign_ticker(company_name: str) -> Optional[str]:
    """Return ticker via longest-match against TICKER_MAP (case-insensitive)."""
    upper = str(company_name).upper()
    for pattern in sorted(TICKER_MAP, key=len, reverse=True):
        if pattern in upper:
            return TICKER_MAP[pattern]
    return None


def _winsorize_per_ticker(
    df: pd.DataFrame,
    col: str,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.DataFrame:
    """Winsorize *col* at the given quantiles within each ticker group."""
    if col not in df.columns:
        return df

    result = df.copy()
    for _, group in df.groupby("ticker"):
        vals = group[col].dropna()
        if len(vals) < 4:          # too few obs — leave untouched
            continue
        q_lo = vals.quantile(lower)
        q_hi = vals.quantile(upper)
        result.loc[group.index, col] = group[col].clip(lower=q_lo, upper=q_hi)
    return result


# ---------------------------------------------------------------------------
# Base clean() — applied inside every loader
# ---------------------------------------------------------------------------

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uniform column-level pipeline:
    1. Drop COLS_TO_DROP (silent if absent).
    2. Rename per RENAME map (silent if absent).
    3. Resolve duplicate names by keeping the less-null copy.
    """
    df = df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns])
    df = df.rename(columns={k: v for k, v in RENAME.items() if k in df.columns})
    df = _deduplicate_columns(df)
    return df


# ---------------------------------------------------------------------------
# High-level dataset cleaners
# ---------------------------------------------------------------------------

def clean_firm_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline for the firm panel DataFrame.

    Steps
    -----
    1. Drop duplicate (ticker, date) — keep first.
    2. Sort by ticker, date ascending.
    3. Null out distorted ratio values:
         price_op  → NaN where |value| > 200  (operating income crossing zero)
         pe        → NaN where |value| > 500
    4. Winsorize per ticker at 1st / 99th percentile:
         ev_multiple, pb
    5. Forward-fill within ticker (≤ 3 monthly periods):
         roe, roic_ce, roic, pretax_noa, int_coverage, leverage, ccc
       Do NOT forward-fill: ev_multiple, pb, price_sales, pe, price_op
    6. Drop rows where roe AND int_coverage AND ev_multiple are ALL null.
    7. Add 'has_beta' (bool) — True where alpha, mkt_beta, ivol all non-null.
    8. Add 'quarter' column using pd.PeriodIndex(date, freq='QE').
    9. Save cleaned panel to data/interim/firm_panel_clean.csv.

    Returns
    -------
    pd.DataFrame  Cleaned firm panel.
    """
    n_before = len(df)

    # 1 — deduplicate
    df = df.drop_duplicates(subset=["ticker", "date"], keep="first")

    # Clean ticker: strip data-entry artefacts (e.g. 'REYN?' → 'REYN')
    df["ticker"] = df["ticker"].str.replace("?", "", regex=False).str.strip()

    # 2 — sort
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # 3 — null out extreme ratio distortions
    if "price_op" in df.columns:
        df.loc[df["price_op"].abs() > 200, "price_op"] = np.nan
    if "pe" in df.columns:
        df.loc[df["pe"].abs() > 500, "pe"] = np.nan

    # 4 — winsorize market-price multiples per ticker
    for col in ("ev_multiple", "pb"):
        df = _winsorize_per_ticker(df, col)

    # 5 — forward-fill fundamentals within ticker (max 3 periods)
    ffill_cols = [c for c in _FFILL_COLS if c in df.columns]
    for col in ffill_cols:
        df[col] = df.groupby("ticker")[col].ffill(limit=3)

    # 6 — drop fully-missing core rows
    core = [c for c in ("roe", "int_coverage", "ev_multiple") if c in df.columns]
    if core:
        all_null = df[core].isna().all(axis=1)
        df = df[~all_null].reset_index(drop=True)

    # 7 — has_beta flag
    beta_cols = [c for c in ("alpha", "mkt_beta", "ivol") if c in df.columns]
    df["has_beta"] = df[beta_cols].notna().all(axis=1) if beta_cols else False

    # 8 — quarter period  (PeriodIndex uses 'Q', not 'QE')
    df["quarter"] = pd.PeriodIndex(df["date"], freq="Q")

    # 9 — save
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    out = INTERIM_DIR / "firm_panel_clean.csv"
    df.to_csv(out, index=False)

    n_after = len(df)
    print(f"  clean_firm_panel: {n_before:,} → {n_after:,} rows  (saved {out.name})")
    return df


def clean_benchmark(df: pd.DataFrame, benchmark_name: str) -> pd.DataFrame:
    """
    Light cleaning for industry benchmark DataFrames.

    Steps
    -----
    - Drop any remaining 'Unnamed: *' columns.
    - Sort by date ascending.
    - Save to data/interim/{benchmark_name}_clean.csv.

    Parameters
    ----------
    df : pd.DataFrame
        Loaded benchmark DataFrame (already renamed by clean()).
    benchmark_name : str
        Used as the file stem — e.g. 'nodur', 'dur'.
    """
    n_before = len(df)

    # Drop any leftover unnamed columns
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    # Sort
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    out = INTERIM_DIR / f"{benchmark_name}_clean.csv"
    df.to_csv(out, index=False)

    print(f"  clean_benchmark [{benchmark_name}]: {n_before:,} → {len(df):,} rows"
          f"  (saved {out.name})")
    return df


def clean_sec_bulk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw SEC 8-K bulk DataFrame.

    Steps
    -----
    - Parse 'Date Filed' → 'date_filed' (datetime).
    - Add 'ticker' column via TICKER_MAP substring lookup.
    - Handle 'RH' via exact company-name match (too short for substring).
    - Filter to panel companies (non-null ticker).
    - Save to data/interim/sec_8k_panel.csv.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame as loaded directly from All_8K_Filings_*.xlsx.
    """
    n_before = len(df)

    # Parse and rename date column
    date_col = next((c for c in df.columns if "date filed" in c.lower()), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        if date_col != "date_filed":
            df = df.rename(columns={date_col: "date_filed"})

    # Assign ticker
    company_col = next(
        (c for c in df.columns if "company" in c.lower() or c.lower() == "name"),
        None,
    )
    if company_col:
        df["ticker"] = df[company_col].apply(_assign_ticker)
        # RH exact match (substring would produce false positives)
        rh_mask = df[company_col].str.strip().str.upper() == "RH"
        df.loc[rh_mask, "ticker"] = "RH"

    # Filter to panel companies
    if "ticker" in df.columns:
        df = df[df["ticker"].notna()].reset_index(drop=True)

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    out = INTERIM_DIR / "sec_8k_panel.csv"
    df.to_csv(out, index=False)

    n_after = len(df)
    print(f"  clean_sec_bulk: {n_before:,} → {n_after:,} rows  (saved {out.name})")
    return df


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_full_cleaning_pipeline() -> None:
    """
    Run all cleaners in sequence, printing before/after row counts.
    Re-runs the quality gate on the cleaned firm panel and prints the result.
    """
    # Lazy imports to avoid circular dependencies at module load time
    from data.loader import load_firm_panel, load_nodur_benchmark, load_dur_benchmark
    from data.quality import check_data_quality, print_quality_result

    RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

    print("=" * 62)
    print("  FULL CLEANING PIPELINE")
    print("=" * 62)

    # 1 — firm panel
    print("\n[1] Firm panel")
    fp_raw = load_firm_panel()
    print(f"  Loaded:  {fp_raw.shape[0]:,} rows × {fp_raw.shape[1]} cols")
    fp_clean = clean_firm_panel(fp_raw)

    # 2 — NODUR benchmark
    print("\n[2] NODUR benchmark")
    nb_raw = load_nodur_benchmark()
    print(f"  Loaded:  {nb_raw.shape[0]:,} rows × {nb_raw.shape[1]} cols")
    clean_benchmark(nb_raw, "nodur")

    # 3 — DUR benchmark
    print("\n[3] DUR benchmark")
    db_raw = load_dur_benchmark()
    print(f"  Loaded:  {db_raw.shape[0]:,} rows × {db_raw.shape[1]} cols")
    clean_benchmark(db_raw, "dur")

    # 4 — SEC bulk (loaded raw; let clean_sec_bulk handle filtering)
    print("\n[4] SEC 8-K bulk")
    sec_path = RAW_DIR / "All_8K_Filings_2023_2025.xlsx"
    sec_raw = pd.read_excel(sec_path, sheet_name="Sheet1")
    print(f"  Loaded:  {sec_raw.shape[0]:,} rows × {sec_raw.shape[1]} cols")
    clean_sec_bulk(sec_raw)

    # 5 — quality gate on cleaned firm panel
    print("\n[5] Quality gate — cleaned firm panel")
    result = check_data_quality(fp_clean, "firm_panel")
    print_quality_result(result, "firm_panel (post-clean)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    run_full_cleaning_pipeline()
