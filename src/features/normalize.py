"""
src/features/normalize.py
--------------------------
Industry-normalized z-score features for the executive-transition project.

Four public functions
---------------------
assign_industry(ticker)
    Return 'NODUR' or 'DUR'; raise ValueError for unknown tickers.

compute_gaps(firm_df, benchmark_df, ticker)
    Inner-join firm and benchmark on date; compute firm − benchmark for each
    of the 10 metrics.  Returns columns:
        date, ticker, {metric}_firm, {metric}_bench, {metric}_gap

compute_rolling_z(gaps_df)
    Rolling 12-month z-score (min_periods=6), winsorized at ±3.
    Appends {metric}_z columns; keeps all gap columns.

build_full_normalized_panel(firm_panel, nodur_bench, dur_bench)
    Orchestrates the pipeline for all 26 tickers; concatenates results;
    saves to data/processed/normalized_panel.csv.

Run as a script:
    python src/features/normalize.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

INTERIM_DIR   = Path(__file__).resolve().parents[2] / "data" / "interim"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

# ---------------------------------------------------------------------------
# Metric registry
# key → (firm_col, benchmark_col, direction)
#   direction: +1 higher-is-better, -1 lower-is-better, 0 neutral / valuation
# ---------------------------------------------------------------------------

METRICS: Dict[str, Tuple[str, str, int]] = {
    "roe":          ("roe",          "roe",           1),
    "roic_ce":      ("roic_ce",      "roic_ce",       1),
    "pretax_noa":   ("pretax_noa",   "pretax_noa",    1),
    "int_coverage": ("int_coverage", "int_coverage",  1),
    "int_ltdebt":   ("int_ltdebt",   "int_ltdebt",   -1),
    "leverage":     ("leverage",     "leverage",      -1),
    "ev_multiple":  ("ev_multiple",  "ev_multiple",   0),
    "pb":           ("pb",           "pb",            0),
    "price_sales":  ("price_sales",  "price_sales",   0),
    "ccc":          ("ccc",          "ccc",           -1),
}

# ---------------------------------------------------------------------------
# Industry assignment
# ---------------------------------------------------------------------------

INDUSTRY_MAP: Dict[str, str] = {
    # Non-Durables
    "KO":   "NODUR", "PEP":  "NODUR", "PG":   "NODUR", "CL":   "NODUR",
    "KMB":  "NODUR", "GIS":  "NODUR", "HSY":  "NODUR", "KHC":  "NODUR",
    "CPB":  "NODUR", "CAG":  "NODUR", "CLX":  "NODUR", "MDLZ": "NODUR",
    "CHD":  "NODUR",
    # Durables
    "NWL":  "DUR",   "WHR":  "DUR",   "MAT":  "DUR",   "SN":   "DUR",
    "W":    "DUR",   "RH":   "DUR",   "WSM":  "DUR",   "BRC":  "DUR",
    "JNJ":  "DUR",   "MMM":  "DUR",   "HELE": "DUR",   "REYN": "DUR",
    "K":    "DUR",
}


def assign_industry(ticker: str) -> str:
    """
    Return the industry group ('NODUR' or 'DUR') for *ticker*.

    Raises
    ------
    ValueError
        If *ticker* is not in the known 26-company panel.
    """
    result = INDUSTRY_MAP.get(ticker.upper())
    if result is None:
        raise ValueError(
            f"Unknown ticker '{ticker}'. "
            f"Valid tickers: {sorted(INDUSTRY_MAP)}"
        )
    return result


# ---------------------------------------------------------------------------
# Gap computation
# ---------------------------------------------------------------------------

def compute_gaps(
    firm_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    """
    Compute raw metric gaps (firm − benchmark) for one ticker.

    Merges firm and benchmark rows on date (inner join), then for each metric
    computes:
        {metric}_gap = {metric}_firm − {metric}_bench

    If either value is null the gap is null.

    Parameters
    ----------
    firm_df : pd.DataFrame
        Rows for a single ticker from the cleaned firm panel (must have 'date').
    benchmark_df : pd.DataFrame
        Cleaned industry benchmark DataFrame (must have 'date').
    ticker : str
        Populates the 'ticker' column in the returned DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns: date, ticker, {metric}_firm, {metric}_bench, {metric}_gap
        for every metric whose columns are present on both sides.
    """
    firm      = firm_df.copy()
    benchmark = benchmark_df.copy()

    firm["date"]      = pd.to_datetime(firm["date"],      errors="coerce")
    benchmark["date"] = pd.to_datetime(benchmark["date"], errors="coerce")

    # Build column lists for each side (date + relevant metric cols only)
    firm_metric_cols  = [fc for fc, (fc_name, _, _) in
                         [(m, v) for m, v in METRICS.items()]
                         if True]   # resolved below
    # Cleaner: collect unique firm and bench column names actually needed
    firm_needed  = ["date"] + list({fc  for fc, _, _ in METRICS.values() if fc  in firm.columns})
    bench_needed = ["date"] + list({bc  for _, bc, _ in METRICS.values() if bc  in benchmark.columns})

    merged = (
        firm[firm_needed]
        .merge(benchmark[bench_needed], on="date", how="inner", suffixes=("_firm", "_bench"))
    )

    if merged.empty:
        return pd.DataFrame(columns=["date", "ticker"])

    out = pd.DataFrame({"date": merged["date"], "ticker": ticker})

    for metric, (firm_col, bench_col, _) in METRICS.items():
        # After the suffixed merge, shared names become col_firm / col_bench.
        # Non-shared names (firm_col != bench_col) keep their original names.
        if firm_col == bench_col:
            fc_merged = f"{firm_col}_firm"
            bc_merged = f"{bench_col}_bench"
        else:
            fc_merged = firm_col  if firm_col  in merged.columns else f"{firm_col}_firm"
            bc_merged = bench_col if bench_col in merged.columns else f"{bench_col}_bench"

        if fc_merged in merged.columns:
            out[f"{metric}_firm"] = merged[fc_merged].values
        if bc_merged in merged.columns:
            out[f"{metric}_bench"] = merged[bc_merged].values

        if fc_merged in merged.columns and bc_merged in merged.columns:
            out[f"{metric}_gap"] = merged[fc_merged].values - merged[bc_merged].values

    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Rolling z-score
# ---------------------------------------------------------------------------

def compute_rolling_z(gaps_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling 12-month z-score columns for every {metric}_gap column.

    Algorithm (per gap column, computed within ticker order):
        rolling_mean = gap.rolling(12, min_periods=6).mean()
        rolling_std  = gap.rolling(12, min_periods=6).std()
        z            = (gap − rolling_mean) / rolling_std
        z            clipped to [−3, +3]

    Result column is named {metric}_z.  All original gap columns are kept.

    Parameters
    ----------
    gaps_df : pd.DataFrame
        Output of compute_gaps(); assumed sorted by date ascending for one
        ticker.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with {metric}_z columns appended.
    """
    df = gaps_df.sort_values("date").reset_index(drop=True)

    for col in [c for c in df.columns if c.endswith("_gap")]:
        metric = col[: -len("_gap")]
        series = df[col]

        roll      = series.rolling(window=12, min_periods=6)
        roll_mean = roll.mean()
        roll_std  = roll.std()

        z = (series - roll_mean) / roll_std
        df[f"{metric}_z"] = z.clip(lower=-3.0, upper=3.0)

    return df


# ---------------------------------------------------------------------------
# Full panel builder
# ---------------------------------------------------------------------------

def build_full_normalized_panel(
    firm_panel: pd.DataFrame,
    nodur_bench: pd.DataFrame,
    dur_bench: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the complete industry-normalized panel for all 26 tickers.

    For each ticker in firm_panel:
      1. assign_industry() → pick NODUR or DUR benchmark.
      2. compute_gaps() → inner-join on date, compute raw gaps.
      3. compute_rolling_z() → rolling z-scores winsorized at ±3.
      4. Concatenate all per-ticker results.

    Prints valid z-score counts per metric.
    Saves to data/processed/normalized_panel.csv.

    Parameters
    ----------
    firm_panel : pd.DataFrame   Cleaned firm panel.
    nodur_bench : pd.DataFrame  Cleaned NODUR benchmark.
    dur_bench : pd.DataFrame    Cleaned DUR benchmark.

    Returns
    -------
    pd.DataFrame
        Combined normalized panel.
    """
    bench_map = {"NODUR": nodur_bench, "DUR": dur_bench}
    pieces    = []
    tickers   = sorted(firm_panel["ticker"].dropna().unique())

    for ticker in tickers:
        # Skip tickers not in the panel map
        try:
            industry = assign_industry(ticker)
        except ValueError:
            print(f"  [SKIP] '{ticker}' not in INDUSTRY_MAP")
            continue

        firm_rows = firm_panel[firm_panel["ticker"] == ticker].copy()

        # Skip tickers with too few rows to produce any z-scores
        if len(firm_rows) < 6:
            print(f"  [SKIP] '{ticker}' has only {len(firm_rows)} rows (< 6 minimum)")
            continue

        benchmark = bench_map[industry]
        gaps      = compute_gaps(firm_rows, benchmark, ticker)

        if gaps.empty:
            print(f"  [WARN] '{ticker}' produced 0 gap rows after inner join — skipped")
            continue

        zscores = compute_rolling_z(gaps)
        pieces.append(zscores)

    if not pieces:
        raise RuntimeError("No normalized data produced — check input DataFrames.")

    panel = pd.concat(pieces, ignore_index=True)
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Print valid z-score counts per metric
    z_cols = [c for c in panel.columns if c.endswith("_z")]
    print(f"\n  Valid z-score counts per metric ({len(panel):,} total rows):")
    for zc in z_cols:
        n_valid = panel[zc].notna().sum()
        print(f"    {zc:<22} {n_valid:>5,}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "normalized_panel.csv"
    panel.to_csv(out, index=False)

    print(
        f"\n  build_full_normalized_panel: {len(panel):,} rows × {len(panel.columns)} cols"
        f"  |  {panel['ticker'].nunique()} tickers  |  {len(z_cols)} z-score cols"
        f"  (saved {out.name})"
    )
    return panel


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    print("Loading cleaned interim files …")

    firm_panel  = pd.read_csv(INTERIM_DIR / "firm_panel_clean.csv",  parse_dates=["date"])
    nodur_bench = pd.read_csv(INTERIM_DIR / "nodur_clean.csv",       parse_dates=["date"])
    dur_bench   = pd.read_csv(INTERIM_DIR / "dur_clean.csv",         parse_dates=["date"])

    print(f"  firm_panel  : {firm_panel.shape[0]:,} rows × {firm_panel.shape[1]} cols")
    print(f"  nodur_bench : {nodur_bench.shape[0]:,} rows × {nodur_bench.shape[1]} cols")
    print(f"  dur_bench   : {dur_bench.shape[0]:,} rows × {dur_bench.shape[1]} cols")

    print("\nBuilding normalized panel …")
    panel = build_full_normalized_panel(firm_panel, nodur_bench, dur_bench)

    z_cols = [c for c in panel.columns if c.endswith("_z")]
    print("\nZ-score summary statistics:")
    if z_cols:
        print(panel[z_cols].describe().round(3).to_string())

    print("\nPer-ticker row counts:")
    print(panel.groupby("ticker").size().to_string())

    print("\nDone.")
