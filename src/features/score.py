"""
src/features/score.py
----------------------
Composite executive-performance scoring built on industry-normalized z-scores.

Three public functions
----------------------
score_executive(ticker, norm_panel)
    Compute monthly profitability / health / market / efficiency category scores
    and a weighted composite exec_score for one firm.

score_all_firms(norm_panel)
    Apply score_executive to every ticker; save combined scores; print ranking.

score_by_era(scores_df)
    Average scores within three strategic eras; print NWL era comparison.

Run as a script:
    python src/features/score.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATEGORY_WEIGHTS: Dict[str, float] = {
    "profitability": 0.25,
    "health":        0.30,
    "market":        0.20,
    "efficiency":    0.25,
}

CATEGORY_METRICS: Dict[str, List[str]] = {
    "profitability": ["roe_z", "roic_ce_z", "pretax_noa_z"],
    "health":        ["int_coverage_z", "int_ltdebt_z", "leverage_z"],
    "market":        ["ev_multiple_z", "pb_z", "price_sales_z"],
    "efficiency":    ["ccc_z"],
}

# Columns where lower is operationally better — negate before averaging so
# that higher always means better across all metrics universally.
NEGATE: List[str] = ["int_ltdebt_z", "leverage_z", "ccc_z"]

ERA_BOUNDS: Dict[str, tuple[str, str]] = {
    "pre_polk_exit": ("2017-12", "2019-03"),
    "saligram_era":  ("2019-04", "2022-12"),
    "post_2022":     ("2023-01", "2024-12"),
}


# ---------------------------------------------------------------------------
# 1. score_executive
# ---------------------------------------------------------------------------

def score_executive(ticker: str, norm_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute monthly composite scores for one firm.

    Steps
    -----
    1. Filter norm_panel to *ticker*, sort by date.
    2. Negate NEGATE columns so higher always means better.
    3. For each category, take equal-weighted mean of available z-score
       columns (skipna=True — partial data still contributes a score).
    4. exec_score = weighted average of category scores, renormalizing
       weights when a category has no data for a given row.

    Parameters
    ----------
    ticker : str
    norm_panel : pd.DataFrame
        Output of build_full_normalized_panel() (normalized_panel.csv).

    Returns
    -------
    pd.DataFrame
        Columns: ticker, date, score_profitability, score_health,
                 score_market, score_efficiency, exec_score
    """
    df = norm_panel[norm_panel["ticker"] == ticker].copy()
    df = df.sort_values("date").reset_index(drop=True)

    if df.empty:
        return pd.DataFrame(columns=[
            "ticker", "date",
            "score_profitability", "score_health",
            "score_market", "score_efficiency", "exec_score",
        ])

    # Negate lower-is-better columns in-place on the working copy
    for col in NEGATE:
        if col in df.columns:
            df[col] = -df[col]

    # Compute one category score per row (equal-weighted, skip nulls)
    cat_score_cols = []
    for cat, metrics in CATEGORY_METRICS.items():
        present = [m for m in metrics if m in df.columns]
        col_name = f"score_{cat}"
        if present:
            df[col_name] = df[present].mean(axis=1, skipna=True)
        else:
            df[col_name] = np.nan
        cat_score_cols.append(col_name)

    # exec_score: renormalize weights when a category score is NaN on a row
    base_weights = {f"score_{cat}": w for cat, w in CATEGORY_WEIGHTS.items()}

    def _weighted_mean(row: pd.Series) -> float:
        available = {c: base_weights[c] for c in cat_score_cols if pd.notna(row[c])}
        if not available:
            return np.nan
        total_w = sum(available.values())
        return sum(row[c] * w for c, w in available.items()) / total_w

    df["exec_score"] = df[cat_score_cols].apply(
        lambda row: _weighted_mean(row), axis=1
    )

    keep = ["ticker", "date"] + cat_score_cols + ["exec_score"]
    return df[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. score_all_firms
# ---------------------------------------------------------------------------

def score_all_firms(norm_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Apply score_executive to every ticker in norm_panel.

    Saves combined scores to data/processed/exec_scores_all_firms.csv.
    Prints a ranking table (descending by mean exec_score).
    Prints NWL's rank and score explicitly.

    Parameters
    ----------
    norm_panel : pd.DataFrame
        Output of build_full_normalized_panel().

    Returns
    -------
    pd.DataFrame
        Combined scores for all firms.
    """
    tickers = sorted(norm_panel["ticker"].dropna().unique())
    pieces  = [score_executive(t, norm_panel) for t in tickers]
    scores  = pd.concat(pieces, ignore_index=True)
    scores  = scores.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "exec_scores_all_firms.csv"
    scores.to_csv(out, index=False)

    # Ranking table
    ranking = (
        scores.groupby("ticker")["exec_score"]
        .mean()
        .reset_index()
        .rename(columns={"exec_score": "avg_exec_score"})
        .sort_values("avg_exec_score", ascending=False)
        .reset_index(drop=True)
    )
    ranking["rank"] = ranking.index + 1

    col_w = max(len(t) for t in ranking["ticker"]) + 2
    print(f"\n  {'Firm':<{col_w}} {'avg_exec_score':>15}  {'rank':>5}")
    print(f"  {'-'*col_w} {'-'*15}  {'-'*5}")
    for _, row in ranking.iterrows():
        print(f"  {row['ticker']:<{col_w}} {row['avg_exec_score']:>15.4f}  {int(row['rank']):>5}")

    nwl_row = ranking[ranking["ticker"] == "NWL"]
    if not nwl_row.empty:
        nwl_rank  = int(nwl_row["rank"].iloc[0])
        nwl_score = nwl_row["avg_exec_score"].iloc[0]
        print(f"\n  NWL rank: {nwl_rank} of {len(ranking)}  |  avg exec_score: {nwl_score:.4f}")

    print(f"\n  Saved {len(scores):,} rows → {out.name}")
    return scores


# ---------------------------------------------------------------------------
# 3. score_by_era
# ---------------------------------------------------------------------------

def score_by_era(scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Average exec_score and all category scores within three strategic eras.

    Era definitions
    ---------------
    pre_polk_exit : 2017-12 → 2019-03
    saligram_era  : 2019-04 → 2022-12
    post_2022     : 2023-01 → 2024-12

    Saves to data/processed/exec_scores_by_era.csv.
    Prints NWL's scores across all 3 eras side by side.

    Parameters
    ----------
    scores_df : pd.DataFrame
        Output of score_all_firms().

    Returns
    -------
    pd.DataFrame
        Columns: ticker, era, exec_score, score_profitability, score_health,
                 score_market, score_efficiency
    """
    df = scores_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    score_cols = [
        "exec_score", "score_profitability", "score_health",
        "score_market", "score_efficiency",
    ]

    era_frames = []
    for era_name, (start, end) in ERA_BOUNDS.items():
        start_ts = pd.Period(start, freq="M").to_timestamp(how="start")
        end_ts   = pd.Period(end,   freq="M").to_timestamp(how="end")

        mask    = (df["date"] >= start_ts) & (df["date"] <= end_ts)
        era_df  = df[mask].copy()
        era_df["era"] = era_name

        agg = (
            era_df.groupby(["ticker", "era"])[score_cols]
            .mean()
            .reset_index()
        )
        era_frames.append(agg)

    era_panel = pd.concat(era_frames, ignore_index=True)
    era_panel = era_panel.sort_values(["ticker", "era"]).reset_index(drop=True)

    # Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "exec_scores_by_era.csv"
    era_panel.to_csv(out, index=False)

    # Print NWL era comparison
    nwl = era_panel[era_panel["ticker"] == "NWL"].copy()
    era_order = list(ERA_BOUNDS.keys())
    nwl["_order"] = nwl["era"].map({e: i for i, e in enumerate(era_order)})
    nwl = nwl.sort_values("_order").drop(columns="_order")

    if not nwl.empty:
        print(f"\n  NWL scores by era")
        print(f"  {'Era':<20} {'exec_score':>11} {'profitability':>14} "
              f"{'health':>8} {'market':>8} {'efficiency':>11}")
        print(f"  {'-'*20} {'-'*11} {'-'*14} {'-'*8} {'-'*8} {'-'*11}")
        for _, row in nwl.iterrows():
            print(
                f"  {row['era']:<20} "
                f"{row['exec_score']:>11.4f} "
                f"{row['score_profitability']:>14.4f} "
                f"{row['score_health']:>8.4f} "
                f"{row['score_market']:>8.4f} "
                f"{row['score_efficiency']:>11.4f}"
            )

        # Confirm core finding
        pre  = nwl[nwl["era"] == "pre_polk_exit"]["exec_score"].values
        sal  = nwl[nwl["era"] == "saligram_era"]["exec_score"].values
        if len(pre) and len(sal):
            delta = sal[0] - pre[0]
            direction = "IMPROVED" if delta > 0 else "DECLINED"
            print(f"\n  NWL exec_score {direction} by {abs(delta):.4f} "
                  f"(pre_polk_exit → saligram_era)")

    print(f"\n  Saved {len(era_panel)} era rows → {out.name}")
    return era_panel[["ticker", "era"] + score_cols]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    norm_path = PROCESSED_DIR / "normalized_panel.csv"
    print(f"Loading {norm_path.name} …")
    norm_panel = pd.read_csv(norm_path, parse_dates=["date"])
    print(f"  {norm_panel.shape[0]:,} rows × {norm_panel.shape[1]} cols  "
          f"|  {norm_panel['ticker'].nunique()} tickers\n")

    print("=" * 62)
    print("  FIRM RANKINGS  (mean exec_score, descending)")
    print("=" * 62)
    scores = score_all_firms(norm_panel)

    print("\n" + "=" * 62)
    print("  ERA ANALYSIS")
    print("=" * 62)
    era_panel = score_by_era(scores)

    print("\nDone.")
