"""
src/analysis/sc_diagnostic.py
------------------------------
Synthetic Control data-readiness diagnostic for NWL (treatment unit).

Checks every potential control firm for:
  - Date-range overlap with NWL
  - Pre-treatment observation count
  - exec_score null rate
  - Pre-treatment correlation with NWL

Saves results to data/processed/sc_diagnostic.csv.

Run with:
    python src/analysis/sc_diagnostic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT   = Path(__file__).resolve().parents[2]
PROCESSED_DIR  = PROJECT_ROOT / "data" / "processed"

TREATED_UNIT    = "NWL"
TREATMENT_DATE  = pd.Timestamp("2019-04-01")
MIN_OVERLAP     = 20      # minimum overlapping months for eligibility
MAX_NULL_RATE   = 0.20    # maximum exec_score null rate for eligibility


def run_diagnostic() -> pd.DataFrame:
    # ------------------------------------------------------------------ load
    scores_path = PROCESSED_DIR / "exec_scores_all_firms.csv"
    df = pd.read_csv(scores_path, parse_dates=["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    all_tickers = sorted(df["ticker"].dropna().unique())

    # ------------------------------------------------------------------ NWL
    nwl = df[df["ticker"] == TREATED_UNIT].copy()
    nwl_pre  = nwl[nwl["date"] <  TREATMENT_DATE]
    nwl_post = nwl[nwl["date"] >= TREATMENT_DATE]
    nwl_min_date = nwl["date"].min()
    nwl_max_date = nwl["date"].max()

    print("=" * 62)
    print(f"  SYNTHETIC CONTROL DIAGNOSTIC  —  Treatment unit: {TREATED_UNIT}")
    print("=" * 62)
    print(f"\n  Treatment date : {TREATMENT_DATE.date()}")
    print(f"\n  {TREATED_UNIT} — observation summary")
    print(f"  {'Total months':<35}: {len(nwl):,}")
    print(f"  {'Pre-treatment months':<35}: {len(nwl_pre):,}")
    print(f"  {'Post-treatment months':<35}: {len(nwl_post):,}")
    pre_null  = nwl_pre["exec_score"].isna().mean()
    post_null = nwl_post["exec_score"].isna().mean()
    print(f"  {'exec_score null rate (pre)':<35}: {pre_null*100:.1f}%")
    print(f"  {'exec_score null rate (post)':<35}: {post_null*100:.1f}%")

    # NWL pre-treatment exec_score series (for correlation)
    nwl_pre_scores = (
        nwl_pre.set_index("date")["exec_score"]
        .dropna()
    )

    # ------------------------------------------------------------------ controls
    controls = [t for t in all_tickers if t != TREATED_UNIT]

    print(f"\n{'─'*62}")
    print(f"  CONTROL FIRM ELIGIBILITY")
    print(f"{'─'*62}")

    hdr = (
        f"  {'Ticker':<8} {'Total':>7} {'Overlap':>9} "
        f"{'Pre-Tx':>8} {'Null%':>7}  Status"
    )
    print(hdr)
    print(f"  {'─'*8} {'─'*7} {'─'*9} {'─'*8} {'─'*7}  {'─'*8}")

    records = []
    for ticker in controls:
        sub = df[df["ticker"] == ticker].copy()
        total_months = len(sub)

        # Overlap: months within NWL's date range
        overlap = sub[
            (sub["date"] >= nwl_min_date) & (sub["date"] <= nwl_max_date)
        ]
        overlap_months = len(overlap)

        # Pre-treatment months
        pre = sub[sub["date"] < TREATMENT_DATE]
        pre_months = len(pre)

        # Null rate on full series
        null_rate = sub["exec_score"].isna().mean()

        # Eligibility
        eligible = (overlap_months >= MIN_OVERLAP) and (null_rate <= MAX_NULL_RATE)
        status = "ELIGIBLE" if eligible else "EXCLUDE"

        # Pre-treatment correlation with NWL
        ctrl_pre_scores = (
            pre.set_index("date")["exec_score"]
            .dropna()
        )
        shared_idx = nwl_pre_scores.index.intersection(ctrl_pre_scores.index)
        if len(shared_idx) >= 6:
            corr = float(
                nwl_pre_scores.loc[shared_idx]
                .corr(ctrl_pre_scores.loc[shared_idx])
            )
        else:
            corr = np.nan

        print(
            f"  {ticker:<8} {total_months:>7,} {overlap_months:>9,} "
            f"{pre_months:>8,} {null_rate*100:>6.1f}%  {status}"
        )

        records.append(
            dict(
                ticker=ticker,
                total_months=total_months,
                overlap_months=overlap_months,
                pre_treatment_months=pre_months,
                null_rate=round(null_rate, 4),
                pre_treatment_correlation=round(corr, 4) if not np.isnan(corr) else np.nan,
                eligible=eligible,
            )
        )

    diag_df = pd.DataFrame(records)

    # ------------------------------------------------------------------ donor pool
    eligible_df = diag_df[diag_df["eligible"]].copy()
    donor_pool  = sorted(eligible_df["ticker"].tolist())

    print(f"\n{'─'*62}")
    print(f"  DONOR POOL  ({len(donor_pool)} eligible firms)")
    print(f"{'─'*62}")
    print(f"  {donor_pool}")

    # ------------------------------------------------------------------ pre-treatment fit
    print(f"\n{'─'*62}")
    print(f"  PRE-TREATMENT FIT  (correlation with NWL exec_score)")
    print(f"  pre-treatment window: {nwl_min_date.date()} → "
          f"{(TREATMENT_DATE - pd.DateOffset(months=1)).date()}")
    print(f"{'─'*62}")

    fit_df = (
        eligible_df[["ticker", "pre_treatment_correlation"]]
        .dropna(subset=["pre_treatment_correlation"])
        .sort_values("pre_treatment_correlation", ascending=False)
        .reset_index(drop=True)
    )

    print(f"  {'Rank':<6} {'Ticker':<10} {'Corr(pre)'}")
    print(f"  {'─'*6} {'─'*10} {'─'*10}")
    for i, row in fit_df.iterrows():
        bar = "█" * int(abs(row["pre_treatment_correlation"]) * 20)
        sign = "+" if row["pre_treatment_correlation"] >= 0 else "-"
        print(
            f"  {i+1:<6} {row['ticker']:<10} "
            f"{row['pre_treatment_correlation']:>+.4f}  {bar}"
        )

    # ------------------------------------------------------------------ save
    out = PROCESSED_DIR / "sc_diagnostic.csv"
    diag_df.to_csv(out, index=False)
    print(f"\n  Saved → {out.name}")

    return diag_df


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    run_diagnostic()
