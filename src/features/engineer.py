"""
src/features/engineer.py
------------------------
Final feature-engineering stage: transition labels, lag features, and the
full model-ready dataset merge.

Three public functions
----------------------
build_transition_labels(exec_transitions_df, firm_panel_df)
    Tag each firm-month with whether a departure/transition occurs in the
    next 1–3 months; also flags CEO and CFO events separately.

add_lag_features(norm_panel)
    Within each ticker add 3-month / 12-month changes and 3-month rolling
    means for every z-score column.

build_model_ready_dataset()
    Orchestrates the full merge → lag features → label attachment → null
    filtering pipeline; saves data/processed/model_dataset.csv.

Run as a script:
    python src/features/engineer.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

INTERIM_DIR   = Path(__file__).resolve().parents[2] / "data" / "interim"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

# Event types that represent leadership exits (not arrivals)
_DEPARTURE_TYPES = {"departure", "transition"}

# Title keyword patterns for CEO / CFO detection
_CEO_KEYWORDS = {"CEO", "CHIEF EXECUTIVE"}
_CFO_KEYWORDS = {"CFO", "CHIEF FINANCIAL"}


# ---------------------------------------------------------------------------
# 1. build_transition_labels
# ---------------------------------------------------------------------------

def build_transition_labels(
    exec_transitions_df: Optional[pd.DataFrame],
    firm_panel_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add binary transition-label columns to firm_panel_df.

    For each firm-month row the function looks ahead 1–3 calendar months.
    If a departure or transition event exists for that ticker in the window,
    transition_within_3m = 1, else 0.  CEO and CFO events are flagged
    separately based on title keywords.

    Only 'departure' and 'transition' event_types are used; 'appointment'
    and 'unknown' are ignored.

    Parameters
    ----------
    exec_transitions_df : pd.DataFrame or None
        Loaded executive_transitions.csv (may be empty or None).
    firm_panel_df : pd.DataFrame
        Must include 'ticker' and 'date' columns.

    Returns
    -------
    pd.DataFrame
        firm_panel_df with added columns:
        transition_within_3m, ceo_event, cfo_event  (all int 0/1)
    """
    panel = firm_panel_df.copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")

    # Default: all zeros
    panel["transition_within_3m"] = 0
    panel["ceo_event"]            = 0
    panel["cfo_event"]            = 0

    # Guard: empty / None transitions table
    empty = (
        exec_transitions_df is None
        or (isinstance(exec_transitions_df, pd.DataFrame) and exec_transitions_df.empty)
    )
    if empty:
        warnings.warn(
            "exec_transitions_df is empty or None — "
            "all transition label columns set to 0.",
            UserWarning,
            stacklevel=2,
        )
        return panel

    events = exec_transitions_df.copy()
    events["date"] = pd.to_datetime(events["date"], errors="coerce")

    # Keep only departure / transition events
    events = events[events["event_type"].isin(_DEPARTURE_TYPES)].copy()
    if events.empty:
        warnings.warn(
            "No departure/transition events found in exec_transitions_df — "
            "all label columns set to 0.",
            UserWarning,
            stacklevel=2,
        )
        return panel

    # Pre-compute CEO / CFO flags on events
    def _is_ceo(titles: str) -> bool:
        if not isinstance(titles, str):
            return False
        upper = titles.upper()
        return any(kw in upper for kw in _CEO_KEYWORDS)

    def _is_cfo(titles: str) -> bool:
        if not isinstance(titles, str):
            return False
        upper = titles.upper()
        return any(kw in upper for kw in _CFO_KEYWORDS)

    events["_is_ceo"] = events["titles"].apply(_is_ceo)
    events["_is_cfo"] = events["titles"].apply(_is_cfo)

    # Build a per-ticker event lookup for fast iteration
    events_by_ticker = {
        ticker: grp.reset_index(drop=True)
        for ticker, grp in events.groupby("ticker")
    }

    for idx, row in panel.iterrows():
        ticker = row["ticker"]
        row_date = row["date"]
        if pd.isna(row_date) or ticker not in events_by_ticker:
            continue

        window_start = row_date + pd.DateOffset(months=1)
        window_end   = row_date + pd.DateOffset(months=3)

        grp = events_by_ticker[ticker]
        in_window = grp[(grp["date"] >= window_start) & (grp["date"] <= window_end)]

        if not in_window.empty:
            panel.at[idx, "transition_within_3m"] = 1
            if in_window["_is_ceo"].any():
                panel.at[idx, "ceo_event"] = 1
            if in_window["_is_cfo"].any():
                panel.at[idx, "cfo_event"] = 1

    return panel


# ---------------------------------------------------------------------------
# 2. add_lag_features
# ---------------------------------------------------------------------------

def add_lag_features(norm_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add within-ticker lag and rolling features for every z-score column and
    for exec_score (if present).

    For each {metric}_z column and for exec_score:
        {col}_chg3   = current value − value 3 months ago
        {col}_chg12  = current value − value 12 months ago
        {col}_roll3  = 3-month rolling mean

    All transformations are computed strictly within ticker groups to prevent
    cross-firm contamination.

    Parameters
    ----------
    norm_panel : pd.DataFrame
        Must include 'ticker', 'date', and one or more {metric}_z columns.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with lag/rolling columns appended.
    """
    df = norm_panel.copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Columns to lag: all z-score columns + exec_score
    z_cols    = [c for c in df.columns if c.endswith("_z")]
    lag_targets = z_cols + (["exec_score"] if "exec_score" in df.columns else [])

    pieces = []
    for ticker, group in df.groupby("ticker"):
        g = group.sort_values("date").copy()
        for col in lag_targets:
            if col not in g.columns:
                continue
            s = g[col]
            g[f"{col}_chg3"]  = s - s.shift(3)
            g[f"{col}_chg12"] = s - s.shift(12)
            g[f"{col}_roll3"] = s.rolling(window=3, min_periods=2).mean()
        pieces.append(g)

    df = pd.concat(pieces).sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 3. build_model_ready_dataset
# ---------------------------------------------------------------------------

def build_model_ready_dataset() -> pd.DataFrame:
    """
    Merge all feature sources, add lags, attach transition labels, and save
    the model-ready dataset.

    Steps
    -----
    1. Load firm_panel_clean.csv, normalized_panel.csv, exec_scores_all_firms.csv.
    2. Left-join all three on (ticker, date) anchored to the normalized panel
       so no firm-months are lost from the z-score data.
    3. Call add_lag_features on the merged panel.
    4. Try to load executive_transitions.csv → build_transition_labels.
       If absent, set all label columns to 0 with a warning.
    5. Drop rows where > 50% of feature columns are null.
    6. Save to data/processed/model_dataset.csv.
    7. Print final shape, label distribution, feature completeness by firm,
       and complete feature column list.

    Returns
    -------
    pd.DataFrame
        Model-ready dataset.
    """
    print("Loading source files …")

    firm_panel = pd.read_csv(
        INTERIM_DIR / "firm_panel_clean.csv", parse_dates=["date"]
    )
    norm_panel = pd.read_csv(
        PROCESSED_DIR / "normalized_panel.csv", parse_dates=["date"]
    )
    exec_scores = pd.read_csv(
        PROCESSED_DIR / "exec_scores_all_firms.csv", parse_dates=["date"]
    )

    print(f"  firm_panel    : {firm_panel.shape[0]:,} rows × {firm_panel.shape[1]} cols")
    print(f"  norm_panel    : {norm_panel.shape[0]:,} rows × {norm_panel.shape[1]} cols")
    print(f"  exec_scores   : {exec_scores.shape[0]:,} rows × {exec_scores.shape[1]} cols")

    # ------------------------------------------------------------------ merge
    print("\nMerging …")

    # Normalise date types
    for df in (firm_panel, norm_panel, exec_scores):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Anchor on norm_panel (z-scores are the core feature set)
    # Bring in exec_score and category scores; ticker is the join key so keep it
    score_merge_cols = ["ticker", "date"] + [
        c for c in exec_scores.columns if c not in ("ticker", "date")
    ]
    merged = norm_panel.merge(
        exec_scores[score_merge_cols],
        on=["ticker", "date"],
        how="left",
        suffixes=("", "_scores"),
    )

    # Bring in firm-level fundamentals from firm_panel (non-duplicate columns)
    fp_cols = ["ticker", "date"] + [
        c for c in firm_panel.columns
        if c not in merged.columns and c not in ("ticker", "date")
    ]
    merged = merged.merge(firm_panel[fp_cols], on=["ticker", "date"], how="left")

    print(f"  merged        : {merged.shape[0]:,} rows × {merged.shape[1]} cols")

    # -------------------------------------------------------------- lag features
    print("Adding lag features …")
    merged = add_lag_features(merged)
    print(f"  with lags     : {merged.shape[0]:,} rows × {merged.shape[1]} cols")

    # --------------------------------------------------------- transition labels
    print("Building transition labels …")
    et_path = INTERIM_DIR / "executive_transitions.csv"
    if et_path.exists():
        exec_transitions = pd.read_csv(et_path)
        if exec_transitions.empty:
            exec_transitions = None
    else:
        exec_transitions = None
        warnings.warn(
            f"executive_transitions.csv not found at {et_path} — "
            "label columns set to 0.",
            UserWarning,
        )

    merged = build_transition_labels(exec_transitions, merged)

    # --------------------------------------------------- drop high-null rows
    print("Filtering high-null rows …")
    # Feature columns = everything except keys and labels
    non_feature = {
        "ticker", "date", "quarter", "has_beta", "industry",
        "transition_within_3m", "ceo_event", "cfo_event",
    }
    feature_cols = [c for c in merged.columns if c not in non_feature]
    null_frac    = merged[feature_cols].isna().mean(axis=1)
    before       = len(merged)
    merged       = merged[null_frac <= 0.50].reset_index(drop=True)
    print(f"  dropped {before - len(merged):,} rows with > 50% null features "
          f"({len(merged):,} remaining)")

    # ------------------------------------------------------------------- save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "model_dataset.csv"
    merged.to_csv(out, index=False)

    # ------------------------------------------------------------------ report
    print(f"\n{'='*62}")
    print(f"  Final dataset: {merged.shape[0]:,} rows × {merged.shape[1]} cols")
    print(f"{'='*62}")

    # Label distribution
    print("\n  Label distribution — transition_within_3m:")
    vc = merged["transition_within_3m"].value_counts().sort_index()
    total = len(merged)
    for val, cnt in vc.items():
        print(f"    {val}: {cnt:,}  ({cnt/total*100:.1f}%)")

    # CEO / CFO breakdown
    print("\n  ceo_event / cfo_event positive counts:")
    print(f"    ceo_event : {merged['ceo_event'].sum():,}")
    print(f"    cfo_event : {merged['cfo_event'].sum():,}")

    # Feature completeness by firm
    print("\n  Feature completeness by firm (% non-null across feature cols):")
    completeness = (
        merged.groupby("ticker")[feature_cols]
        .apply(lambda g: g.notna().mean().mean() * 100)
        .sort_values(ascending=False)
        .reset_index()
    )
    completeness.columns = ["ticker", "pct_complete"]
    col_w = max(len(t) for t in completeness["ticker"]) + 2
    for _, row in completeness.iterrows():
        bar = "#" * int(row["pct_complete"] / 5)
        print(f"    {row['ticker']:<{col_w}} {row['pct_complete']:5.1f}%  {bar}")

    # All feature columns
    print(f"\n  Feature columns ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols, 1):
        print(f"    {i:>3}. {col}")

    print(f"\n  Saved → {out}")
    return merged


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    print("=" * 62)
    print("  BUILD MODEL-READY DATASET")
    print("=" * 62 + "\n")

    dataset = build_model_ready_dataset()

    print(f"\nFinal shape          : {dataset.shape}")
    print(f"Label distribution   :")
    print(dataset["transition_within_3m"].value_counts().to_string())
    print("\nDone.")
