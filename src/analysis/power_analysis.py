"""
src/analysis/power_analysis.py
-------------------------------
Statistical power analysis for:
  A) XGBoost classification task (n=1,902 firm-months)
  B) Difference-in-Differences t-test on a small panel (n=12 per group)

Run with:
    python src/analysis/power_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from statsmodels.stats.power import TTestIndPower
except ImportError:
    import subprocess, sys as _sys
    subprocess.check_call([_sys.executable, "-m", "pip", "install", "statsmodels", "-q"])
    from statsmodels.stats.power import TTestIndPower

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

ALPHA      = 0.05
TARGET_PWR = 0.80
TWO_SIDED  = "two-sided"

EFFECT_SIZES = [
    (0.2, "Small"),
    (0.3, "Med-S"),
    (0.5, "Medium"),
    (0.8, "Large"),
]


def run_power_analysis() -> None:
    print("=" * 62)
    print("  POWER ANALYSIS")
    print("=" * 62)

    # ------------------------------------------------------------------ load
    df = pd.read_csv(PROCESSED_DIR / "model_dataset.csv", parse_dates=["date"])
    target = "transition_within_3m"
    df = df[df[target].notna()]

    n_total   = len(df)
    n_pos     = int(df[target].sum())
    n_neg     = n_total - n_pos
    base_rate = n_pos / n_total

    print(f"\n  Dataset statistics")
    print(f"  {'Total firm-months':<35}: {n_total:,}")
    print(f"  {'Positive labels (transition)':<35}: {n_pos:,}  ({base_rate*100:.1f}%)")
    print(f"  {'Negative labels (no transition)':<35}: {n_neg:,}  ({(1-base_rate)*100:.1f}%)")
    print(f"  {'Random-classifier baseline AP':<35}: {base_rate:.4f}")

    analysis = TTestIndPower()

    # ------------------------------------------------------------------ classification power
    print(f"\n{'─'*62}")
    print(f"  XGBoost Classification Power  (your n = {n_total:,}, α = {ALPHA})")
    print(f"{'─'*62}")
    print(
        f"  {'Effect':>6}  {'Label':<8} {'Required N':>12} "
        f"{'Your N':>8} {'Power':>8}"
    )
    print(
        f"  {'─'*6}  {'─'*8} {'─'*12} "
        f"{'─'*8} {'─'*8}"
    )

    for d, label in EFFECT_SIZES:
        required_n = analysis.solve_power(
            effect_size=d,
            power=TARGET_PWR,
            alpha=ALPHA,
            alternative=TWO_SIDED,
        )
        actual_pwr = analysis.solve_power(
            effect_size=d,
            nobs1=n_total,
            alpha=ALPHA,
            alternative=TWO_SIDED,
        )
        # statsmodels overflows to nan when power ≈ 1; treat as >0.9999
        pwr_display = f"{actual_pwr:>8.4f}" if not np.isnan(actual_pwr) else "  >0.9999"
        print(
            f"  {d:>6.1f}  {label:<8} {int(np.ceil(required_n)):>12,} "
            f"{n_total:>8,} {pwr_display}"
        )

    # ------------------------------------------------------------------ DiD power
    n_did = 12   # ~12 quarterly observations per group in a typical DiD
    print(f"\n{'─'*62}")
    print(f"  DiD t-test Power  (n = {n_did} per group, α = {ALPHA})")
    print(f"  (e.g. quarterly means, pre vs post, treated vs control)")
    print(f"{'─'*62}")
    print(f"  {'Effect':>6}  {'Label':<8} {'Power with n=12':>16}")
    print(f"  {'─'*6}  {'─'*8} {'─'*16}")

    for d, label in EFFECT_SIZES:
        pwr_did = analysis.solve_power(
            effect_size=d,
            nobs1=n_did,
            alpha=ALPHA,
            alternative=TWO_SIDED,
        )
        flag = "  ← underpowered" if pwr_did < TARGET_PWR else ""
        print(f"  {d:>6.1f}  {label:<8} {pwr_did:>16.4f}{flag}")

    # ------------------------------------------------------------------ interpretation
    print(f"\n{'─'*62}")
    print(f"  INTERPRETATION")
    print(f"{'─'*62}")
    print(f"""
  XGBoost classifier (n = {n_total:,}):
  ─ With {n_total:,} firm-months, the model is well-powered (> 0.80) to
    detect medium or larger effects (d ≥ 0.5).
  ─ For small effects (d = 0.2) the dataset is still adequately
    powered — a common threshold is ~394 per group; at {n_total:,} total
    we far exceed that.
  ─ Avg Precision of {0.1228:.4f} vs baseline {base_rate:.4f} represents a
    {(0.1228/base_rate - 1)*100:.0f}% lift — a real but modest signal given
    the rarity of transitions and the 26-firm panel size.

  DiD t-test (n = {n_did} per group):
  ─ With only ~12 observations per cell (e.g. quarterly averages,
    4 pre-quarters and 8 post-quarters), the DiD is severely
    underpowered even for large effects (d = 0.8: power ≈ {
        analysis.solve_power(effect_size=0.8, nobs1=n_did,
                             alpha=ALPHA, alternative=TWO_SIDED):.4f}).
  ─ This is a fundamental constraint of studying a single treated
    firm over ~7 years. The DiD result should be treated as
    descriptive / exploratory, not confirmatory.
  ─ The Synthetic Control approach is preferred precisely because
    it constructs a credible counterfactual from the full monthly
    time series, bypassing the small-n DiD limitation.
""")


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    run_power_analysis()
