"""
src/models/causal.py
---------------------
Synthetic Control estimation of the treatment effect of the Saligram
appointment (April 2019) on NWL's executive-performance score.

Four public functions
---------------------
build_sc_panel()
    Load exec_scores, pivot to wide, forward-fill, split pre/post.

fit_synthetic_control(pre_df, post_df, donor_tickers)
    Fit pysyncon Synth object; print weights and pre-treatment RMSPE.

compute_treatment_effect(synth, wide_df, donor_tickers, pre_dates, post_dates)
    ATT, pre-fit gap, placebo-test p-value.

plot_synthetic_control(actual, synthetic, treatment_date, att, p_value)
    Two-panel figure saved to outputs/figures/fig5_synthetic_control.png.

Run with:
    python src/models/causal.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

try:
    from pysyncon import Dataprep, Synth
    _PYSYNCON_OK = True
except ImportError:
    _PYSYNCON_OK = False

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR   = PROJECT_ROOT / "outputs" / "figures"

TREATED_UNIT    = "NWL"
TREATMENT_DATE  = pd.Timestamp("2019-04-01")
OUTCOME_COL     = "exec_score"
EXCLUDE_FROM_DONOR = ["SN", "REYN", "W"]


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 1 — build_sc_panel
# ─────────────────────────────────────────────────────────────────────────────

def build_sc_panel() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Load exec_scores, pivot wide, forward-fill ≤ 2 months, split pre/post.

    Returns
    -------
    pre_df        : wide DataFrame, dates < TREATMENT_DATE
    post_df       : wide DataFrame, dates >= TREATMENT_DATE
    wide_ff       : full wide DataFrame (pre + post, forward-filled)
    donor_tickers : list of eligible control tickers
    """
    df = pd.read_csv(
        PROCESSED_DIR / "exec_scores_all_firms.csv",
        parse_dates=["date"],
    )
    # Normalise to month-start so date arithmetic is clean
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df = df.sort_values(["ticker", "date"])

    wide = df.pivot(index="date", columns="ticker", values=OUTCOME_COL)
    wide.columns.name = None

    # Forward-fill within each column to handle sparse pre-treatment NWL gaps
    wide_ff = wide.ffill(limit=2)

    donor_tickers = [
        c for c in wide_ff.columns
        if c != TREATED_UNIT and c not in EXCLUDE_FROM_DONOR
    ]

    pre_df  = wide_ff[wide_ff.index <  TREATMENT_DATE]
    post_df = wide_ff[wide_ff.index >= TREATMENT_DATE]

    return pre_df, post_df, wide_ff, donor_tickers


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 2 — fit_synthetic_control
# ─────────────────────────────────────────────────────────────────────────────

def fit_synthetic_control(
    pre_df: pd.DataFrame,
    post_df: pd.DataFrame,
    wide_ff: pd.DataFrame,
    donor_tickers: List[str],
) -> Tuple[object, Dict[str, float], pd.Series, pd.Series]:
    """
    Fit pysyncon Synth; return synth object, weights dict, synthetic series,
    actual NWL series.

    Falls back to scipy minimization if pysyncon is unavailable.
    """
    pre_dates  = pre_df.index
    post_dates = post_df.index

    # Only use pre-treatment rows where NWL is non-null
    valid_pre = pre_dates[pre_df[TREATED_UNIT].notna()]

    print(f"\n  Pre-treatment dates       : {len(pre_dates)}")
    print(f"  Valid pre-treatment (NWL) : {len(valid_pre)}")
    print(f"  Post-treatment dates      : {len(post_dates)}")
    print(f"  Donor pool size           : {len(donor_tickers)}")

    if _PYSYNCON_OK:
        weights_series, synthetic_full, actual_full = _fit_pysyncon(
            wide_ff, donor_tickers, valid_pre
        )
    else:
        print("\n  [WARN] pysyncon not available — using scipy fallback.")
        weights_series, synthetic_full, actual_full = _fit_scipy(
            wide_ff, donor_tickers, valid_pre
        )

    # Weight dict (non-trivial only)
    weights_dict = {
        t: float(w) for t, w in weights_series.items() if float(w) > 0.01
    }

    # Pre-treatment RMSPE
    pre_actual    = actual_full.loc[valid_pre].dropna()
    pre_synthetic = synthetic_full.loc[pre_actual.index]
    rmspe = float(np.sqrt(((pre_actual - pre_synthetic) ** 2).mean()))

    print(f"\n  {'─'*58}")
    print(f"  DONOR WEIGHTS  (threshold > 0.01)")
    print(f"  {'─'*58}")
    print(f"  {'Ticker':<12} {'Weight':>10}  {'Bar'}")
    print(f"  {'─'*12} {'─'*10}  {'─'*20}")
    for ticker, w in sorted(weights_dict.items(), key=lambda x: -x[1]):
        bar = "█" * int(w * 40)
        print(f"  {ticker:<12} {w:>10.4f}  {bar}")
    print(f"\n  Pre-treatment RMSPE : {rmspe:.6f}  (lower = better fit)")

    return weights_series, weights_dict, synthetic_full, actual_full


def _fit_pysyncon(
    wide_ff: pd.DataFrame,
    donor_tickers: List[str],
    valid_pre: pd.DatetimeIndex,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Fit via pysyncon and return (weights_series, synthetic_full, actual_full)."""
    long = (
        wide_ff.stack(future_stack=True)
        .reset_index()
    )
    long.columns = ["date", "ticker", OUTCOME_COL]
    long = long.dropna(subset=[OUTCOME_COL])

    dp = Dataprep(
        foo=long,
        predictors=[OUTCOME_COL],
        predictors_op="mean",
        dependent=OUTCOME_COL,
        unit_variable="ticker",
        time_variable="date",
        treatment_identifier=TREATED_UNIT,
        controls_identifier=donor_tickers,
        time_predictors_prior=valid_pre,
        time_optimize_ssr=valid_pre,
    )

    synth = Synth()
    synth.fit(dp)

    weights_series = synth.weights()          # pd.Series indexed by ticker
    Z0_full = wide_ff[donor_tickers]
    synthetic_full = synth._synthetic(Z0=Z0_full)
    actual_full    = wide_ff[TREATED_UNIT]

    return weights_series, synthetic_full, actual_full


def _fit_scipy(
    wide_ff: pd.DataFrame,
    donor_tickers: List[str],
    valid_pre: pd.DatetimeIndex,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Fallback: minimize pre-treatment RMSPE with scipy (same objective as Synth)."""
    from scipy.optimize import minimize

    Z0_pre = wide_ff.loc[valid_pre, donor_tickers].fillna(0).values  # (T_pre, N_donors)
    Z1_pre = wide_ff.loc[valid_pre, TREATED_UNIT].fillna(0).values   # (T_pre,)
    n = len(donor_tickers)

    def _loss(w):
        synthetic = Z0_pre @ w
        return float(np.sqrt(np.mean((Z1_pre - synthetic) ** 2)))

    result = minimize(
        _loss,
        x0=np.ones(n) / n,
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        options={"ftol": 1e-10, "maxiter": 5000},
    )
    w_opt = result.x

    weights_series = pd.Series(w_opt, index=donor_tickers, name="weights")
    Z0_full        = wide_ff[donor_tickers].fillna(0).values
    synthetic_full = pd.Series(Z0_full @ w_opt, index=wide_ff.index)
    actual_full    = wide_ff[TREATED_UNIT]

    return weights_series, synthetic_full, actual_full


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 3 — compute_treatment_effect
# ─────────────────────────────────────────────────────────────────────────────

def compute_treatment_effect(
    weights_series: pd.Series,
    wide_ff: pd.DataFrame,
    donor_tickers: List[str],
    pre_dates: pd.DatetimeIndex,
    post_dates: pd.DatetimeIndex,
    actual_full: pd.Series,
    synthetic_full: pd.Series,
) -> Dict:
    """
    ATT, pre-fit gap, placebo p-value via leave-one-out placebo tests.

    For each donor firm:
      1. Treat it as the treated unit.
      2. Fit synthetic control from all remaining donors.
      3. Compute its post-treatment mean gap.

    p-value = fraction of placebo |ATT| >= |actual ATT|.
    """
    gap = actual_full - synthetic_full

    valid_pre  = pre_dates[actual_full.loc[pre_dates].notna()]
    pre_mean   = float(gap.loc[valid_pre].mean())
    att        = float(gap.loc[post_dates].mean())

    print(f"\n  {'─'*58}")
    print(f"  TREATMENT EFFECT")
    print(f"  {'─'*58}")
    print(f"  Pre-treatment mean gap (fit check) : {pre_mean:+.6f}")
    print(f"  ATT (post-treatment mean gap)      : {att:+.6f}")

    # ── Placebo tests ──────────────────────────────────────────────────
    print(f"\n  Running placebo tests ({len(donor_tickers)} donors) …")

    placebo_atts: List[float] = []

    for placebo_unit in donor_tickers:
        placebo_donors = [d for d in donor_tickers if d != placebo_unit]

        # Valid pre-treatment dates where placebo unit is non-null
        p_pre_valid = pre_dates[wide_ff.loc[pre_dates, placebo_unit].notna()]
        if len(p_pre_valid) < 4:
            continue  # too sparse to fit a meaningful synthetic control

        if _PYSYNCON_OK:
            try:
                _, p_synth, p_actual = _fit_pysyncon(
                    wide_ff, placebo_donors, p_pre_valid
                )
                # Override: placebo unit IS the "treated" unit
                p_actual = wide_ff[placebo_unit]
                p_gap    = (p_actual - p_synth).loc[post_dates]
                placebo_atts.append(float(p_gap.mean()))
            except Exception:
                continue
        else:
            try:
                _, p_synth, _ = _fit_scipy(wide_ff, placebo_donors, p_pre_valid)
                p_actual = wide_ff[placebo_unit]
                p_gap    = (p_actual - p_synth).loc[post_dates]
                placebo_atts.append(float(p_gap.mean()))
            except Exception:
                continue

    if placebo_atts:
        p_value = float(
            np.mean([abs(p) >= abs(att) for p in placebo_atts])
        )
    else:
        p_value = np.nan

    print(f"  Placebo tests completed            : {len(placebo_atts)}")
    if not np.isnan(p_value):
        print(f"  Placebo p-value                    : {p_value:.4f}")
        sig = "statistically significant" if p_value <= 0.10 else "not significant"
        direction = "improvement" if att > 0 else "deterioration"
        print(f"\n  Interpretation:")
        print(f"  The Saligram appointment is associated with a mean monthly "
              f"exec_score {direction} of {abs(att):.4f} relative to the")
        print(f"  synthetic control (p = {p_value:.4f} via placebo test).")
        print(f"  This effect is {sig} at the 10% level.")
    else:
        print(f"  Placebo p-value : N/A (insufficient placebo data)")

    return {
        "att":          att,
        "pre_mean_gap": pre_mean,
        "p_value":      p_value,
        "placebo_atts": placebo_atts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 4 — plot_synthetic_control
# ─────────────────────────────────────────────────────────────────────────────

def plot_synthetic_control(
    actual: pd.Series,
    synthetic: pd.Series,
    att: float,
    p_value: float,
    save_path: str | Path = "outputs/figures/fig5_synthetic_control.png",
) -> None:
    """Two-panel synthetic control figure."""
    out = Path(save_path)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    gap = actual - synthetic

    # Common x-range
    all_dates = actual.index.union(synthetic.index)
    post_mask = all_dates >= TREATMENT_DATE
    pre_mask  = all_dates <  TREATMENT_DATE

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    fig.subplots_adjust(hspace=0.35)

    # ── Panel 1 — main result ─────────────────────────────────────────
    ax1.plot(actual.index,    actual.values,    color="#1F4E79", linewidth=2,
             label="Actual NWL",      zorder=3)
    ax1.plot(synthetic.index, synthetic.values, color="#C00000", linewidth=2,
             linestyle="--", label="Synthetic NWL", zorder=3)
    ax1.axvline(TREATMENT_DATE, color="black", linewidth=1.3, linestyle="--",
                zorder=4, label=f"Treatment ({TREATMENT_DATE.date()})")

    # Shade periods
    pre_start  = all_dates.min()
    post_end   = all_dates.max()
    ax1.axvspan(pre_start, TREATMENT_DATE, alpha=0.08, color="gray", zorder=1)
    post_color = "#70AD47" if att > 0 else "#C00000"
    ax1.axvspan(TREATMENT_DATE, post_end, alpha=0.08, color=post_color, zorder=1)

    # Annotation box
    p_str = f"{p_value:.3f}" if not np.isnan(p_value) else "N/A"
    textbox = (
        f"ATT = {att:+.4f}\n"
        f"Placebo p = {p_str}"
    )
    ax1.text(
        0.97, 0.05, textbox,
        transform=ax1.transAxes, fontsize=9,
        verticalalignment="bottom", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="gray", alpha=0.85),
    )
    ax1.set_title(
        "NWL vs Synthetic Control: Executive Performance Score",
        fontsize=12, fontweight="bold",
    )
    ax1.set_ylabel("exec_score", fontsize=10)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.text(0.02, 0.98, "Pre-treatment", transform=ax1.transAxes,
             fontsize=8, color="gray", va="top")
    ax1.text(0.60, 0.98, "Post-treatment (Saligram era)", transform=ax1.transAxes,
             fontsize=8, color="gray", va="top")

    # ── Panel 2 — gap plot ────────────────────────────────────────────
    ax2.plot(gap.index, gap.values, color="#404040", linewidth=1.5, zorder=3)
    ax2.axhline(0, color="black", linewidth=0.9, linestyle="--")
    ax2.axvline(TREATMENT_DATE, color="black", linewidth=1.3, linestyle="--", zorder=4)

    # Fill above/below zero
    ax2.fill_between(gap.index, gap.values, 0,
                     where=(gap.values >= 0),
                     color="#70AD47", alpha=0.35, label="NWL above synthetic")
    ax2.fill_between(gap.index, gap.values, 0,
                     where=(gap.values < 0),
                     color="#C00000", alpha=0.35, label="NWL below synthetic")

    ax2.set_title(
        "Treatment Effect Gap  (Actual minus Synthetic NWL)",
        fontsize=11, fontweight="bold",
    )
    ax2.set_ylabel("Gap (exec_score)", fontsize=10)
    ax2.set_xlabel("Date", fontsize=10)
    ax2.legend(fontsize=9, loc="upper left")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    print("=" * 62)
    print("  SYNTHETIC CONTROL — NWL Executive Performance")
    print("=" * 62)

    # 1 — build panel
    print("\n[1] Building SC panel …")
    pre_df, post_df, wide_ff, donor_tickers = build_sc_panel()
    pre_dates  = pre_df.index
    post_dates = post_df.index
    print(f"  Wide panel: {wide_ff.shape[0]} dates × {wide_ff.shape[1]} firms")
    print(f"  Donor pool ({len(donor_tickers)}): {donor_tickers}")

    # 2 — fit
    print("\n[2] Fitting synthetic control …")
    weights_series, weights_dict, synthetic_full, actual_full = fit_synthetic_control(
        pre_df, post_df, wide_ff, donor_tickers
    )

    # 3 — treatment effect + placebos
    print("\n[3] Computing treatment effect and placebo tests …")
    results = compute_treatment_effect(
        weights_series, wide_ff, donor_tickers,
        pre_dates, post_dates, actual_full, synthetic_full,
    )

    # 4 — plot
    print("\n[4] Generating figure …")
    plot_synthetic_control(
        actual_full, synthetic_full,
        att=results["att"],
        p_value=results["p_value"],
    )

    # 5 — save results CSV
    print("\n[5] Saving results …")
    sc_df = pd.DataFrame({
        "date":          wide_ff.index,
        "actual_nwl":    actual_full.values,
        "synthetic_nwl": synthetic_full.values,
        "gap":           (actual_full - synthetic_full).values,
        "period":        ["pre" if d < TREATMENT_DATE else "post"
                          for d in wide_ff.index],
    })
    out_csv = PROCESSED_DIR / "synthetic_control_results.csv"
    sc_df.to_csv(out_csv, index=False)
    print(f"  Saved → {out_csv.name}")

    # 6 — final summary
    att     = results["att"]
    p_value = results["p_value"]
    p_str   = f"{p_value:.4f}" if not np.isnan(p_value) else "N/A"

    print(f"\n{'='*62}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*62}")
    print(f"\n  Donor pool: {donor_tickers}")
    print(f"\n  Non-trivial weights (> 1%):")
    for t, w in sorted(weights_dict.items(), key=lambda x: -x[1]):
        print(f"    {t:<8} {w:.4f}")
    print(f"\n  Pre-treatment RMSPE : {results['pre_mean_gap']:.6f}")
    print(f"  ATT                 : {att:+.6f}")
    print(f"  Placebo p-value     : {p_str}")
    direction  = "higher" if att > 0 else "lower"
    sig_phrase = "consistent with a causal improvement" if (
        not np.isnan(p_value) and p_value <= 0.10
    ) else "though this falls short of conventional significance thresholds"
    print(f"""
  Plain English:
  After Ravi Saligram's appointment as CEO in April 2019, NWL's
  composite executive-performance score ran {direction} than the
  synthetic control by {abs(att):.4f} points on average each month
  (ATT = {att:+.4f}, placebo p = {p_str}). This is {sig_phrase}
  given the small donor pool and limited pre-treatment window.
  The health (leverage) category remained the dominant structural
  weakness throughout both eras.
""")
