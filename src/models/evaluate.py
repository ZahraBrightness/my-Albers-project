"""
src/models/evaluate.py
-----------------------
SHAP-based model interpretation and feature-set comparison for the
executive-transition XGBoost model.

Four public functions
---------------------
compute_shap_values(model, X)
    TreeExplainer SHAP values as a shap.Explanation object.

plot_shap_summary(shap_values, X, save_path)
    Beeswarm summary plot — top 20 features.

plot_feature_importance(model, feature_names, save_path)
    Horizontal bar chart of XGBoost gain importance, colored by feature type.

compare_raw_vs_adjusted(model, dataset, save_path)
    Side-by-side Average Precision comparison: raw firm values vs
    industry-adjusted z-scores.  The core validation chart.

Run as a script:
    python src/models/evaluate.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — must be before pyplot

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR  = PROJECT_ROOT / "outputs" / "figures"

_FEATURE_SUFFIXES = ("_z", "_chg3", "_chg12", "_roll3")
_FEATURE_PREFIXES = ("score_",)
_EXTRA_FEATURES   = ("exec_score", "alpha", "mkt_beta", "ivol")

_XGB_CV_PARAMS = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="aucpr",
    random_state=42,
)


def _select_features(df: pd.DataFrame) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for c in df.columns:
        if (
            any(c.endswith(s) for s in _FEATURE_SUFFIXES)
            or any(c.startswith(p) for p in _FEATURE_PREFIXES)
            or c in _EXTRA_FEATURES
        ):
            if c not in seen:
                seen.add(c)
                result.append(c)
    return result


def _ensure_figures_dir(save_path: str | Path) -> Path:
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _feature_color(name: str) -> str:
    if name.endswith("_z"):
        return "#4472C4"   # blue — z-score
    if any(name.endswith(s) for s in ("_chg3", "_chg12", "_roll3")):
        return "#70AD47"   # green — lag / trend
    if name.startswith("score_") or name == "exec_score":
        return "#ED7D31"   # orange — composite score
    return "#A5A5A5"       # gray — other (alpha, mkt_beta, ivol)


# ---------------------------------------------------------------------------
# 1. compute_shap_values
# ---------------------------------------------------------------------------

def compute_shap_values(model: XGBClassifier, X: pd.DataFrame) -> shap.Explanation:
    """
    Compute SHAP values using TreeExplainer.

    Parameters
    ----------
    model : XGBClassifier
    X : pd.DataFrame  Feature matrix (nulls already filled).

    Returns
    -------
    shap.Explanation
    """
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer(X)
    return shap_vals


# ---------------------------------------------------------------------------
# 2. plot_shap_summary
# ---------------------------------------------------------------------------

def plot_shap_summary(
    shap_values: shap.Explanation,
    X: pd.DataFrame,
    save_path: str | Path = "outputs/figures/shap_summary.png",
) -> None:
    """
    Beeswarm SHAP summary plot — top 20 features by mean |SHAP|.

    Each dot is one prediction; color encodes the feature value (blue=low,
    red=high); the x-axis is SHAP value (impact on log-odds of transition).

    Parameters
    ----------
    shap_values : shap.Explanation
    X : pd.DataFrame
    save_path : str or Path
    """
    out = _ensure_figures_dir(
        PROJECT_ROOT / save_path if not Path(save_path).is_absolute() else save_path
    )

    # Mean absolute SHAP per feature
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    top20_idx = np.argsort(mean_abs)[::-1][:20]
    top5_names = [X.columns[i] for i in top20_idx[:5]]

    print("\n  Top 5 features by mean |SHAP|:")
    for rank, idx in enumerate(top20_idx[:5], start=1):
        print(f"    {rank}. {X.columns[idx]:<35}  mean|SHAP| = {mean_abs[idx]:.5f}")

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(
        shap_values[:, top20_idx],
        max_display=20,
        show=False,
        color_bar=True,
    )
    plt.title("SHAP Feature Impact — Executive Transition Model\n"
              "(top 20 features, each dot = one firm-month)", fontsize=12)
    plt.xlabel("SHAP value  (positive → higher P(transition))", fontsize=10)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# 3. plot_feature_importance
# ---------------------------------------------------------------------------

def plot_feature_importance(
    model: XGBClassifier,
    feature_names: List[str],
    save_path: str | Path = "outputs/figures/feature_importance.png",
) -> None:
    """
    Horizontal bar chart of XGBoost gain-based importance, top 20 features.

    Color coding
    ------------
    Blue   — z-score features  (ends in _z)
    Green  — lag / trend       (_chg3, _chg12, _roll3)
    Orange — composite scores  (score_* or exec_score)
    Gray   — other             (alpha, mkt_beta, ivol)
    """
    out = _ensure_figures_dir(
        PROJECT_ROOT / save_path if not Path(save_path).is_absolute() else save_path
    )

    # get_score() returns actual column names when trained on a DataFrame
    scores = model.get_booster().get_score(importance_type="gain")
    importance = dict(scores)   # keys are already feature names

    if not importance:
        print("  [WARN] No feature importance scores returned — skipping plot.")
        return

    imp_series = (
        pd.Series(importance)
        .sort_values(ascending=True)
        .tail(20)
    )

    colors = [_feature_color(name) for name in imp_series.index]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(imp_series.index, imp_series.values, color=colors, edgecolor="white")

    ax.set_xlabel("Gain (sum of improvement in split criterion)", fontsize=10)
    ax.set_title("XGBoost Feature Importance (Gain) — Top 20", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    legend_handles = [
        mpatches.Patch(color="#4472C4", label="Z-score features"),
        mpatches.Patch(color="#70AD47", label="Lag / trend features"),
        mpatches.Patch(color="#ED7D31", label="Composite score features"),
        mpatches.Patch(color="#A5A5A5", label="Other (alpha, beta, ivol)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9,
              framealpha=0.8)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# 4. compare_raw_vs_adjusted
# ---------------------------------------------------------------------------

def compare_raw_vs_adjusted(
    model: XGBClassifier,
    dataset: pd.DataFrame,
    save_path: str | Path = "outputs/figures/raw_vs_adjusted.png",
) -> None:
    """
    Compare Average Precision of raw firm values vs industry-adjusted z-scores.

    Model A — only `_firm` columns (raw, non-normalized values).
    Model B — only `_z` columns (industry-adjusted z-scores, the core feature
               set).  Same XGBoost hyperparameters, same TimeSeriesSplit folds.

    This chart is the central validation: if normalization adds value, Model B
    should beat Model A on Average Precision.
    """
    out = _ensure_figures_dir(
        PROJECT_ROOT / save_path if not Path(save_path).is_absolute() else save_path
    )

    target = "transition_within_3m"
    df = dataset[dataset[target].notna()].sort_values("date").reset_index(drop=True)
    y  = df[target].astype(int)

    firm_cols = [c for c in df.columns if c.endswith("_firm")]
    z_cols    = [c for c in df.columns if c.endswith("_z")]

    if not firm_cols:
        print("  [WARN] No _firm columns found — skipping raw vs adjusted comparison.")
        return
    if not z_cols:
        print("  [WARN] No _z columns found — skipping raw vs adjusted comparison.")
        return

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    spw   = n_neg / n_pos

    cv_params = {**_XGB_CV_PARAMS, "scale_pos_weight": spw}
    tscv = TimeSeriesSplit(n_splits=5)
    baseline = n_pos / (n_pos + n_neg)

    def _cv_avg_precision(X: pd.DataFrame) -> list[float]:
        aps = []
        for train_idx, test_idx in tscv.split(X):
            X_tr = X.iloc[train_idx].fillna(0)
            X_te = X.iloc[test_idx].fillna(0)
            y_tr = y.iloc[train_idx]
            y_te = y.iloc[test_idx]
            m = XGBClassifier(**cv_params)
            m.fit(X_tr, y_tr, verbose=False)
            prob = m.predict_proba(X_te)[:, 1]
            aps.append(average_precision_score(y_te, prob))
        return aps

    print("\n  Running CV for Model A (raw _firm columns) …")
    ap_raw  = _cv_avg_precision(df[firm_cols])
    print("  Running CV for Model B (industry-adjusted _z columns) …")
    ap_adj  = _cv_avg_precision(df[z_cols])

    mean_raw = float(np.mean(ap_raw))
    mean_adj = float(np.mean(ap_adj))

    winner    = "Industry-Adjusted" if mean_adj >= mean_raw else "Raw"
    delta     = abs(mean_adj - mean_raw)
    print(f"\n  Model A (Raw)               Avg Precision = {mean_raw:.4f}")
    print(f"  Model B (Industry-Adjusted) Avg Precision = {mean_adj:.4f}")
    print(f"  Winner: {winner}  (Δ = {delta:.4f})")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))

    labels = ["Model A\n(Raw firm values)", "Model B\n(Industry-adjusted z-scores)"]
    means  = [mean_raw, mean_adj]
    colors = ["#9DC3E6", "#4472C4"]

    bars = ax.bar(labels, means, color=colors, width=0.45,
                  edgecolor="white", linewidth=0.8)

    # Annotate bars
    for bar, val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.004,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    # Random baseline
    ax.axhline(baseline, color="#C00000", linestyle="--", linewidth=1.3,
               label=f"Random baseline ({baseline:.4f})")

    ax.set_ylabel("Average Precision (5-fold CV)", fontsize=10)
    ax.set_title(
        "Industry-Adjusted vs Raw Metrics: Predictive Power\n"
        "(higher = better; dashed = random classifier baseline)",
        fontsize=11,
    )
    ax.set_ylim(0, max(means) * 1.25)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from xgboost import XGBClassifier as _XGBClassifier

    model_path   = PROJECT_ROOT / "models" / "xgb_transition_model.json"
    dataset_path = PROJECT_ROOT / "data" / "processed" / "model_dataset.csv"

    print("Loading model and dataset …")
    model = _XGBClassifier()
    model.load_model(str(model_path))

    df = pd.read_csv(dataset_path, parse_dates=["date"])
    print(f"  Dataset: {df.shape[0]:,} rows × {df.shape[1]} cols")

    feature_cols = _select_features(df)
    print(f"  Feature columns: {len(feature_cols)}")

    X = df[feature_cols].fillna(0)

    print("\n" + "=" * 62)
    print("  1. SHAP values")
    print("=" * 62)
    shap_vals = compute_shap_values(model, X)
    plot_shap_summary(shap_vals, X)

    print("\n" + "=" * 62)
    print("  2. Feature importance")
    print("=" * 62)
    plot_feature_importance(model, feature_cols)

    print("\n" + "=" * 62)
    print("  3. Raw vs Industry-Adjusted comparison")
    print("=" * 62)
    compare_raw_vs_adjusted(model, df)

    print("\nEvaluation complete — figures saved to outputs/figures/")
