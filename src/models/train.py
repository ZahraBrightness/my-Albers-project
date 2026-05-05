"""
src/models/train.py
--------------------
Time-series cross-validated XGBoost training for executive-transition
prediction.

One public function
-------------------
train_model(dataset_path='data/processed/model_dataset.csv')
    Loads the model-ready dataset, runs 5-fold TimeSeriesSplit CV,
    logs to MLflow, trains a final model on all data, and saves it to
    models/xgb_transition_model.json.

Run as a script:
    python src/models/train.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

# Optional MLflow — gracefully degraded if unavailable
try:
    import mlflow
    import mlflow.xgboost
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR   = PROJECT_ROOT / "models"

# ---------------------------------------------------------------------------
# Feature column selector
# ---------------------------------------------------------------------------

_FEATURE_SUFFIXES = ("_z", "_chg3", "_chg12", "_roll3")
_FEATURE_PREFIXES = ("score_",)
_EXTRA_FEATURES   = ("exec_score", "alpha", "mkt_beta", "ivol")


def _select_features(df: pd.DataFrame) -> List[str]:
    """Return feature column names present in *df*."""
    cols: List[str] = []
    for c in df.columns:
        if any(c.endswith(s) for s in _FEATURE_SUFFIXES):
            cols.append(c)
        elif any(c.startswith(p) for p in _FEATURE_PREFIXES):
            cols.append(c)
        elif c in _EXTRA_FEATURES:
            cols.append(c)
    # Deduplicate while preserving order
    seen: set[str] = set()
    result: List[str] = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result


# ---------------------------------------------------------------------------
# train_model
# ---------------------------------------------------------------------------

def train_model(
    dataset_path: str = "data/processed/model_dataset.csv",
) -> tuple[XGBClassifier, Dict[str, Any]]:
    """
    Train an XGBoost transition-prediction model with TimeSeriesSplit CV.

    Parameters
    ----------
    dataset_path : str
        Path to the model-ready dataset (relative to cwd or absolute).

    Returns
    -------
    model : XGBClassifier
        Final model trained on the full dataset.
    scores : dict
        roc_auc_mean, roc_auc_std, avg_precision_mean, avg_precision_std,
        roc_auc_folds (list), avg_precision_folds (list).
    """
    if _MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    path = Path(dataset_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    print(f"Loading {path.name} …")
    df = pd.read_csv(path, parse_dates=["date"])
    print(f"  {df.shape[0]:,} rows × {df.shape[1]} cols loaded")

    # ------------------------------------------------------------------
    # 2. Drop rows where label is null
    # ------------------------------------------------------------------
    target = "transition_within_3m"
    df = df[df[target].notna()].reset_index(drop=True)
    print(f"  {df.shape[0]:,} rows after dropping null labels")

    # ------------------------------------------------------------------
    # 3. Feature selection
    # ------------------------------------------------------------------
    feature_cols = _select_features(df)
    print(f"  {len(feature_cols)} feature columns selected")

    X = df[feature_cols].copy()
    y = df[target].astype(int)

    # ------------------------------------------------------------------
    # 4 & 5. Sort by date (already done in engineer.py, but enforce here)
    # ------------------------------------------------------------------
    sort_idx = df["date"].argsort()
    X = X.iloc[sort_idx].reset_index(drop=True)
    y = y.iloc[sort_idx].reset_index(drop=True)

    # ------------------------------------------------------------------
    # 7. Class imbalance weight
    # ------------------------------------------------------------------
    n_neg = int((y == 0).sum())
    n_pos = int((y == 1).sum())
    scale_pos_weight = n_neg / n_pos
    print(f"  class balance: {n_neg:,} neg / {n_pos:,} pos  "
          f"(scale_pos_weight = {scale_pos_weight:.2f})")

    # ------------------------------------------------------------------
    # 8. XGBoost hyperparameters
    # ------------------------------------------------------------------
    params: Dict[str, Any] = {
        "n_estimators":      300,
        "max_depth":         4,
        "learning_rate":     0.05,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "eval_metric":       "aucpr",
        "scale_pos_weight":  scale_pos_weight,
        "random_state":      42,
    }

    # ------------------------------------------------------------------
    # 6 & 9. TimeSeriesSplit cross-validation
    # ------------------------------------------------------------------
    tscv = TimeSeriesSplit(n_splits=5)

    roc_auc_folds:       List[float] = []
    avg_precision_folds: List[float] = []

    print(f"\n{'='*62}")
    print(f"  TimeSeriesSplit CV  (5 folds, date-ordered)")
    print(f"{'='*62}")
    print(f"  {'Fold':>4}  {'ROC-AUC':>10}  {'Avg Precision':>14}  "
          f"{'Test size':>10}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*14}  {'-'*10}")

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_tr, X_te = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_tr, y_te = y.iloc[train_idx],         y.iloc[test_idx]

        # Fill nulls: zero = no signal available (not imputation)
        X_tr = X_tr.fillna(0)
        X_te = X_te.fillna(0)

        model = XGBClassifier(**params)
        model.fit(X_tr, y_tr, verbose=False)

        y_prob = model.predict_proba(X_te)[:, 1]

        roc     = roc_auc_score(y_te, y_prob)
        avg_pre = average_precision_score(y_te, y_prob)

        roc_auc_folds.append(roc)
        avg_precision_folds.append(avg_pre)

        print(f"  {fold:>4}  {roc:>10.4f}  {avg_pre:>14.4f}  "
              f"{len(test_idx):>10,}")

    roc_auc_mean       = float(np.mean(roc_auc_folds))
    roc_auc_std        = float(np.std(roc_auc_folds))
    avg_precision_mean = float(np.mean(avg_precision_folds))
    avg_precision_std  = float(np.std(avg_precision_folds))

    print(f"\n  Mean ROC-AUC       : {roc_auc_mean:.4f}  ± {roc_auc_std:.4f}")
    print(f"  Mean Avg Precision : {avg_precision_mean:.4f}  ± {avg_precision_std:.4f}")
    print(f"  Baseline Avg Prec  : {n_pos / (n_pos + n_neg):.4f}  (random classifier)")

    scores: Dict[str, Any] = {
        "roc_auc_mean":         roc_auc_mean,
        "roc_auc_std":          roc_auc_std,
        "avg_precision_mean":   avg_precision_mean,
        "avg_precision_std":    avg_precision_std,
        "roc_auc_folds":        roc_auc_folds,
        "avg_precision_folds":  avg_precision_folds,
    }

    # ------------------------------------------------------------------
    # 10. MLflow logging
    # ------------------------------------------------------------------
    if _MLFLOW_AVAILABLE:
        try:
            mlflow.set_experiment("executive_transitions")
            with mlflow.start_run(run_name="xgb_timeseries_cv"):
                # Hyperparameters
                for k, v in params.items():
                    mlflow.log_param(k, v)
                mlflow.log_param("n_features",   len(feature_cols))
                mlflow.log_param("n_splits",     5)
                mlflow.log_param("fill_strategy","zero")

                # Per-fold scores
                for i, (roc, ap) in enumerate(
                    zip(roc_auc_folds, avg_precision_folds), start=1
                ):
                    mlflow.log_metric(f"fold_{i}_roc_auc",       roc)
                    mlflow.log_metric(f"fold_{i}_avg_precision",  ap)

                # Aggregate metrics
                mlflow.log_metric("mean_roc_auc",       roc_auc_mean)
                mlflow.log_metric("std_roc_auc",        roc_auc_std)
                mlflow.log_metric("mean_avg_precision", avg_precision_mean)
                mlflow.log_metric("std_avg_precision",  avg_precision_std)

                # Train and log final model as artifact
                X_all = X.fillna(0)
                final_model = XGBClassifier(**params)
                final_model.fit(X_all, y, verbose=False)
                mlflow.xgboost.log_model(final_model, name="xgb_model")

            print("\n  MLflow run logged successfully.")
        except Exception as exc:
            print(f"\n  MLflow logging failed (non-fatal): {exc}")
            # Fall through to train final model outside MLflow block
            X_all = X.fillna(0)
            final_model = XGBClassifier(**params)
            final_model.fit(X_all, y, verbose=False)
    else:
        print("\n  MLflow not installed — skipping experiment tracking.")
        X_all = X.fillna(0)
        final_model = XGBClassifier(**params)
        final_model.fit(X_all, y, verbose=False)

    # ------------------------------------------------------------------
    # 11 & 12. Save final model
    # ------------------------------------------------------------------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "xgb_transition_model.json"
    final_model.save_model(str(model_path))
    print(f"  Model saved → {model_path}")

    return final_model, scores


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    model, scores = train_model()

    print(f"\n{'='*62}")
    print(f"  FINAL SCORES")
    print(f"{'='*62}")
    print(f"  Mean ROC-AUC       : {scores['roc_auc_mean']:.4f}  "
          f"± {scores['roc_auc_std']:.4f}")
    print(f"  Mean Avg Precision : {scores['avg_precision_mean']:.4f}  "
          f"± {scores['avg_precision_std']:.4f}")
    print(f"\nDone.")
