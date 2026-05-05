# Executive Transition Research Project
### Industry-Adjusted Financial Performance Analysis of Consumer Goods Firms (2017–2024)

This project analyzes whether industry-adjusted financial metrics can predict and explain executive transitions across 26 consumer goods firms from 2017 to 2024. Newell Brands (NWL) serves as the focal case study, with particular attention to the April 2019 CEO transition from Michael Polk to Ravi Saligram and its antecedents in firm-relative financial performance. Methods include XGBoost classification with time-series cross-validation, synthetic control causal inference, and SHAP feature importance analysis to identify which normalized metrics carry the strongest predictive signal. This work was built as part of a Research Assistant role at the Albers School of Business, Seattle University.

---

## Key Findings

- NWL improved **+0.673** in composite executive score from the Polk era (−0.288) to the Saligram era (+0.385), then deteriorated to −0.235 post-2022
- Synthetic control **ATT = +0.100** above counterfactual trajectory (dominant donors: KHC 32.4%, CPB 31.9%)
- **`leverage_z_chg3`** is the strongest financial predictor of executive transitions (top SHAP feature)
- XGBoost **Average Precision: 0.1228** — 36% above the 0.0899 random baseline
- Post-2022 deterioration is broad-based across all 4 performance categories (macro-driven), structurally different from the Polk-era collapse which was concentrated in Health and Profitability
- NWL outperforms DUR peers on interest coverage (mean z = +0.28) despite ROE underperformance — debt is serviceable but consumes equity return capacity

---

## Project Structure

```
my-Albers-project/
├── src/
│   ├── data/
│   │   ├── loader.py        # 11 datasets, RENAME map
│   │   ├── quality.py       # 5-check quality gate
│   │   └── cleaner.py       # Distortion handling, ffill
│   ├── features/
│   │   ├── normalize.py     # Industry z-scores
│   │   ├── score.py         # Composite exec scoring
│   │   └── engineer.py      # Lag features, labels
│   ├── models/
│   │   ├── train.py         # XGBoost + MLflow
│   │   ├── evaluate.py      # SHAP + raw vs adjusted
│   │   └── causal.py        # Synthetic control
│   └── analysis/
│       ├── power_analysis.py
│       └── sc_diagnostic.py
├── tests/
│   ├── test_loader.py       # 5 tests
│   ├── test_quality.py      # 5 tests
│   ├── test_cleaner.py      # 7 tests
│   └── test_normalize.py    # 7 tests
├── data/
│   ├── interim/             # Cleaned datasets
│   └── processed/           # Model-ready outputs
├── outputs/
│   └── figures/             # 9 charts
├── models/                  # XGBoost artifact
└── notebooks/               # EDA
```

---

## Methodology

### Industry Normalization

Each firm's metrics are compared against an industry benchmark rather than analyzed in isolation: the raw gap is computed as `firm − benchmark` before z-scoring. Ten metrics span four performance categories, capturing profitability, financial health, market valuation, and operational efficiency. Rolling 12-month z-scores (minimum 6 periods) are winsorized at ±3 to prevent outlier ratios from distorting the composite. This design separates firm-specific performance from sector-wide conditions — a CEO should not receive credit or blame for a macro tailwind shared by all peers.

### Composite Scoring

The composite executive score aggregates four categories: Profitability (25%), Health (30%), Market (20%), and Efficiency (25%). Health carries the highest weight because leverage is NWL's dominant structural constraint — a firm that cannot reduce its debt load faces binding limits on strategic flexibility regardless of operational improvement. Era analysis segments the full panel into three periods: `pre_polk_exit` (through March 2019), `saligram_era` (April 2019–December 2022), and `post_2022` (January 2023 onward), enabling before/after attribution at the executive level.

### XGBoost Classification

The binary target `transition_within_3m` flags quarters where a CEO or CFO departure is disclosed within three months, sourced from SEC 8-K Item 5.02 filings. `TimeSeriesSplit(n_splits=5)` enforces strict temporal ordering so no future data can leak into training folds. The 9% positive class rate is corrected via `scale_pos_weight`, and features include z-scores, 3- and 12-month changes, and rolling 3-month averages across all normalized metrics. A power analysis confirms the model is fully powered (≥0.80) at N=1,902 for all medium and large effect sizes.

### Synthetic Control

NWL is the single treated unit with a 21-firm donor pool drawn from the same consumer goods universe. The treatment date is April 2019, coinciding with Polk's departure. The optimizer fits pre-treatment donor weights to minimize RMSPE against the NWL trajectory, achieving near-perfect pre-treatment fit. The placebo p-value of 0.524 indicates directional correctness but limited inferential power — an expected result given only 16 pre-treatment months are available; conventional significance requires 30 or more, a constraint quantified directly by the accompanying power analysis.

---

## Setup and Replication

```bash
# Clone and setup
git clone https://github.com/ZahraBrightness/my-Albers-project.git
cd my-Albers-project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run full pipeline in order
python src/data/cleaner.py
python src/features/normalize.py
python src/features/score.py
python src/features/engineer.py
python src/models/train.py
python src/models/evaluate.py
python src/models/causal.py
python src/analysis/power_analysis.py

# Run tests
pytest tests/ -v

# View MLflow experiment tracking
mlflow server --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db
```

---

## Key Technical Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Causal method | Synthetic Control | Single treated unit — DiD requires multiple treated firms |
| CV strategy | TimeSeriesSplit | Prevents future data leaking into past predictions |
| Normalization | Rolling 12m z-score | Captures relative standing vs peers in same time period |
| Health weight | 30% (highest) | Leverage is NWL's dominant structural constraint |
| Donor exclusions | REYN, SN, W | Zero pre-treatment periods / negative correlation / too few months |
| Label source | SEC 8-K Item 5.02 | Legally mandated C-suite change disclosure within 4 business days |
| Winsorize at ±3 | Z-score clipping | Prevents ratio distortions from dominating composite score |

---

## Limitations

- **Label completeness:** SEC 8-K scraping misses some transitions; Execucomp would provide cleaner labels and likely improve AP score
- **Pre-treatment window:** 16 months limits synthetic control inference power (placebo p=0.524); 30+ months needed for conventional significance
- **Panel size:** 26 firms limits generalizability; findings are specific to the consumer goods sector
- **Single treated unit:** Synthetic control identifies direction of effect but cannot rule out firm-specific confounders coinciding with the transition
- **Public data ceiling:** Board decisions reflect private information unavailable in financial filings

---

## Results Summary

| Method | Key Result | Confidence |
|---|---|---|
| Era Scoring | +0.673 improvement Polk→Saligram | High — descriptive |
| XGBoost AP | 0.1228 vs 0.0899 baseline | High — fully powered |
| SHAP | leverage_z_chg3 top feature | High — model-based |
| Synthetic Control ATT | +0.100 above counterfactual | Medium — correct direction |
| Placebo p-value | 0.524 | Low power — 16 pre-periods |
| Power Analysis | DiD max 46.6% power at d=0.8 | Confirms SC over DiD |

---

## Tech Stack

- Python 3.12
- XGBoost — gradient boosted classification
- SHAP — feature importance and explainability
- pysyncon — synthetic control optimization
- MLflow — experiment tracking
- pandas / numpy — data pipeline
- matplotlib / seaborn — visualization
- statsmodels — power analysis
- pytest — unit testing (24 tests)
- SEC EDGAR — executive transition data source

---

## Author

Research Assistant — Albers School of Business  
Seattle University | Aug 2025 – Dec 2025
