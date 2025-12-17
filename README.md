# MiTM Attack Detection on Electricity Market Bids

Detection framework for Man-in-the-Middle attacks on wholesale electricity market generator bids using machine learning.

## Overview

This repository implements a hybrid unsupervised-supervised anomaly detection framework that learns operational patterns from normal historical data to detect strategic bidding attacks on electricity markets. The framework addresses three deployment barriers: (1) detection without labeled attack data, (2) computational scalability, and (3) economic justification through ROI quantification.

## Attack Types

| Type | Description | Detection Signal |
|------|-------------|------------------|
| **Ramp Rate** | Misrepresent ramping capabilities | Inter-temporal dispatch anomalies |
| **Upper Limit** | Withhold generation capacity | Artificial scarcity patterns |
| **Lower Limit** | Inflate minimum generation | Must-run constraint violations |
| **Economic** | Manipulate cost coefficients | Merit order deviations |

## Repository Structure

```
├── Cases/                          # IEEE PES-PGLib test case definitions
├── OPF/                            # DC-OPF solver and attack injection
├── data/                           # Generated datasets (not included)
├── results/                        # Experiment outputs (not included)
│
├── 01_data_EDA.ipynb              # Data exploration (N14/N57/N300 variants)
├── 02_Realistic_Scenarios.ipynb   # ML pipeline with temporal validation
├── 03_Revenue_analysis_*.ipynb    # Economic impact quantification
│
└── advanced_features_pure*.py     # Feature engineering modules
```

## Quick Start

### 1. Generate Data
Use the `Cases/` and `OPF/` folders to generate DC-OPF solutions with attack injections:

```python
# Data structure: 34,032 features + metadata for N300 system
# Features: Bus powers × 24h + Voltage angles × 24h + Line flows × 24h
# Metadata: fval, ssn (season), type_of_attack, gen_attacked
```

### 2. Run Experiments
```bash
# Execute ML pipeline for IEEE 300-bus system
jupyter notebook 02_Realistic_Scenarios.ipynb
```

### 3. Economic Analysis
```bash
# Quantify detection value (marginal costs, 5-year projections)
jupyter notebook 03_Revenue_analysis_N300.ipynb
```

## Experimental Configurations

- **Systems**: IEEE 14-bus (5 gen), 57-bus (7 gen), 300-bus (69 gen)
- **Train/Test Splits**: 80/20, 60/40, 40/60
- **Scenarios**: Balanced (50% attacks), Imbalanced (25-37.5% attacks)
- **Label Noise**: Clean and 3% corrupted variants

## Models

**Base Learners**: Random Forest, Extra Trees, Gradient Boosting, XGBoost, LightGBM, CatBoost, Logistic Regression, MLP

**Ensembles**: Soft Voting, TPR-Weighted Voting, Attack-Optimized Stacking, High-Recall Ensemble

## Key Results

| System | Best F1 | Cost Ratio (TP/FN) |
|--------|---------|-------------------|
| 14-bus | 0.91 | 79× |
| 57-bus | 0.80 | 6.2× |
| 300-bus | 0.96 | 17× |

*Missed attacks cluster near normal operational costs; detected attacks show severe economic impact.*

## Requirements

```
numpy
pandas
scikit-learn
xgboost
lightgbm
catboost
matplotlib
seaborn
```

## Data Note

Raw data files and results folders are excluded due to size. Regenerate using:
1. Run OPF solvers on IEEE test cases in `Cases/`
2. Execute attack injection via `OPF/` scripts
3. Process with `01_data_EDA.ipynb`

## Citation

```bibtex
@article{mitm_market_detection,
  title={Market Operator Detection of Isolated Man-in-the-Middle Attacks 
         on Wholesale Market Generator Bids via Machine Learning},
  author={},
  journal={IEEE Transactions on Smart Grid},
  year={2025}
}
```

## License

MIT License
