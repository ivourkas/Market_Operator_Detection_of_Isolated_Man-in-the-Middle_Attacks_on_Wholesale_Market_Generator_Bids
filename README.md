# MiTM Attack Detection on Electricity Market Bids

Detection framework for Man-in-the-Middle (MITM) attacks on wholesale
electricity market generator bids using DC optimal power flow simulations and
machine learning.

## Overview

This repository implements a hybrid unsupervised-supervised detection workflow
for bid-integrity attacks in wholesale electricity markets. The framework learns
normal generator operating patterns from synthetic DC-OPF market-clearing data,
builds OPF-derived operational features, and evaluates whether manipulated bid
parameters leave detectable signatures in dispatch, flows, and operating costs.

The project studies isolated single-generator, single-parameter attacks as the
main benchmark. Additional analysis scripts support paired normal-attack impact
assessment, feature-family ablations, and coordinated-attack stress tests.

## Attack Types

| Type | Description | Detection Signal |
|---|---|---|
| Ramp Rate | Misrepresent ramping capability | Inter-temporal dispatch anomalies |
| Upper Limit | Withhold available generation capacity | Artificial scarcity and capacity-binding patterns |
| Lower Limit | Inflate minimum generation requirements | Must-run behavior and fleet redispatch |
| Economic Bid | Manipulate bid cost coefficients | Merit-order and cost-pattern deviations |

## Repository Structure

```text
├── Cases/                         # IEEE PES-PGLib test case definitions
├── OPF/                           # MATLAB DC-OPF and attack-injection routines
├── data/                          # Generated synthetic OPF datasets
├── scripts/                       # Reproducible analysis and evaluation scripts
│
├── 01_data_EDA*.ipynb             # Data exploration and preprocessing notebooks
├── 02_Realistic_Scenarios*.ipynb  # Main ML experiment notebooks
├── 03_Revenue_analysis_*.ipynb    # Economic impact analysis notebooks
│
├── advanced_features_pure*.py     # Feature engineering modules
└── results/                       # Compact analysis outputs and figures
```

Large solver outputs, derived transposed datasets, model pickle files, and
legacy experiment folders are intentionally kept outside normal Git tracking.
They can be regenerated from the MATLAB routines, notebooks, and Python scripts.

## Quick Start

### 1. Generate OPF Data

Use the case files in `Cases/` with the MATLAB routines in `OPF/`.
MATPOWER must be available on the MATLAB path.

The generated datasets follow the same general structure:

```text
OPF variables over a 24-hour horizon
metadata: fval, season, attack type, attacked generator
```

### 2. Run Main ML Experiments

The notebook workflow is organized by system size:

```bash
jupyter notebook 02_Realistic_Scenarios_N14_Giannis.ipynb
jupyter notebook 02_Realistic_Scenarios_N57.ipynb
jupyter notebook 02_Realistic_Scenarios.ipynb
```

### 3. Run Economic Analysis

```bash
jupyter notebook 03_Revenue_analysis_N14.ipynb
jupyter notebook 03_Revenue_analysis_N57.ipynb
jupyter notebook 03_Revenue_analysis_N300.ipynb
```

### 4. Run Scripted Validation Analyses

The scripted workflow reproduces the paired-impact and model-diagnostic analyses
used to support the current manuscript results:

```bash
python3 scripts/validate_normal_attack_pairs.py
python3 scripts/compute_paired_impact_metrics.py
python3 scripts/classify_high_impact_attacks.py
python3 scripts/build_revision_tables.py
python3 scripts/plot_impact_quadrants.py
python3 scripts/run_severity_conditioned_ml.py --systems N14 N57 N300 --n-jobs 1
python3 scripts/run_feature_ablation_ml.py --systems N14 N57 N300 --n-iter 500 --n-jobs 1
```

For coordinated-attack stress testing on the IEEE 300-bus system:

```bash
python3 scripts/export_n300_heldout_normal_loads.py
```

Then generate the matched coordinated OPF samples in MATLAB and evaluate them:

```bash
python3 scripts/run_ood_stress_test_ml.py \
  --ood-files \
  "<one_generator_two_parameters_matched_file.csv>" \
  "<two_generators_same_parameter_matched_file.csv>" \
  --models CatBoost XGBoost "Random Forest" \
  --n-iter 500 \
  --n-jobs 1
```

## Experimental Configurations

- Systems: IEEE 14-bus, IEEE 57-bus, IEEE 300-bus
- Train/test settings: 80/20, 60/40, 40/60
- Class balance settings: balanced and imbalanced attack prevalence
- Robustness settings: clean labels and noisy-label variants
- Stress tests: coordinated multi-generator and multi-parameter attacks on
  matched held-out 300-bus load profiles

## Models

The experiments evaluate tree ensembles, gradient boosting, linear models,
neural networks, and voting/stacking ensembles, including:

- Random Forest
- Extra Trees
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost
- Logistic Regression
- Multi-layer Perceptron
- Soft voting, TPR-weighted voting, high-recall, and stacking ensembles

## Key Outputs

The analysis reports:

- detection accuracy, precision, recall, and F1-score across systems and data
  settings;
- attack-type recall for ramp-rate, capacity-limit, and economic-bid attacks;
- paired normal-attack cost, dispatch, and line-flow impact metrics;
- feature-family ablations to evaluate which operational signals drive model
  performance;
- coordinated-attack stress-test performance on the IEEE 300-bus system;
- economic comparisons between missed and detected attack sets.

## Requirements

Python:

```text
numpy
pandas
scikit-learn
xgboost
lightgbm
catboost
matplotlib
seaborn
```

MATLAB:

```text
MATPOWER
Optimization Toolbox
```

## Data Note

The data are synthetic OPF outputs generated from the included case files and
attack-injection routines. Large derived CSV files and trained model dumps are
not meant to be committed directly. Regenerate missing derived artifacts from
the corresponding OPF routine, notebook, or script.

## License

MIT License
