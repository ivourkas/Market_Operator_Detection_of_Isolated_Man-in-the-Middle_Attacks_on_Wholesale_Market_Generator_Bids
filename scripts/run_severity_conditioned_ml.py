#!/usr/bin/env python3
"""
Run severity-conditioned ML evaluation for the revision workflow.

The script reproduces the approved notebook-style Balanced 80/20 setup, runs
the existing feature engineering and model suite, then connects row-level ML
predictions to pair-level physical/economic impact labels.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from audit_scenario_identity import build_scenario_with_existing_logic, SystemAuditConfig  # noqa: E402


warnings.filterwarnings("ignore", category=PerformanceWarning)
warnings.filterwarnings("ignore", message="'penalty' was deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")


@dataclass(frozen=True)
class FinalMLConfig:
    """System-specific settings that must stay aligned with the scenario notebooks."""

    system: str
    csv_path: Path
    scenario_family: str
    feature_module_name: str
    expected_train_attacks: int
    expected_train_normals: int
    expected_test_attacks: int
    expected_test_normals: int


@dataclass(frozen=True)
class ModelInputs:
    """Model-ready matrices and labels, before any feature-ranking step."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]


@dataclass(frozen=True)
class FeatureRankedInputs:
    """Model matrices after mutual-information feature ranking."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]
    feature_scores: pd.DataFrame


@dataclass(frozen=True)
class ModelSpec:
    """One classifier definition used in the final severity-conditioned run."""

    name: str
    grid_key: str
    estimator: object


@dataclass(frozen=True)
class ModelEvaluation:
    """Standard train/test metrics for one fitted classifier."""

    model_name: str
    estimator: object
    train_accuracy: float
    train_precision: float
    train_recall: float
    train_f1: float
    test_accuracy: float
    test_precision: float
    test_recall: float
    test_f1: float
    test_roc_auc: float | None
    accuracy_gap: float
    f1_gap: float
    test_predictions: pd.Series
    test_attack_score: pd.Series


@dataclass(frozen=True)
class TunedModelResult:
    """Search result and final test evaluation for one classifier."""

    model_name: str
    best_estimator: object
    best_params: dict[str, object]
    best_cv_score: float
    scoring_name: str
    scaling_method: str
    scaled_feature_count: int
    requested_iterations: int
    actual_iterations: int
    evaluation: ModelEvaluation


@dataclass(frozen=True)
class PreparedSystemData:
    """All data products needed before model tuning begins for one system."""

    config: FinalMLConfig
    train_raw: pd.DataFrame
    test_raw: pd.DataFrame
    train_features: pd.DataFrame
    test_features: pd.DataFrame
    model_inputs: ModelInputs
    ranked_inputs: FeatureRankedInputs


@dataclass(frozen=True)
class SystemPipelineRun:
    """Files and in-memory results produced by one system-level ML run."""

    prepared_data: PreparedSystemData
    tuned_results: list[TunedModelResult]
    model_summary: pd.DataFrame
    test_predictions: pd.DataFrame
    selected_features: pd.DataFrame
    predictions_with_impact: pd.DataFrame
    recall_by_impact: pd.DataFrame
    severity_recall_table: pd.DataFrame


@dataclass(frozen=True)
class FinalPipelineRun:
    """Consolidated results from running one or more power systems."""

    system_runs: list[SystemPipelineRun]
    model_summary: pd.DataFrame
    selected_features: pd.DataFrame
    test_predictions: pd.DataFrame
    predictions_with_impact: pd.DataFrame
    recall_by_impact: pd.DataFrame
    severity_recall_table: pd.DataFrame


MODEL_METADATA_COLUMNS = {
    "attack",
    "type_of_attack",
    "gen_attacked",
    "raw_row_index",
    "repetition",
    "session_in_rep",
    "temporal_index",
    "ssn",
    "year",
    "session_day",
    "day_in_season",
    "temporal_position",
    "original_ssn",
    "_ft_local",
}
TEST_PREDICTION_METADATA_COLUMNS = (
    "raw_row_index",
    "attack",
    "type_of_attack",
    "gen_attacked",
    "ssn",
    "original_ssn",
    "year",
    "repetition",
    "session_day",
    "day_in_season",
    "temporal_index",
)
TEXT_FEATURE_PATTERNS = ("peak_hour_bus",)
LINEAR_SCALE_PATTERNS = (
    "distance",
    "score",
    "zscore",
    "deviation",
    "_mean",
    "_std",
    "_max",
    "_min",
    "_range",
    "_capacity_factor",
    "_utilization",
    "ramp_up",
    "ramp_down",
    "ramp_std",
    "flow_deviation",
    "angle_deviation",
    "fval",
    "roll_mean",
    "lag",
    "diff",
    "total_generation",
    "gen_concentration",
    "gen_diversity",
    "knn_distance",
)
NO_SCALE_PATTERNS = (
    "violations",
    "starts",
    "stops",
    "cycles",
    "hours_",
    "consecutive_",
    "active_generators",
    "is_",
    "has_",
    "binary",
    "detected",
    "type",
    "rank",
    "order",
    "position",
    "temporal_index",
    "ssn",
    "repetition",
    "year",
    "day",
    "hour",
    "correlation",
    "inversions",
)
MI_SCORE_FLOOR = 0.0001
RANDOM_STATE = 42
SEVERITY_ML_OUTPUT_DIR = PROJECT_ROOT / "results" / "revision" / "severity_ml"
PAIR_CLASSIFICATION_PATH = PROJECT_ROOT / "results" / "revision" / "paired_impact_classification.csv"
FINAL_SYSTEMS = ("N14", "N57", "N300")
CONSOLIDATED_OUTPUT_FILES = {
    "model_summary": "model_summary.csv",
    "selected_features": "selected_features.csv",
    "test_predictions": "test_predictions.csv",
    "predictions_with_impact": "predictions_with_impact.csv",
    "recall_by_impact": "recall_by_impact_pattern.csv",
    "severity_recall_table": "severity_recall_table.csv",
}
IMPACT_PATTERN_ORDER = (
    "overall_attack",
    "weak_or_nonbinding_attack",
    "economic_impact_attack",
    "hidden_redispatch_attack",
    "critical_attack",
)
IMPACT_LABEL_COLUMNS = (
    "pair_id",
    "system",
    "season",
    "normal_raw_row_index",
    "attacked_raw_row_index",
    "attack_type",
    "attack_type_name",
    "target_generator",
    "delta_cost_abs",
    "delta_cost_pct",
    "dispatch_l1",
    "target_dispatch_change",
    "non_target_dispatch_change",
    "flow_l1",
    "ramp_l1",
    "cost_severity",
    "dispatch_severity",
    "impact_pattern",
    "zero_response",
)


def samples_per_cv_repetition(config: FinalMLConfig) -> int:
    """
    Return the notebook's repetition-size assumption for internal CV.

    N14 and N57 use 4 seasons times 100 days. N300 uses 4 seasons times 96
    days. This only controls how the training set is divided during
    hyperparameter search; it does not change the final train/test split.
    """

    return 384 if config.system == "N300" else 400


def large_grid_minimum_samples(config: FinalMLConfig) -> int:
    """
    Return the notebook threshold for the 80/20 hyperparameter grid.

    The N300 notebook uses 96-day seasons, so its large-grid threshold is lower
    than the N14/N57 threshold. Our current revision script supports only the
    Balanced 80/20 scenario, where all three systems meet this condition.
    """

    return 1200 if config.system == "N300" else 1300


def create_search_grids(
    config: FinalMLConfig,
    n_samples: int,
    attack_ratio: float,
) -> dict[str, dict[str, list[object]]]:
    """
    Build the notebook's conservative large-sample hyperparameter grids.

    These grids are intentionally regularized: shallow trees, minimum leaf
    sizes, row/feature subsampling, and explicit penalties reduce the chance
    that high test scores come from a brittle model.
    """

    if n_samples < large_grid_minimum_samples(config):
        raise ValueError(
            f"{config.system} has {n_samples} training rows. This revision script "
            "currently implements only the notebook's Balanced 80/20 large-sample grid."
        )

    is_imbalanced = attack_ratio < 0.4 or attack_ratio > 0.6
    scale_pos_weight = None
    if 0 < attack_ratio < 0.4:
        scale_pos_weight = (1 - attack_ratio) / attack_ratio

    return {
        "extra_trees": {
            "n_estimators": [100, 200, 300, 400],
            "max_depth": [3, 5, 7, 10, None],
            "min_samples_split": [20, 50, 100],
            "min_samples_leaf": [10, 20, 50],
            "max_features": ["sqrt", 0.1, 0.3, 0.5],
            "bootstrap": [True],
            "max_samples": [0.5, 0.7, 0.9],
            "class_weight": (
                ["balanced", {0: 1, 1: scale_pos_weight}]
                if scale_pos_weight
                else [None, "balanced"]
            ),
        },
        "random_forest": {
            "n_estimators": [50, 100, 200, 300, 400],
            "max_depth": [3, 5, 7, 10, None],
            "min_samples_split": [20, 50, 80, 100],
            "min_samples_leaf": [10, 20, 50],
            "max_features": ["sqrt", 0.1, 0.3, 0.5],
            "bootstrap": [True],
            "max_samples": [0.5, 0.7, 0.9],
            "class_weight": (
                ["balanced", {0: 1, 1: scale_pos_weight}]
                if scale_pos_weight
                else [None, "balanced"]
            ),
        },
        "gradient_boosting": {
            "n_estimators": [50, 100, 200, 300],
            "learning_rate": [0.001, 0.005, 0.01, 0.1],
            "max_depth": [2, 3, 4, 5],
            "min_samples_split": [50, 100, 200],
            "min_samples_leaf": [20, 50, 75, 100],
            "subsample": [0.3, 0.5, 0.6, 0.7, 0.8],
            "max_features": ["sqrt", 0.1, 0.2, 0.3, 0.5],
        },
        "xgboost": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.001, 0.01, 0.05, 0.1],
            "max_depth": [2, 3, 4, 5],
            "min_child_weight": [10, 30, 50, 100],
            "gamma": [1, 5, 10, 20],
            "subsample": [0.5, 0.6, 0.7],
            "colsample_bytree": [0.3, 0.5, 0.7],
            "scale_pos_weight": (
                [1, scale_pos_weight, scale_pos_weight * 1.5]
                if scale_pos_weight
                else [1]
            ),
            "reg_alpha": [5, 10, 20, 50],
            "reg_lambda": [10, 20, 50, 100],
        },
        "lightgbm": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.001, 0.01, 0.05, 0.1],
            "max_depth": [2, 3, 4, 5],
            "num_leaves": [10, 20, 31],
            "min_child_samples": [50, 100, 200],
            "reg_alpha": [5, 10, 20],
            "reg_lambda": [5, 10, 20],
            "subsample": [0.5, 0.6, 0.7],
            "colsample_bytree": [0.3, 0.5, 0.7],
            "scale_pos_weight": (
                [1, scale_pos_weight, scale_pos_weight * 1.5]
                if scale_pos_weight
                else [1]
            ),
            "is_unbalanced": [True, False],
            "min_gain_to_split": [1.0, 2.0, 5.0],
            "boosting_type": ["gbdt"],
            "objective": ["binary"],
            "metric": ["binary_logloss"],
            "verbosity": [-1],
            "force_col_wise": [True],
        },
        "catboost": {
            "iterations": [100, 200, 300],
            "learning_rate": [0.001, 0.01, 0.05, 0.1],
            "depth": [3, 4, 5, 6],
            "l2_leaf_reg": [10, 20, 50],
            "subsample": [0.5, 0.6, 0.7],
            "colsample_bylevel": [0.3, 0.5, 0.7],
            "scale_pos_weight": (
                [1, scale_pos_weight, scale_pos_weight * 1.5]
                if scale_pos_weight
                else [1]
            ),
            "random_strength": [2, 5, 10],
            "bagging_temperature": [0, 1, 2],
        },
        "logistic_regression": {
            "penalty": ["l2"],
            "C": [0.1],
            "max_iter": [1000],
            "solver": ["lbfgs"],
            "class_weight": ["balanced"] if is_imbalanced else [None],
        },
        "neural_network": {
            "hidden_layer_sizes": [(100,), (200,), (100, 50), (200, 100), (300, 150, 75)],
            "activation": ["relu"],
            "solver": ["adam"],
            "alpha": [0.1, 0.5, 1.0, 2.0],
            "learning_rate": ["adaptive"],
            "learning_rate_init": [0.0001, 0.0005, 0.001],
            "max_iter": [2000],
            "early_stopping": [True],
            "validation_fraction": [0.2],
            "n_iter_no_change": [50],
            "batch_size": ["auto", 32, 64],
        },
    }


def default_search_iterations(config: FinalMLConfig, n_samples: int) -> int:
    """
    Return the notebook's RandomizedSearchCV budget for this scenario size.

    In the current Balanced 80/20 revision run, all supported systems land in
    the large-sample branch, so the default budget is 500 sampled settings.
    """

    if config.system == "N300":
        if n_samples < 800:
            return 200
        if n_samples < 1200:
            return 300
        return 500

    if n_samples < 900:
        return 200
    if n_samples < 1300:
        return 300
    return 500


def count_grid_settings(search_grid: dict[str, list[object]]) -> int:
    """Count how many hyperparameter combinations are present in one grid."""

    total = 1
    for values in search_grid.values():
        total *= len(values)
    return total


def choose_search_scorer(attack_ratio: float):
    """
    Match the notebook scoring rule: F1 for balanced data, F2 for imbalanced.

    F2 weights recall more heavily. It is only activated when attacks are a
    minority class in the training split.
    """

    if attack_ratio < 0.4 or attack_ratio > 0.6:
        try:
            from sklearn.metrics import fbeta_score, make_scorer
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("scikit-learn is required for F2 scoring.") from exc
        return make_scorer(fbeta_score, beta=2, zero_division=0), "f2"

    return "f1", "f1"


def features_to_scale_for_linear_models(feature_names: list[str]) -> list[str]:
    """
    Apply the notebook's feature-name rule for LR/NN scaling.

    Continuous magnitudes are scaled; counts, flags, ranks, IDs, and bounded
    correlation-like quantities are left untouched.
    """

    scaled_features: list[str] = []
    for feature_name in feature_names:
        feature_lower = feature_name.lower()
        if any(pattern in feature_lower for pattern in NO_SCALE_PATTERNS):
            continue
        if any(pattern in feature_lower for pattern in LINEAR_SCALE_PATTERNS):
            scaled_features.append(feature_name)
        else:
            scaled_features.append(feature_name)
    return scaled_features


def prepare_estimator_for_search(
    model_spec: ModelSpec,
    search_grid: dict[str, list[object]],
    feature_names: list[str],
) -> tuple[object, dict[str, list[object]], str, int]:
    """
    Attach notebook-style scaling for models that need feature magnitudes aligned.

    Tree-based models see the engineered features directly. Logistic Regression
    uses standard scaling, and the Neural Network uses min-max scaling.
    """

    if model_spec.name not in {"Logistic Regression", "Neural Network"}:
        return model_spec.estimator, search_grid, "none", 0

    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("scikit-learn is required for model-specific scaling.") from exc

    scaled_features = features_to_scale_for_linear_models(feature_names)
    if not scaled_features:
        return model_spec.estimator, search_grid, "none", 0

    if model_spec.name == "Neural Network":
        scaler = MinMaxScaler()
        scaling_method = "minmax"
    else:
        scaler = StandardScaler()
        scaling_method = "standard"

    preprocessor = ColumnTransformer(
        transformers=[("scaled", scaler, scaled_features)],
        remainder="passthrough",
    )
    estimator = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model_spec.estimator),
        ]
    )
    prefixed_grid = {f"model__{key}": value for key, value in search_grid.items()}
    return estimator, prefixed_grid, scaling_method, len(scaled_features)


class SequentialTimeCV:
    """
    Cross-validation splitter that keeps validation later than training.

    The notebooks use this logic to tune models inside the training period
    without touching the held-out test year. The exact split boundaries depend
    on how many training repetitions are present.
    """

    def __init__(
        self,
        n_splits: int = 3,
        samples_per_repetition: int = 400,
        min_samples_per_split: int = 50,
        overlap_ratio: float = 0.2,
    ) -> None:
        self.n_splits = n_splits
        self.samples_per_repetition = samples_per_repetition
        self.min_samples_per_split = min_samples_per_split
        self.overlap_ratio = overlap_ratio

    def split(self, X: pd.DataFrame, y: pd.Series | None = None, groups=None):
        """Yield train/validation index arrays in temporal order."""

        n_samples = len(X)
        estimated_repetitions = max(1, round(n_samples / self.samples_per_repetition))

        if estimated_repetitions <= 1:
            split_generator = self._very_small_dataset_splits(n_samples)
        elif estimated_repetitions == 2:
            split_generator = self._overlapping_temporal_splits(n_samples)
        elif estimated_repetitions == 3:
            split_generator = self._medium_dataset_splits(n_samples)
        else:
            split_generator = self._large_dataset_splits(n_samples, estimated_repetitions)

        for split_number, (train_idx, validation_idx) in enumerate(split_generator, start=1):
            if split_number > self.n_splits:
                break
            self._check_split(train_idx, validation_idx, n_samples)
            if y is not None:
                self._check_label_coverage(y, train_idx, validation_idx)
            yield train_idx, validation_idx

    def _very_small_dataset_splits(self, n_samples: int):
        """Use simple expanding temporal splits when little training history exists."""

        for train_fraction in (0.6, 0.7, 0.8):
            split_point = int(n_samples * train_fraction)
            train_idx = np.arange(0, split_point)
            validation_idx = np.arange(split_point, n_samples)
            if self._has_enough_rows(train_idx, validation_idx):
                yield train_idx, validation_idx

    def _overlapping_temporal_splits(self, n_samples: int):
        """Use shifted temporal windows for two-repetition training sets."""

        window_size = int(n_samples * 0.6)
        overlap = int(window_size * self.overlap_ratio)
        start_positions = [0]
        if n_samples > window_size + overlap:
            start_positions.append(overlap)
        if n_samples > window_size + 2 * overlap:
            start_positions.append(2 * overlap)

        for start in start_positions:
            train_end = min(start + window_size, n_samples - self.min_samples_per_split)
            validation_idx = np.arange(train_end, n_samples)
            train_idx = np.arange(start, train_end)
            if self._has_enough_rows(train_idx, validation_idx):
                yield train_idx, validation_idx

    def _medium_dataset_splits(self, n_samples: int):
        """Use repetition-sized temporal blocks for three-repetition training sets."""

        repetition_size = n_samples // 3
        candidate_splits = (
            (0, repetition_size, repetition_size, 2 * repetition_size),
            (0, 2 * repetition_size, 2 * repetition_size, 3 * repetition_size),
            (repetition_size, 2 * repetition_size, 2 * repetition_size, 3 * repetition_size),
        )

        for train_start, train_end, validation_start, validation_end in candidate_splits:
            train_idx = np.arange(train_start, train_end)
            validation_idx = np.arange(validation_start, validation_end)
            if self._has_enough_rows(train_idx, validation_idx):
                yield train_idx, validation_idx

    def _large_dataset_splits(self, n_samples: int, n_repetitions: int):
        """Use the notebook's 80/20-style internal validation blocks."""

        repetition_size = n_samples // n_repetitions
        candidate_splits = (
            (0, 2 * repetition_size, 2 * repetition_size, 3 * repetition_size),
            (0, 3 * repetition_size, 3 * repetition_size, 4 * repetition_size),
            (repetition_size, 3 * repetition_size, 3 * repetition_size, 4 * repetition_size),
        )

        for train_start, train_end, validation_start, validation_end in candidate_splits:
            if validation_end > n_samples:
                continue
            train_idx = np.arange(train_start, train_end)
            validation_idx = np.arange(validation_start, validation_end)
            if self._has_enough_rows(train_idx, validation_idx):
                yield train_idx, validation_idx

    def _has_enough_rows(self, train_idx: np.ndarray, validation_idx: np.ndarray) -> bool:
        """Reject tiny train or validation folds that would make tuning unstable."""

        return (
            len(train_idx) >= self.min_samples_per_split
            and len(validation_idx) >= self.min_samples_per_split
        )

    @staticmethod
    def _check_split(
        train_idx: np.ndarray,
        validation_idx: np.ndarray,
        n_samples: int,
    ) -> None:
        """Fail fast if a split violates temporal order or row exclusivity."""

        if len(set(train_idx.tolist()) & set(validation_idx.tolist())) > 0:
            raise ValueError("Internal CV split has overlapping train/validation rows.")
        if train_idx.min() < 0 or validation_idx.max() >= n_samples:
            raise ValueError("Internal CV split produced an out-of-range row index.")
        if train_idx.max() >= validation_idx.min():
            raise ValueError("Internal CV split does not keep validation after training.")

    @staticmethod
    def _check_label_coverage(
        y: pd.Series | np.ndarray,
        train_idx: np.ndarray,
        validation_idx: np.ndarray,
    ) -> None:
        """Make sure each internal fold contains both normal and attack rows."""

        labels = np.asarray(y)
        if np.unique(labels).size <= 1:
            raise ValueError("Internal CV labels contain only one class.")
        if np.unique(labels[train_idx]).size <= 1:
            raise ValueError("Internal CV training fold contains only one class.")
        if np.unique(labels[validation_idx]).size <= 1:
            raise ValueError("Internal CV validation fold contains only one class.")

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


def create_model_suite() -> list[ModelSpec]:
    """
    Create the same family of classifiers used in the ML notebooks.

    Hyperparameter tuning will be added in a later step. Here we only define the
    estimators, so the model list is explicit before we attach the training loop.
    """

    try:
        from sklearn.ensemble import (
            ExtraTreesClassifier,
            GradientBoostingClassifier,
            RandomForestClassifier,
        )
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("scikit-learn is required for the ML model suite.") from exc

    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        raise RuntimeError(
            "XGBoost could not be loaded. On macOS this often means the OpenMP "
            "runtime is missing; install it with: brew install libomp"
        ) from exc

    try:
        from lightgbm import LGBMClassifier
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "LightGBM is required for the full ML model suite. Install it with: "
            "python3 -m pip install lightgbm"
        ) from exc

    try:
        from catboost import CatBoostClassifier
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "CatBoost is required for the full ML model suite. Install it with: "
            "python3 -m pip install catboost"
        ) from exc

    return [
        ModelSpec("Extra Trees", "extra_trees", ExtraTreesClassifier(random_state=RANDOM_STATE)),
        ModelSpec("Random Forest", "random_forest", RandomForestClassifier(random_state=RANDOM_STATE)),
        ModelSpec(
            "Gradient Boosting",
            "gradient_boosting",
            GradientBoostingClassifier(random_state=RANDOM_STATE),
        ),
        ModelSpec(
            "XGBoost",
            "xgboost",
            XGBClassifier(
                random_state=RANDOM_STATE,
                use_label_encoder=False,
                eval_metric="logloss",
                verbosity=0,
            ),
        ),
        ModelSpec(
            "LightGBM",
            "lightgbm",
            LGBMClassifier(
                random_state=RANDOM_STATE,
                verbosity=-1,
                force_col_wise=True,
            ),
        ),
        ModelSpec(
            "CatBoost",
            "catboost",
            CatBoostClassifier(
                random_state=RANDOM_STATE,
                verbose=False,
                allow_writing_files=False,
            ),
        ),
        ModelSpec(
            "Logistic Regression",
            "logistic_regression",
            LogisticRegression(random_state=RANDOM_STATE),
        ),
        ModelSpec("Neural Network", "neural_network", MLPClassifier(random_state=RANDOM_STATE)),
    ]


def evaluate_fitted_model(
    model_name: str,
    estimator: object,
    model_inputs: FeatureRankedInputs,
) -> ModelEvaluation:
    """
    Measure one already-fitted model on the train and test splits.

    Recall is the main detection metric: among true attacks, it tells us how
    many the model catches. Train-test gaps are kept because very high train
    scores with weaker test scores are a warning sign for overfitting.
    """

    try:
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("scikit-learn is required for model evaluation.") from exc

    train_predictions = pd.Series(
        estimator.predict(model_inputs.X_train),
        index=model_inputs.X_train.index,
        name="predicted_attack",
    ).astype(int)
    test_predictions = pd.Series(
        estimator.predict(model_inputs.X_test),
        index=model_inputs.X_test.index,
        name="predicted_attack",
    ).astype(int)

    if hasattr(estimator, "predict_proba"):
        test_attack_score = pd.Series(
            estimator.predict_proba(model_inputs.X_test)[:, 1],
            index=model_inputs.X_test.index,
            name="attack_score",
        )
    elif hasattr(estimator, "decision_function"):
        test_attack_score = pd.Series(
            estimator.decision_function(model_inputs.X_test),
            index=model_inputs.X_test.index,
            name="attack_score",
        )
    else:
        test_attack_score = test_predictions.astype(float).rename("attack_score")

    test_roc_auc = None
    if model_inputs.y_test.nunique() > 1:
        try:
            test_roc_auc = float(roc_auc_score(model_inputs.y_test, test_attack_score))
        except ValueError:
            test_roc_auc = None

    train_accuracy = float(accuracy_score(model_inputs.y_train, train_predictions))
    train_precision = float(
        precision_score(model_inputs.y_train, train_predictions, zero_division=0)
    )
    train_recall = float(recall_score(model_inputs.y_train, train_predictions, zero_division=0))
    train_f1 = float(f1_score(model_inputs.y_train, train_predictions, zero_division=0))

    test_accuracy = float(accuracy_score(model_inputs.y_test, test_predictions))
    test_precision = float(
        precision_score(model_inputs.y_test, test_predictions, zero_division=0)
    )
    test_recall = float(recall_score(model_inputs.y_test, test_predictions, zero_division=0))
    test_f1 = float(f1_score(model_inputs.y_test, test_predictions, zero_division=0))

    return ModelEvaluation(
        model_name=model_name,
        estimator=estimator,
        train_accuracy=train_accuracy,
        train_precision=train_precision,
        train_recall=train_recall,
        train_f1=train_f1,
        test_accuracy=test_accuracy,
        test_precision=test_precision,
        test_recall=test_recall,
        test_f1=test_f1,
        test_roc_auc=test_roc_auc,
        accuracy_gap=train_accuracy - test_accuracy,
        f1_gap=train_f1 - test_f1,
        test_predictions=test_predictions,
        test_attack_score=test_attack_score,
    )


def tune_one_model(
    config: FinalMLConfig,
    model_spec: ModelSpec,
    model_inputs: FeatureRankedInputs,
    n_iter: int | None = None,
    n_jobs: int = -1,
) -> TunedModelResult:
    """
    Tune one classifier with the notebook's temporal RandomizedSearchCV setup.

    The search sees only the training split. After the best setting is selected,
    the refitted estimator is evaluated once on the held-out test split.
    """

    try:
        from sklearn.model_selection import RandomizedSearchCV
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("scikit-learn is required for RandomizedSearchCV.") from exc

    n_samples = len(model_inputs.X_train)
    attack_ratio = float(model_inputs.y_train.mean())
    search_grids = create_search_grids(config, n_samples=n_samples, attack_ratio=attack_ratio)

    if model_spec.grid_key not in search_grids:
        raise ValueError(f"No search grid is defined for model '{model_spec.name}'.")

    search_grid = search_grids[model_spec.grid_key]
    estimator, search_grid, scaling_method, scaled_feature_count = prepare_estimator_for_search(
        model_spec=model_spec,
        search_grid=search_grid,
        feature_names=model_inputs.feature_names,
    )
    requested_iterations = n_iter or default_search_iterations(config, n_samples)
    actual_iterations = min(requested_iterations, count_grid_settings(search_grid))
    scorer, scoring_name = choose_search_scorer(attack_ratio)
    cv = SequentialTimeCV(
        n_splits=3,
        samples_per_repetition=samples_per_cv_repetition(config),
    )

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=search_grid,
        n_iter=actual_iterations,
        cv=cv,
        scoring=scorer,
        n_jobs=n_jobs,
        random_state=RANDOM_STATE,
        error_score="raise",
        refit=True,
    )
    search.fit(model_inputs.X_train, model_inputs.y_train)

    evaluation = evaluate_fitted_model(
        model_name=model_spec.name,
        estimator=search.best_estimator_,
        model_inputs=model_inputs,
    )

    return TunedModelResult(
        model_name=model_spec.name,
        best_estimator=search.best_estimator_,
        best_params=dict(search.best_params_),
        best_cv_score=float(search.best_score_),
        scoring_name=scoring_name,
        scaling_method=scaling_method,
        scaled_feature_count=scaled_feature_count,
        requested_iterations=requested_iterations,
        actual_iterations=actual_iterations,
        evaluation=evaluation,
    )


def load_final_ml_inputs(system: str) -> FinalMLConfig:
    """
    Return the dataset, split, and feature-module settings for one power system.

    This function does not load data or train a model. It only prevents fragile
    path/count choices from being repeated later in the script.
    """

    configs = {
        "N14": FinalMLConfig(
            system="N14",
            csv_path=PROJECT_ROOT / "data" / "N14_G5_transposed.csv",
            scenario_family="n14_n57_balanced_80_20",
            feature_module_name="advanced_features_pureN14",
            expected_train_attacks=800,
            expected_train_normals=800,
            expected_test_attacks=200,
            expected_test_normals=200,
        ),
        "N57": FinalMLConfig(
            system="N57",
            csv_path=PROJECT_ROOT / "data" / "N57_G7_transposed.csv",
            scenario_family="n14_n57_balanced_80_20",
            feature_module_name="advanced_features_pureN57",
            expected_train_attacks=800,
            expected_train_normals=800,
            expected_test_attacks=200,
            expected_test_normals=200,
        ),
        "N300": FinalMLConfig(
            system="N300",
            csv_path=PROJECT_ROOT / "data" / "N300_G69_transposed_new.csv",
            scenario_family="n300_balanced_80_20",
            feature_module_name="advanced_features_pure",
            expected_train_attacks=768,
            expected_train_normals=768,
            expected_test_attacks=192,
            expected_test_normals=192,
        ),
    }

    normalized_system = system.strip().upper()
    if normalized_system not in configs:
        supported = ", ".join(configs)
        raise ValueError(f"Unsupported system '{system}'. Choose one of: {supported}.")

    return configs[normalized_system]


def build_audited_train_test(config: FinalMLConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the notebook-style train/test split while preserving raw row identity.

    The returned frames are still raw OPF rows plus temporal metadata. No feature
    engineering or model fitting happens here.
    """

    audit_config = SystemAuditConfig(
        system=config.system,
        csv_path=config.csv_path,
        scenario_family=config.scenario_family,
        expected_train_attacks=config.expected_train_attacks,
        expected_train_normals=config.expected_train_normals,
        expected_test_attacks=config.expected_test_attacks,
        expected_test_normals=config.expected_test_normals,
    )
    train_raw, test_raw = build_scenario_with_existing_logic(audit_config)

    actual_counts = {
        "train_attacks": int(train_raw["attack"].sum()),
        "train_normals": int((train_raw["attack"] == 0).sum()),
        "test_attacks": int(test_raw["attack"].sum()),
        "test_normals": int((test_raw["attack"] == 0).sum()),
    }
    expected_counts = {
        "train_attacks": config.expected_train_attacks,
        "train_normals": config.expected_train_normals,
        "test_attacks": config.expected_test_attacks,
        "test_normals": config.expected_test_normals,
    }
    if actual_counts != expected_counts:
        raise ValueError(
            f"{config.system} split counts changed. "
            f"Expected {expected_counts}, got {actual_counts}."
        )

    if train_raw["raw_row_index"].isna().any() or test_raw["raw_row_index"].isna().any():
        raise ValueError(f"{config.system} split has missing raw_row_index values.")

    train_ids = set(train_raw["raw_row_index"].tolist())
    test_ids = set(test_raw["raw_row_index"].tolist())
    if train_ids & test_ids:
        raise ValueError(f"{config.system} split reuses raw rows across train and test.")

    return train_raw, test_raw


def import_feature_module(config: FinalMLConfig):
    """
    Import the existing system-specific feature engineering module.

    The module is returned but not executed here. This keeps the next feature
    engineering step explicit and easy to audit.
    """

    try:
        return importlib.import_module(config.feature_module_name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Could not import feature module '{config.feature_module_name}' "
            f"for {config.system}."
        ) from exc


def engineer_train_test_features(
    config: FinalMLConfig,
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply the existing feature engineering code to the audited train/test split.

    The feature module learns normal-behavior baselines from training data only,
    then applies those stored baselines to the test data. Row identity must remain
    unchanged so predictions can later be merged to impact labels.
    """

    feature_module = import_feature_module(config)
    feature_store: dict[str, object] = {}

    train_engineered, feature_store = feature_module.engineer_advanced_features(
        train_raw.copy(),
        is_training=True,
        feature_store=feature_store,
    )
    test_engineered, _ = feature_module.engineer_advanced_features(
        test_raw.copy(),
        is_training=False,
        feature_store=feature_store,
    )

    for split_name, raw_df, engineered_df in (
        ("train", train_raw, train_engineered),
        ("test", test_raw, test_engineered),
    ):
        if len(raw_df) != len(engineered_df):
            raise ValueError(
                f"{config.system} {split_name} feature engineering changed row count "
                f"from {len(raw_df)} to {len(engineered_df)}."
            )
        if "raw_row_index" not in engineered_df.columns:
            raise ValueError(f"{config.system} {split_name} features lost raw_row_index.")
        raw_ids = pd.to_numeric(raw_df["raw_row_index"], errors="raise")
        engineered_ids = pd.to_numeric(engineered_df["raw_row_index"], errors="raise")
        if raw_ids.nunique() != len(raw_ids) or engineered_ids.nunique() != len(engineered_ids):
            raise ValueError(f"{config.system} {split_name} features contain duplicated raw rows.")
        if set(raw_ids.tolist()) != set(engineered_ids.tolist()):
            raise ValueError(f"{config.system} {split_name} features lost or added raw rows.")

    return train_engineered, test_engineered


def is_raw_opf_column(column_name: str) -> bool:
    """Return True for original OPF exports such as ft1, ft2, ..., ftN."""

    return column_name.startswith("ft") and column_name[2:].isdigit()


def can_be_numeric(series: pd.Series) -> bool:
    """
    Mirror the notebook's practical numeric check.

    Numeric dtypes are accepted directly. Object-like columns are accepted only
    when more than half of the training values can be converted to numbers.
    """

    if pd.api.types.is_numeric_dtype(series):
        return True

    converted = pd.to_numeric(series, errors="coerce")
    return converted.notna().sum() > len(series) * 0.5


def build_model_inputs(
    train_engineered: pd.DataFrame,
    test_engineered: pd.DataFrame,
) -> ModelInputs:
    """
    Build clean model matrices from engineered train/test data.

    This step removes labels, temporal identifiers, row-tracking metadata, raw
    OPF columns, and text-like helper columns. It does not perform mutual
    information ranking or train any model.
    """

    if "attack" not in train_engineered.columns or "attack" not in test_engineered.columns:
        raise ValueError("Both train and test data must contain the attack label.")

    candidate_features: list[str] = []
    for column in train_engineered.columns:
        if column in MODEL_METADATA_COLUMNS:
            continue
        if is_raw_opf_column(column):
            continue
        if any(pattern in column for pattern in TEXT_FEATURE_PATTERNS):
            continue
        if column not in test_engineered.columns:
            raise ValueError(f"Feature '{column}' is missing from the engineered test data.")
        if can_be_numeric(train_engineered[column]):
            candidate_features.append(column)

    if not candidate_features:
        raise ValueError("No numeric model features were available after filtering.")

    X_train = (
        train_engineered[candidate_features]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )
    X_test = (
        test_engineered[candidate_features]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )
    y_train = train_engineered["attack"].astype(int)
    y_test = test_engineered["attack"].astype(int)

    return ModelInputs(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=candidate_features,
    )


def rank_features_by_information(model_inputs: ModelInputs) -> FeatureRankedInputs:
    """
    Rank candidate features with the notebook's mutual-information rule.

    The ranking is learned from the training split only. Test data is reduced to
    the same selected columns afterward, which avoids using test labels or test
    distribution information during feature selection.
    """

    try:
        from sklearn.feature_selection import mutual_info_classif
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "scikit-learn is required for mutual-information feature ranking."
        ) from exc

    if len(model_inputs.X_train) != len(model_inputs.y_train):
        raise ValueError("X_train and y_train have different row counts.")
    if len(model_inputs.X_test) != len(model_inputs.y_test):
        raise ValueError("X_test and y_test have different row counts.")

    mi_scores = mutual_info_classif(
        model_inputs.X_train,
        model_inputs.y_train,
        random_state=RANDOM_STATE,
    )
    feature_scores = (
        pd.DataFrame(
            {
                "feature": model_inputs.feature_names,
                "mi_score": mi_scores,
            }
        )
        .sort_values("mi_score", ascending=False)
        .reset_index(drop=True)
    )

    n_samples = len(model_inputs.X_train)
    n_candidates = len(model_inputs.feature_names)
    if n_samples > 1200:
        feature_limit = min(200, n_candidates)
    elif n_samples > 900:
        feature_limit = min(100, n_candidates)
    else:
        feature_limit = min(50, n_candidates)

    selected_scores = feature_scores[feature_scores["mi_score"] > MI_SCORE_FLOOR].head(feature_limit)
    selected_features = selected_scores["feature"].tolist()
    if not selected_features:
        raise ValueError(
            f"No features passed the MI score floor of {MI_SCORE_FLOOR}."
        )

    X_train = (
        model_inputs.X_train[selected_features]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )
    X_test = (
        model_inputs.X_test[selected_features]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )

    return FeatureRankedInputs(
        X_train=X_train,
        X_test=X_test,
        y_train=model_inputs.y_train,
        y_test=model_inputs.y_test,
        feature_names=selected_features,
        feature_scores=feature_scores,
    )


def select_model_specs(model_names: list[str] | None = None) -> list[ModelSpec]:
    """
    Choose the model specifications to run.

    Names are matched case-insensitively against the notebook model names. If no
    names are provided, the full eight-model suite is returned.
    """

    model_suite = create_model_suite()
    if model_names is None:
        return model_suite

    available_by_name = {model.name.lower(): model for model in model_suite}
    selected: list[ModelSpec] = []
    unknown_names: list[str] = []

    for model_name in model_names:
        normalized_name = model_name.strip().lower()
        if normalized_name in available_by_name:
            selected.append(available_by_name[normalized_name])
        else:
            unknown_names.append(model_name)

    if unknown_names:
        available = ", ".join(model.name for model in model_suite)
        unknown = ", ".join(unknown_names)
        raise ValueError(f"Unknown model name(s): {unknown}. Available models: {available}.")

    if not selected:
        raise ValueError("At least one model must be selected.")

    return selected


def clean_best_params(best_params: dict[str, object]) -> dict[str, object]:
    """Remove Pipeline prefixes from tuned parameters before writing CSV output."""

    cleaned: dict[str, object] = {}
    for key, value in best_params.items():
        readable_key = key.removeprefix("model__")
        cleaned[readable_key] = value
    return cleaned


def summarize_tuned_results(
    config: FinalMLConfig,
    ranked_inputs: FeatureRankedInputs,
    tuned_results: list[TunedModelResult],
) -> pd.DataFrame:
    """
    Convert tuned model results into a compact model-performance table.

    This table is the ordinary ML summary. Severity-conditioned recall is added
    later by merging row-level predictions with the paired impact labels.
    """

    rows: list[dict[str, object]] = []
    for result in tuned_results:
        evaluation = result.evaluation
        rows.append(
            {
                "system": config.system,
                "scenario_family": config.scenario_family,
                "model": result.model_name,
                "train_rows": len(ranked_inputs.X_train),
                "test_rows": len(ranked_inputs.X_test),
                "selected_features": len(ranked_inputs.feature_names),
                "scoring": result.scoring_name,
                "scaling": result.scaling_method,
                "scaled_features": result.scaled_feature_count,
                "requested_iterations": result.requested_iterations,
                "actual_iterations": result.actual_iterations,
                "best_cv_score": result.best_cv_score,
                "test_accuracy": evaluation.test_accuracy,
                "test_precision": evaluation.test_precision,
                "test_recall": evaluation.test_recall,
                "test_f1": evaluation.test_f1,
                "test_roc_auc": evaluation.test_roc_auc,
                "train_accuracy": evaluation.train_accuracy,
                "train_precision": evaluation.train_precision,
                "train_recall": evaluation.train_recall,
                "train_f1": evaluation.train_f1,
                "accuracy_gap": evaluation.accuracy_gap,
                "f1_gap": evaluation.f1_gap,
                "best_params": json.dumps(
                    clean_best_params(result.best_params),
                    sort_keys=True,
                    default=str,
                ),
            }
        )

    return pd.DataFrame(rows)


def tune_selected_models(
    config: FinalMLConfig,
    ranked_inputs: FeatureRankedInputs,
    model_names: list[str] | None = None,
    n_iter: int | None = None,
    n_jobs: int = -1,
) -> tuple[list[TunedModelResult], pd.DataFrame]:
    """
    Tune selected models and return the system-level model summary.

    This is intentionally model-agnostic: it delegates each individual search to
    tune_one_model, then collects the results in a paper-audit-friendly format.
    """

    selected_models = select_model_specs(model_names)
    tuned_results: list[TunedModelResult] = []
    for model_spec in selected_models:
        tuned_results.append(
            tune_one_model(
                config=config,
                model_spec=model_spec,
                model_inputs=ranked_inputs,
                n_iter=n_iter,
                n_jobs=n_jobs,
            )
        )

    summary = summarize_tuned_results(config, ranked_inputs, tuned_results)
    return tuned_results, summary


def build_test_prediction_table(
    config: FinalMLConfig,
    test_metadata: pd.DataFrame,
    tuned_results: list[TunedModelResult],
) -> pd.DataFrame:
    """
    Build row-level test predictions with the identifiers needed downstream.

    The key column is raw_row_index. It lets us connect each detected or missed
    test attack back to the paired OPF impact classification.
    """

    if not tuned_results:
        raise ValueError("No tuned model results were provided.")
    if "raw_row_index" not in test_metadata.columns:
        raise ValueError("Test metadata must contain raw_row_index.")
    if "attack" not in test_metadata.columns:
        raise ValueError("Test metadata must contain the attack label.")

    raw_ids = pd.to_numeric(test_metadata["raw_row_index"], errors="raise")
    if raw_ids.nunique() != len(raw_ids):
        raise ValueError(f"{config.system} test metadata contains duplicated raw rows.")

    metadata_columns = [
        column for column in TEST_PREDICTION_METADATA_COLUMNS if column in test_metadata.columns
    ]
    base = test_metadata[metadata_columns].copy()
    base.insert(0, "test_row_position", np.arange(len(base)))
    base["raw_row_index"] = raw_ids.astype(int)
    base["true_attack"] = pd.to_numeric(base["attack"], errors="raise").astype(int)
    base = base.drop(columns=["attack"])

    if "original_ssn" in base.columns:
        base["season"] = pd.to_numeric(base["original_ssn"], errors="coerce").astype("Int64")
    elif "ssn" in base.columns:
        base["season"] = pd.to_numeric(base["ssn"], errors="coerce").astype("Int64")

    prediction_frames: list[pd.DataFrame] = []
    for result in tuned_results:
        evaluation = result.evaluation
        predictions = evaluation.test_predictions.reindex(test_metadata.index)
        attack_scores = evaluation.test_attack_score.reindex(test_metadata.index)

        if predictions.isna().any() or attack_scores.isna().any():
            raise ValueError(
                f"{config.system} {result.model_name} predictions do not align with test rows."
            )

        model_frame = base.copy()
        model_frame.insert(0, "model", result.model_name)
        model_frame.insert(0, "scenario_family", config.scenario_family)
        model_frame.insert(0, "system", config.system)
        model_frame["predicted_attack"] = predictions.astype(int).to_numpy()
        model_frame["attack_score"] = pd.to_numeric(attack_scores, errors="raise").to_numpy()
        model_frame["correct_prediction"] = (
            model_frame["predicted_attack"] == model_frame["true_attack"]
        ).astype(int)
        model_frame["false_positive"] = (
            (model_frame["true_attack"] == 0) & (model_frame["predicted_attack"] == 1)
        ).astype(int)
        model_frame["false_negative"] = (
            (model_frame["true_attack"] == 1) & (model_frame["predicted_attack"] == 0)
        ).astype(int)
        prediction_frames.append(model_frame)

    return pd.concat(prediction_frames, ignore_index=True)


def load_impact_labels(path: Path = PAIR_CLASSIFICATION_PATH) -> pd.DataFrame:
    """
    Load the pair-level impact labels used for severity-conditioned recall.

    Each row describes one normal/attack OPF pair. The attacked raw row index is
    the bridge back to the ML test prediction for that exact attacked case.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Impact classification file not found: {path}. "
            "Run scripts/classify_high_impact_attacks.py first."
        )

    impact_labels = pd.read_csv(path)
    missing_columns = [
        column for column in IMPACT_LABEL_COLUMNS if column not in impact_labels.columns
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Impact classification file is missing column(s): {missing}.")

    impact_labels = impact_labels[list(IMPACT_LABEL_COLUMNS)].copy()
    impact_labels["season"] = pd.to_numeric(impact_labels["season"], errors="raise").astype(int)
    impact_labels["attacked_raw_row_index"] = pd.to_numeric(
        impact_labels["attacked_raw_row_index"], errors="raise"
    ).astype(int)

    duplicate_keys = impact_labels.duplicated(
        ["system", "season", "attacked_raw_row_index"]
    )
    if duplicate_keys.any():
        duplicate_count = int(duplicate_keys.sum())
        raise ValueError(
            f"Impact labels contain {duplicate_count} duplicated attacked-row keys."
        )

    return impact_labels


def attack_predictions_only(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only true attack rows from the row-level prediction table.

    Normal rows are useful for accuracy and false-positive analysis, but they do
    not have pair-level impact labels. Severity-conditioned recall is defined
    only over attacks.
    """

    required_columns = {
        "system",
        "scenario_family",
        "model",
        "raw_row_index",
        "true_attack",
        "predicted_attack",
    }
    missing_columns = required_columns - set(predictions.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Prediction table is missing column(s): {missing}.")

    attack_rows = predictions[predictions["true_attack"].astype(int) == 1].copy()
    if attack_rows.empty:
        raise ValueError("Prediction table contains no true attack rows.")

    if "season" not in attack_rows.columns:
        if "original_ssn" in attack_rows.columns:
            attack_rows["season"] = pd.to_numeric(
                attack_rows["original_ssn"], errors="raise"
            ).astype(int)
        elif "ssn" in attack_rows.columns:
            attack_rows["season"] = pd.to_numeric(
                attack_rows["ssn"], errors="raise"
            ).astype(int)
        else:
            raise ValueError("Prediction table needs season, original_ssn, or ssn.")
    else:
        attack_rows["season"] = pd.to_numeric(
            attack_rows["season"], errors="raise"
        ).astype(int)

    attack_rows["raw_row_index"] = pd.to_numeric(
        attack_rows["raw_row_index"], errors="raise"
    ).astype(int)
    attack_rows["predicted_attack"] = pd.to_numeric(
        attack_rows["predicted_attack"], errors="raise"
    ).astype(int)

    return attack_rows


def check_impact_merge_consistency(merged_predictions: pd.DataFrame) -> None:
    """
    Confirm that merged impact labels describe the same attack as the ML row.

    The raw row index is the primary key. These extra checks catch accidental
    merges where the row index matches but the attack metadata does not.
    """

    if merged_predictions["pair_id"].isna().any():
        missing_count = int(merged_predictions["pair_id"].isna().sum())
        raise ValueError(f"{missing_count} attacked prediction rows lack impact labels.")

    if {"type_of_attack", "attack_type"}.issubset(merged_predictions.columns):
        predicted_type = pd.to_numeric(
            merged_predictions["type_of_attack"], errors="coerce"
        )
        impact_type = pd.to_numeric(merged_predictions["attack_type"], errors="coerce")
        type_mismatch = predicted_type.ne(impact_type)
        if type_mismatch.any():
            mismatch_count = int(type_mismatch.sum())
            raise ValueError(f"{mismatch_count} merged rows have inconsistent attack types.")

    if {"gen_attacked", "target_generator"}.issubset(merged_predictions.columns):
        predicted_generator = pd.to_numeric(
            merged_predictions["gen_attacked"], errors="coerce"
        )
        impact_generator = pd.to_numeric(
            merged_predictions["target_generator"], errors="coerce"
        )
        comparable = predicted_generator.notna() & impact_generator.notna()
        generator_mismatch = comparable & predicted_generator.ne(impact_generator)
        if generator_mismatch.any():
            mismatch_count = int(generator_mismatch.sum())
            raise ValueError(
                f"{mismatch_count} merged rows have inconsistent target generators."
            )


def attach_impact_labels_to_predictions(
    config: FinalMLConfig,
    predictions: pd.DataFrame,
    impact_labels: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach pair-level impact metrics to each true attacked test prediction.

    The merge key is system, season, and the attacked raw row index. This keeps
    the calculation row-faithful even if the ML table has one row per model.
    """

    attack_rows = attack_predictions_only(predictions)
    system_attack_rows = attack_rows[attack_rows["system"] == config.system].copy()
    if system_attack_rows.empty:
        raise ValueError(f"No attacked prediction rows were found for {config.system}.")

    relevant_labels = impact_labels[impact_labels["system"] == config.system].copy()
    if relevant_labels.empty:
        raise ValueError(f"No impact labels were found for {config.system}.")

    expected_rows = len(system_attack_rows)
    merged = system_attack_rows.merge(
        relevant_labels,
        how="left",
        left_on=["system", "season", "raw_row_index"],
        right_on=["system", "season", "attacked_raw_row_index"],
        validate="many_to_one",
    )
    if len(merged) != expected_rows:
        raise ValueError(
            f"{config.system} impact merge changed row count from {expected_rows} "
            f"to {len(merged)}."
        )

    check_impact_merge_consistency(merged)

    metric_columns = [
        "delta_cost_abs",
        "delta_cost_pct",
        "dispatch_l1",
        "target_dispatch_change",
        "non_target_dispatch_change",
        "flow_l1",
        "ramp_l1",
    ]
    for column in metric_columns:
        merged[column] = pd.to_numeric(merged[column], errors="raise")

    merged["detected_attack"] = merged["predicted_attack"].astype(int)
    merged["missed_attack"] = (1 - merged["detected_attack"]).astype(int)

    return merged


def recall_summary(
    attack_predictions: pd.DataFrame,
    group_columns: list[str],
    impact_pattern: str | None = None,
) -> pd.DataFrame:
    """
    Summarize recall for either all attacks or one impact-pattern grouping.

    Recall is detected attacks divided by all true attacks in the group.
    """

    grouped = (
        attack_predictions.groupby(group_columns, dropna=False)
        .agg(
            attack_rows=("detected_attack", "size"),
            detected_rows=("detected_attack", "sum"),
            missed_rows=("missed_attack", "sum"),
            zero_response_rows=("zero_response", "sum"),
        )
        .reset_index()
    )
    grouped["recall"] = grouped["detected_rows"] / grouped["attack_rows"]

    if impact_pattern is not None:
        grouped["impact_pattern"] = impact_pattern

    return grouped


def summarize_recall_by_impact(predictions_with_impact: pd.DataFrame) -> pd.DataFrame:
    """
    Compute model recall overall and inside each physical-impact pattern.

    This is the revision-specific ML result: it shows whether a model mostly
    misses weak/non-binding attacks or also misses operationally important ones.
    """

    base_columns = ["system", "scenario_family", "model"]
    overall = recall_summary(
        predictions_with_impact,
        group_columns=base_columns,
        impact_pattern="overall_attack",
    )
    by_pattern = recall_summary(
        predictions_with_impact,
        group_columns=base_columns + ["impact_pattern"],
    )
    recall_table = pd.concat([overall, by_pattern], ignore_index=True)

    pattern_order = {
        impact_pattern: order
        for order, impact_pattern in enumerate(IMPACT_PATTERN_ORDER)
    }
    recall_table["impact_pattern_order"] = (
        recall_table["impact_pattern"].map(pattern_order).fillna(999).astype(int)
    )
    recall_table = recall_table.sort_values(
        ["system", "scenario_family", "model", "impact_pattern_order"]
    ).drop(columns=["impact_pattern_order"])

    return recall_table


def build_severity_recall_table(recall_by_impact: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the long recall summary into one compact row per system/model.

    This table shape is easier to read in the paper: each impact pattern becomes
    a pair of columns for sample count and recall.
    """

    id_columns = ["system", "scenario_family", "model"]
    rows: list[dict[str, object]] = []

    for key, group in recall_by_impact.groupby(id_columns, dropna=False):
        row = dict(zip(id_columns, key))
        by_pattern = group.set_index("impact_pattern")
        for impact_pattern in IMPACT_PATTERN_ORDER:
            short_name = impact_pattern.removesuffix("_attack")
            if impact_pattern in by_pattern.index:
                pattern_row = by_pattern.loc[impact_pattern]
                row[f"{short_name}_rows"] = int(pattern_row["attack_rows"])
                row[f"{short_name}_recall"] = float(pattern_row["recall"])
            else:
                row[f"{short_name}_rows"] = 0
                row[f"{short_name}_recall"] = np.nan
        rows.append(row)

    return pd.DataFrame(rows).sort_values(id_columns).reset_index(drop=True)


def build_impact_recall_outputs(
    config: FinalMLConfig,
    predictions: pd.DataFrame,
    impact_path: Path = PAIR_CLASSIFICATION_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build the severity-conditioned prediction and recall tables in memory.

    This is the analysis step. Writing to disk is kept separate so the final
    runner can save only consolidated outputs.
    """

    impact_labels = load_impact_labels(impact_path)
    predictions_with_impact = attach_impact_labels_to_predictions(
        config=config,
        predictions=predictions,
        impact_labels=impact_labels,
    )
    recall_by_impact = summarize_recall_by_impact(predictions_with_impact)
    severity_recall_table = build_severity_recall_table(recall_by_impact)

    return predictions_with_impact, recall_by_impact, severity_recall_table


def prepare_system_data(system: str) -> PreparedSystemData:
    """
    Run the approved data-preparation stages for one system.

    This freezes the upstream pipeline: audited split, existing feature
    engineering, numeric model matrix construction, and training-only MI feature
    selection.
    """

    config = load_final_ml_inputs(system)
    train_raw, test_raw = build_audited_train_test(config)
    train_features, test_features = engineer_train_test_features(config, train_raw, test_raw)
    model_inputs = build_model_inputs(train_features, test_features)
    ranked_inputs = rank_features_by_information(model_inputs)

    return PreparedSystemData(
        config=config,
        train_raw=train_raw,
        test_raw=test_raw,
        train_features=train_features,
        test_features=test_features,
        model_inputs=model_inputs,
        ranked_inputs=ranked_inputs,
    )


def build_selected_feature_table(
    config: FinalMLConfig,
    model_inputs: ModelInputs,
    ranked_inputs: FeatureRankedInputs,
) -> pd.DataFrame:
    """
    Create an auditable table of all candidate features and their MI scores.

    Selected features are marked explicitly, so the paper/debug trail can show
    both the full candidate pool and the subset passed into the models.
    """

    feature_table = ranked_inputs.feature_scores.copy()
    feature_table.insert(0, "system", config.system)
    feature_table.insert(1, "scenario_family", config.scenario_family)
    feature_table.insert(2, "candidate_rank", np.arange(1, len(feature_table) + 1))
    feature_table["candidate_feature_count"] = len(model_inputs.feature_names)
    feature_table["selected_feature_count"] = len(ranked_inputs.feature_names)

    selected_rank = {
        feature_name: rank
        for rank, feature_name in enumerate(ranked_inputs.feature_names, start=1)
    }
    feature_table["selected"] = feature_table["feature"].isin(selected_rank)
    feature_table["selected_rank"] = (
        feature_table["feature"].map(selected_rank).astype("Int64")
    )
    return feature_table


def run_system_pipeline(
    system: str,
    model_names: list[str] | None = None,
    n_iter: int | None = None,
    n_jobs: int = -1,
    output_dir: Path = SEVERITY_ML_OUTPUT_DIR,
    write_system_files: bool = True,
) -> SystemPipelineRun:
    """
    Run the current frozen ML pipeline for one system.

    This runs the approved train/test split, existing feature engineering,
    training-only feature ranking, model tuning, row-level prediction export,
    and severity-conditioned recall export.
    """

    prepared_data = prepare_system_data(system)
    selected_features = build_selected_feature_table(
        config=prepared_data.config,
        model_inputs=prepared_data.model_inputs,
        ranked_inputs=prepared_data.ranked_inputs,
    )
    tuned_results, model_summary = tune_selected_models(
        config=prepared_data.config,
        ranked_inputs=prepared_data.ranked_inputs,
        model_names=model_names,
        n_iter=n_iter,
        n_jobs=n_jobs,
    )
    test_predictions = build_test_prediction_table(
        config=prepared_data.config,
        test_metadata=prepared_data.test_features,
        tuned_results=tuned_results,
    )
    predictions_with_impact, recall_by_impact, severity_recall_table = (
        build_impact_recall_outputs(
            config=prepared_data.config,
            predictions=test_predictions,
        )
    )

    if write_system_files:
        output_dir.mkdir(parents=True, exist_ok=True)
        system_prefix = prepared_data.config.system.lower()
        selected_features.to_csv(
            output_dir / f"{system_prefix}_selected_features.csv",
            index=False,
        )
        model_summary.to_csv(
            output_dir / f"{system_prefix}_model_summary.csv",
            index=False,
        )
        test_predictions.to_csv(
            output_dir / f"{system_prefix}_test_predictions.csv",
            index=False,
        )
        predictions_with_impact.to_csv(
            output_dir / f"{system_prefix}_predictions_with_impact.csv",
            index=False,
        )
        recall_by_impact.to_csv(
            output_dir / f"{system_prefix}_recall_by_impact_pattern.csv",
            index=False,
        )
        severity_recall_table.to_csv(
            output_dir / f"{system_prefix}_severity_recall_table.csv",
            index=False,
        )

    return SystemPipelineRun(
        prepared_data=prepared_data,
        tuned_results=tuned_results,
        model_summary=model_summary,
        test_predictions=test_predictions,
        selected_features=selected_features,
        predictions_with_impact=predictions_with_impact,
        recall_by_impact=recall_by_impact,
        severity_recall_table=severity_recall_table,
    )


def sort_output_table(table_name: str, table: pd.DataFrame) -> pd.DataFrame:
    """Apply stable row ordering before writing consolidated CSV files."""

    if table_name == "recall_by_impact" and "impact_pattern" in table.columns:
        pattern_order = {
            impact_pattern: order
            for order, impact_pattern in enumerate(IMPACT_PATTERN_ORDER)
        }
        ordered_table = table.copy()
        ordered_table["_impact_pattern_order"] = (
            ordered_table["impact_pattern"].map(pattern_order).fillna(999).astype(int)
        )
        return (
            ordered_table.sort_values(
                ["system", "scenario_family", "model", "_impact_pattern_order"]
            )
            .drop(columns=["_impact_pattern_order"])
            .reset_index(drop=True)
        )

    sort_columns_by_table = {
        "model_summary": ["system", "scenario_family", "model"],
        "selected_features": ["system", "scenario_family", "candidate_rank"],
        "test_predictions": ["system", "scenario_family", "model", "test_row_position"],
        "predictions_with_impact": ["system", "scenario_family", "model", "test_row_position"],
        "recall_by_impact": ["system", "scenario_family", "model", "impact_pattern"],
        "severity_recall_table": ["system", "scenario_family", "model"],
    }
    sort_columns = [
        column for column in sort_columns_by_table.get(table_name, []) if column in table.columns
    ]
    if not sort_columns:
        return table.reset_index(drop=True)
    return table.sort_values(sort_columns).reset_index(drop=True)


def combine_system_runs(system_runs: list[SystemPipelineRun]) -> FinalPipelineRun:
    """
    Concatenate per-system pipeline products into final revision tables.

    The per-system objects are still kept inside the returned object, but the
    dataframes here are the only final CSVs we need to write for the paper run.
    """

    if not system_runs:
        raise ValueError("No system runs were provided for consolidation.")

    tables = {
        "model_summary": pd.concat(
            [run.model_summary for run in system_runs],
            ignore_index=True,
        ),
        "selected_features": pd.concat(
            [run.selected_features for run in system_runs],
            ignore_index=True,
        ),
        "test_predictions": pd.concat(
            [run.test_predictions for run in system_runs],
            ignore_index=True,
        ),
        "predictions_with_impact": pd.concat(
            [run.predictions_with_impact for run in system_runs],
            ignore_index=True,
        ),
        "recall_by_impact": pd.concat(
            [run.recall_by_impact for run in system_runs],
            ignore_index=True,
        ),
        "severity_recall_table": pd.concat(
            [run.severity_recall_table for run in system_runs],
            ignore_index=True,
        ),
    }
    sorted_tables = {
        table_name: sort_output_table(table_name, table)
        for table_name, table in tables.items()
    }

    return FinalPipelineRun(
        system_runs=system_runs,
        model_summary=sorted_tables["model_summary"],
        selected_features=sorted_tables["selected_features"],
        test_predictions=sorted_tables["test_predictions"],
        predictions_with_impact=sorted_tables["predictions_with_impact"],
        recall_by_impact=sorted_tables["recall_by_impact"],
        severity_recall_table=sorted_tables["severity_recall_table"],
    )


def write_consolidated_outputs(
    final_run: FinalPipelineRun,
    output_dir: Path = SEVERITY_ML_OUTPUT_DIR,
) -> dict[str, Path]:
    """
    Write the final consolidated CSV files for the revision ML analysis.

    These filenames are intentionally system-neutral. They are the clean outputs
    we should inspect and decide from before touching the manuscript text.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "model_summary": final_run.model_summary,
        "selected_features": final_run.selected_features,
        "test_predictions": final_run.test_predictions,
        "predictions_with_impact": final_run.predictions_with_impact,
        "recall_by_impact": final_run.recall_by_impact,
        "severity_recall_table": final_run.severity_recall_table,
    }
    written_paths: dict[str, Path] = {}
    for table_name, table in tables.items():
        output_path = output_dir / CONSOLIDATED_OUTPUT_FILES[table_name]
        table.to_csv(output_path, index=False)
        written_paths[table_name] = output_path
    return written_paths


def run_final_pipeline(
    systems: list[str] | None = None,
    model_names: list[str] | None = None,
    n_iter: int | None = None,
    n_jobs: int = -1,
    output_dir: Path = SEVERITY_ML_OUTPUT_DIR,
    write_system_files: bool = False,
) -> FinalPipelineRun:
    """
    Run the severity-conditioned ML workflow and save consolidated outputs.

    By default this writes only the six final CSV files. Set write_system_files
    to True only when debugging one system at a time.
    """

    selected_systems = list(systems or FINAL_SYSTEMS)
    system_runs: list[SystemPipelineRun] = []
    for system in selected_systems:
        print(f"\n=== Running {system} ===", flush=True)
        system_runs.append(
            run_system_pipeline(
                system=system,
                model_names=model_names,
                n_iter=n_iter,
                n_jobs=n_jobs,
                output_dir=output_dir,
                write_system_files=write_system_files,
            )
        )

    final_run = combine_system_runs(system_runs)
    written_paths = write_consolidated_outputs(final_run, output_dir=output_dir)
    print("\n=== Consolidated outputs ===", flush=True)
    for table_name, output_path in written_paths.items():
        print(f"{table_name}: {output_path}", flush=True)

    return final_run


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for the final severity ML run."""

    parser = argparse.ArgumentParser(
        description="Run severity-conditioned ML analysis and write consolidated CSVs."
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        default=list(FINAL_SYSTEMS),
        help="Power systems to run. Default: N14 N57 N300.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "Model names to run. Omit this option for the full model suite. "
            "Use quotes for names with spaces, e.g. 'Logistic Regression'."
        ),
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=None,
        help="RandomizedSearchCV iterations per model. Omit for notebook default.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for RandomizedSearchCV. Use 1 for the safest local run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SEVERITY_ML_OUTPUT_DIR,
        help="Directory for consolidated CSV outputs.",
    )
    parser.add_argument(
        "--write-system-files",
        action="store_true",
        help="Also write per-system CSV files such as n14_test_predictions.csv.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""

    args = parse_args(argv)
    run_final_pipeline(
        systems=[system.upper() for system in args.systems],
        model_names=args.models,
        n_iter=args.n_iter,
        n_jobs=args.n_jobs,
        output_dir=args.output_dir,
        write_system_files=args.write_system_files,
    )


if __name__ == "__main__":
    main()
