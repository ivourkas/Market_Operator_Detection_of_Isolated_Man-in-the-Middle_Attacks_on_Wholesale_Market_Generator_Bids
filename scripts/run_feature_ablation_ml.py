#!/usr/bin/env python3
"""
Run feature-family ablations for the revision ML analysis.

This script is intentionally built on top of run_severity_conditioned_ml.py.
The ablations should use the same audited Balanced 80/20 split, the same
feature engineering code, and the same severity-label merge as the final ML
table. Only the feature set is allowed to change.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from run_severity_conditioned_ml import (
    FINAL_SYSTEMS,
    PROJECT_ROOT,
    FeatureRankedInputs,
    ModelEvaluation,
    ModelInputs,
    PreparedSystemData,
    build_impact_recall_outputs,
    build_selected_feature_table,
    build_test_prediction_table,
    prepare_system_data,
    rank_features_by_information,
    tune_selected_models,
)


ABLATION_OUTPUT_DIR = PROJECT_ROOT / "results" / "revision" / "feature_ablation_ml"
DEFAULT_TREE_MODELS = ("XGBoost", "CatBoost", "Random Forest")
KNN_DISTANCE_FEATURE = "gen_anomaly_knn_distance"
THRESHOLD_MODEL_NAME = "kNN threshold"
TRAIN_OUTPUT_FILES = {
    "feature_counts": "ablation_feature_counts.csv",
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


@dataclass(frozen=True)
class FeatureAblation:
    """One controlled feature set used to test where detection signal comes from."""

    name: str
    description: str
    model_names: tuple[str, ...]


APPROVED_ABLATIONS = (
    FeatureAblation(
        name="knn_threshold",
        description=(
            "Use only the generator kNN anomaly distance with a training-normal "
            "threshold. This tests the simplest unsupervised detector."
        ),
        model_names=(),
    ),
    FeatureAblation(
        name="knn_logistic",
        description=(
            "Use only the generator kNN anomaly distance inside Logistic Regression. "
            "This tests whether supervised calibration adds value."
        ),
        model_names=("Logistic Regression",),
    ),
    FeatureAblation(
        name="full_current",
        description=(
            "Use the current feature set after training-only mutual-information ranking. "
            "This is the approved severity-conditioned ML baseline."
        ),
        model_names=DEFAULT_TREE_MODELS,
    ),
    FeatureAblation(
        name="full_without_knn",
        description=(
            "Use the current feature set after removing the kNN distance. This tests "
            "whether kNN adds information beyond the engineered features."
        ),
        model_names=DEFAULT_TREE_MODELS,
    ),
    FeatureAblation(
        name="compressed_generator_plus_network",
        description=(
            "Remove raw per-generator daily descriptors but keep the kNN distance, "
            "network-flow, angle, market, and system-summary features. This tests "
            "whether kNN can replace the large generator feature block."
        ),
        model_names=DEFAULT_TREE_MODELS,
    ),
    FeatureAblation(
        name="generator_features_only",
        description=(
            "Use raw per-generator descriptors and generator-system summaries, but "
            "exclude the kNN distance. This tests whether generator features alone "
            "explain the classifier behavior."
        ),
        model_names=DEFAULT_TREE_MODELS,
    ),
    FeatureAblation(
        name="network_market_only",
        description=(
            "Use network, angle, market, bus-distance, and phantom-generator features "
            "while excluding kNN and direct generator descriptors."
        ),
        model_names=DEFAULT_TREE_MODELS,
    ),
    FeatureAblation(
        name="full_without_fval",
        description=(
            "Use the current feature set after removing objective-value features. This "
            "checks whether detection depends on a feature that may be less available "
            "in real deployments."
        ),
        model_names=DEFAULT_TREE_MODELS,
    ),
)


def is_raw_generator_feature(feature_name: str) -> bool:
    """Return True for per-generator daily descriptors such as gen_4_bus249_mean."""

    return bool(re.match(r"^gen_\d+_bus\d+_", feature_name))


def is_generator_system_summary(feature_name: str) -> bool:
    """Return True for fleet-level generation summaries, not individual units."""

    return (
        feature_name.startswith("active_generators")
        or feature_name.startswith("total_generation")
        or feature_name.startswith("gen_concentration")
        or feature_name.startswith("gen_diversity")
    )


def is_direct_generator_feature(feature_name: str) -> bool:
    """Return True when a feature directly describes generator dispatch behavior."""

    return is_raw_generator_feature(feature_name) or is_generator_system_summary(feature_name)


def is_network_market_feature(feature_name: str) -> bool:
    """
    Return True for features that describe grid redistribution or market structure.

    These are allowed in the network_market_only ablation because they do not expose
    raw per-generator daily profiles or the generator kNN distance directly.
    """

    return (
        feature_name.startswith("flow_deviation")
        or feature_name.startswith("angle_deviation")
        or feature_name.startswith("bus_dist")
        or feature_name.startswith("high_bus_distance")
        or feature_name.startswith("phantom")
        or feature_name.startswith("merit_order")
        or "fval" in feature_name
    )


def select_features_for_ablation(
    feature_names: list[str],
    ablation_name: str,
) -> list[str]:
    """
    Select the candidate features allowed by one ablation.

    The selector is applied before mutual-information ranking, so each ablation
    gets its own training-only feature ranking within the allowed feature family.
    """

    if ablation_name in {"knn_threshold", "knn_logistic"}:
        return [feature for feature in feature_names if feature == KNN_DISTANCE_FEATURE]

    if ablation_name == "full_current":
        return list(feature_names)

    if ablation_name == "full_without_knn":
        return [feature for feature in feature_names if feature != KNN_DISTANCE_FEATURE]

    if ablation_name == "compressed_generator_plus_network":
        return [
            feature
            for feature in feature_names
            if not is_raw_generator_feature(feature)
        ]

    if ablation_name == "generator_features_only":
        return [
            feature
            for feature in feature_names
            if is_direct_generator_feature(feature) and feature != KNN_DISTANCE_FEATURE
        ]

    if ablation_name == "network_market_only":
        return [
            feature
            for feature in feature_names
            if is_network_market_feature(feature)
            and feature != KNN_DISTANCE_FEATURE
            and not is_direct_generator_feature(feature)
        ]

    if ablation_name == "full_without_fval":
        return [feature for feature in feature_names if "fval" not in feature]

    raise ValueError(f"Unknown ablation '{ablation_name}'.")


def build_ablation_inputs(
    base_inputs: ModelInputs,
    ablation_name: str,
) -> ModelInputs:
    """Return model matrices restricted to one ablation's allowed features."""

    selected_features = select_features_for_ablation(
        feature_names=base_inputs.feature_names,
        ablation_name=ablation_name,
    )
    if not selected_features:
        raise ValueError(f"Ablation '{ablation_name}' selected zero features.")

    return ModelInputs(
        X_train=base_inputs.X_train[selected_features].copy(),
        X_test=base_inputs.X_test[selected_features].copy(),
        y_train=base_inputs.y_train.copy(),
        y_test=base_inputs.y_test.copy(),
        feature_names=selected_features,
    )


def get_ablation(ablation_name: str) -> FeatureAblation:
    """Return the approved ablation definition with this exact name."""

    approved_by_name = {ablation.name: ablation for ablation in APPROVED_ABLATIONS}
    if ablation_name not in approved_by_name:
        approved = ", ".join(approved_by_name)
        raise ValueError(f"Unknown ablation '{ablation_name}'. Choose one of: {approved}.")
    return approved_by_name[ablation_name]


def expand_requested_ablations(ablation_names: list[str]) -> list[FeatureAblation]:
    """Resolve CLI ablation names, including the convenience alias 'all'."""

    normalized_names = [name.strip() for name in ablation_names]
    if any(name.lower() == "all" for name in normalized_names):
        if len(normalized_names) != 1:
            raise ValueError("Use either '--ablations all' or explicit ablation names, not both.")
        return list(APPROVED_ABLATIONS)

    return [get_ablation(ablation_name) for ablation_name in normalized_names]


def rank_ablation_features(model_inputs: ModelInputs) -> FeatureRankedInputs:
    """
    Rank features inside one ablation without peeking at test labels.

    Multi-feature ablations use the same mutual-information rule as the final
    ML pipeline. Single-feature ablations keep their one feature by design,
    because the experiment is asking whether that specific signal is sufficient.
    """

    if len(model_inputs.feature_names) != 1:
        return rank_features_by_information(model_inputs)

    feature_name = model_inputs.feature_names[0]
    try:
        from sklearn.feature_selection import mutual_info_classif
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "scikit-learn is required for ablation feature ranking."
        ) from exc

    mi_score = mutual_info_classif(
        model_inputs.X_train[[feature_name]],
        model_inputs.y_train,
        random_state=42,
    )[0]
    feature_scores = pd.DataFrame(
        {
            "feature": [feature_name],
            "mi_score": [float(mi_score) if np.isfinite(mi_score) else np.nan],
        }
    )

    return FeatureRankedInputs(
        X_train=model_inputs.X_train[[feature_name]].copy(),
        X_test=model_inputs.X_test[[feature_name]].copy(),
        y_train=model_inputs.y_train.copy(),
        y_test=model_inputs.y_test.copy(),
        feature_names=[feature_name],
        feature_scores=feature_scores,
    )


def add_ablation_columns(
    table: pd.DataFrame,
    ablation: FeatureAblation,
    candidate_features: int,
) -> pd.DataFrame:
    """Insert ablation metadata into an output table."""

    enriched = table.copy()
    metadata = {
        "ablation": ablation.name,
        "ablation_description": ablation.description,
        "ablation_candidate_features": candidate_features,
    }
    insert_at = min(2, len(enriched.columns))
    for column, value in metadata.items():
        if column in enriched.columns:
            enriched[column] = value
        else:
            enriched.insert(insert_at, column, value)
            insert_at += 1
    return enriched


def summarize_ablation_feature_counts(
    prepared_data: PreparedSystemData,
) -> pd.DataFrame:
    """Summarize how many candidate features each ablation keeps for one system."""

    rows: list[dict[str, object]] = []
    base_features = prepared_data.model_inputs.feature_names
    for ablation in APPROVED_ABLATIONS:
        selected_features = select_features_for_ablation(base_features, ablation.name)
        rows.append(
            {
                "system": prepared_data.config.system,
                "scenario_family": prepared_data.config.scenario_family,
                "ablation": ablation.name,
                "candidate_features": len(selected_features),
                "models": ", ".join(ablation.model_names) if ablation.model_names else "threshold",
                "description": ablation.description,
            }
        )

    return pd.DataFrame(rows)


@dataclass(frozen=True)
class PredictionOnlyResult:
    """Minimal prediction result accepted by build_test_prediction_table."""

    model_name: str
    evaluation: ModelEvaluation


@dataclass(frozen=True)
class AblationRun:
    """All tables produced by one feature ablation."""

    model_summary: pd.DataFrame
    selected_features: pd.DataFrame
    test_predictions: pd.DataFrame
    predictions_with_impact: pd.DataFrame
    recall_by_impact: pd.DataFrame
    severity_recall_table: pd.DataFrame


def run_supervised_ablation(
    prepared_data: PreparedSystemData,
    ablation: FeatureAblation,
    n_iter: int | None,
    n_jobs: int,
) -> AblationRun:
    """
    Train the approved supervised models for one feature ablation.

    Threshold-only ablations are handled separately because they do not fit a
    classifier. This function keeps the existing tuning and impact-recall logic.
    """

    if not ablation.model_names:
        raise ValueError(f"Ablation '{ablation.name}' is not a supervised model run.")

    ablation_inputs = build_ablation_inputs(
        base_inputs=prepared_data.model_inputs,
        ablation_name=ablation.name,
    )
    ranked_inputs = rank_ablation_features(ablation_inputs)
    candidate_features = len(ablation_inputs.feature_names)

    selected_features = build_selected_feature_table(
        config=prepared_data.config,
        model_inputs=ablation_inputs,
        ranked_inputs=ranked_inputs,
    )
    selected_features = add_ablation_columns(
        selected_features,
        ablation=ablation,
        candidate_features=candidate_features,
    )

    tuned_results, model_summary = tune_selected_models(
        config=prepared_data.config,
        ranked_inputs=ranked_inputs,
        model_names=list(ablation.model_names),
        n_iter=n_iter,
        n_jobs=n_jobs,
    )
    model_summary = add_ablation_columns(
        model_summary,
        ablation=ablation,
        candidate_features=candidate_features,
    )

    test_predictions = build_test_prediction_table(
        config=prepared_data.config,
        test_metadata=prepared_data.test_features,
        tuned_results=tuned_results,
    )
    test_predictions = add_ablation_columns(
        test_predictions,
        ablation=ablation,
        candidate_features=candidate_features,
    )

    predictions_with_impact, recall_by_impact, severity_recall_table = (
        build_impact_recall_outputs(
            config=prepared_data.config,
            predictions=test_predictions,
        )
    )
    predictions_with_impact = add_ablation_columns(
        predictions_with_impact,
        ablation=ablation,
        candidate_features=candidate_features,
    )
    recall_by_impact = add_ablation_columns(
        recall_by_impact,
        ablation=ablation,
        candidate_features=candidate_features,
    )
    severity_recall_table = add_ablation_columns(
        severity_recall_table,
        ablation=ablation,
        candidate_features=candidate_features,
    )

    return AblationRun(
        model_summary=model_summary,
        selected_features=selected_features,
        test_predictions=test_predictions,
        predictions_with_impact=predictions_with_impact,
        recall_by_impact=recall_by_impact,
        severity_recall_table=severity_recall_table,
    )


def evaluate_threshold_detector(
    ranked_inputs: FeatureRankedInputs,
    threshold_quantile: float,
) -> tuple[ModelEvaluation, float, int]:
    """
    Evaluate the kNN-distance threshold learned from training normal rows only.

    The threshold is the selected upper quantile of normal training distances.
    A test row is flagged as attacked only when its distance is greater than
    that threshold.
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
        raise ModuleNotFoundError("scikit-learn is required for threshold evaluation.") from exc

    if KNN_DISTANCE_FEATURE not in ranked_inputs.feature_names:
        raise ValueError("The kNN threshold detector requires gen_anomaly_knn_distance.")
    if not 0 < threshold_quantile < 1:
        raise ValueError("threshold_quantile must be strictly between 0 and 1.")

    train_scores = pd.to_numeric(
        ranked_inputs.X_train[KNN_DISTANCE_FEATURE], errors="raise"
    )
    test_scores = pd.to_numeric(
        ranked_inputs.X_test[KNN_DISTANCE_FEATURE], errors="raise"
    )
    normal_train_scores = train_scores[ranked_inputs.y_train.astype(int) == 0]
    if normal_train_scores.empty:
        raise ValueError("The threshold detector needs normal training rows.")

    threshold_value = float(normal_train_scores.quantile(threshold_quantile))
    train_predictions = (train_scores > threshold_value).astype(int)
    test_predictions = (test_scores > threshold_value).astype(int)

    test_roc_auc = None
    if ranked_inputs.y_test.nunique() > 1:
        try:
            test_roc_auc = float(roc_auc_score(ranked_inputs.y_test, test_scores))
        except ValueError:
            test_roc_auc = None

    evaluation = ModelEvaluation(
        model_name=THRESHOLD_MODEL_NAME,
        estimator=None,
        train_accuracy=float(accuracy_score(ranked_inputs.y_train, train_predictions)),
        train_precision=float(
            precision_score(ranked_inputs.y_train, train_predictions, zero_division=0)
        ),
        train_recall=float(
            recall_score(ranked_inputs.y_train, train_predictions, zero_division=0)
        ),
        train_f1=float(f1_score(ranked_inputs.y_train, train_predictions, zero_division=0)),
        test_accuracy=float(accuracy_score(ranked_inputs.y_test, test_predictions)),
        test_precision=float(
            precision_score(ranked_inputs.y_test, test_predictions, zero_division=0)
        ),
        test_recall=float(
            recall_score(ranked_inputs.y_test, test_predictions, zero_division=0)
        ),
        test_f1=float(f1_score(ranked_inputs.y_test, test_predictions, zero_division=0)),
        test_roc_auc=test_roc_auc,
        accuracy_gap=float(
            accuracy_score(ranked_inputs.y_train, train_predictions)
            - accuracy_score(ranked_inputs.y_test, test_predictions)
        ),
        f1_gap=float(
            f1_score(ranked_inputs.y_train, train_predictions, zero_division=0)
            - f1_score(ranked_inputs.y_test, test_predictions, zero_division=0)
        ),
        test_predictions=pd.Series(
            test_predictions,
            index=ranked_inputs.X_test.index,
            name="predicted_attack",
        ),
        test_attack_score=pd.Series(
            test_scores,
            index=ranked_inputs.X_test.index,
            name="attack_score",
        ),
    )
    return evaluation, threshold_value, len(normal_train_scores)


def summarize_threshold_result(
    prepared_data: PreparedSystemData,
    ablation: FeatureAblation,
    ranked_inputs: FeatureRankedInputs,
    evaluation: ModelEvaluation,
    threshold_quantile: float,
    threshold_value: float,
    train_normal_rows: int,
) -> pd.DataFrame:
    """Build a model-summary row for the threshold detector."""

    summary = pd.DataFrame(
        [
            {
                "system": prepared_data.config.system,
                "scenario_family": prepared_data.config.scenario_family,
                "model": THRESHOLD_MODEL_NAME,
                "train_rows": len(ranked_inputs.X_train),
                "test_rows": len(ranked_inputs.X_test),
                "selected_features": len(ranked_inputs.feature_names),
                "scoring": f"normal_q{int(round(threshold_quantile * 100))}_threshold",
                "scaling": "none",
                "scaled_features": 0,
                "requested_iterations": 0,
                "actual_iterations": 0,
                "best_cv_score": np.nan,
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
                    {
                        "distance_feature": KNN_DISTANCE_FEATURE,
                        "threshold_quantile": threshold_quantile,
                        "threshold_value": threshold_value,
                        "train_normal_rows": train_normal_rows,
                    },
                    sort_keys=True,
                ),
            }
        ]
    )
    return add_ablation_columns(
        summary,
        ablation=ablation,
        candidate_features=len(ranked_inputs.feature_names),
    )


def run_threshold_ablation(
    prepared_data: PreparedSystemData,
    ablation: FeatureAblation,
    threshold_quantile: float,
) -> AblationRun:
    """
    Run the kNN-distance threshold detector for one system.

    This is the non-parametric baseline: no classifier is trained, and the only
    learned quantity is a normal-operation distance threshold from training rows.
    """

    if ablation.name != "knn_threshold":
        raise ValueError("run_threshold_ablation only supports knn_threshold.")

    ablation_inputs = build_ablation_inputs(
        base_inputs=prepared_data.model_inputs,
        ablation_name=ablation.name,
    )
    ranked_inputs = rank_ablation_features(ablation_inputs)
    candidate_features = len(ablation_inputs.feature_names)

    selected_features = build_selected_feature_table(
        config=prepared_data.config,
        model_inputs=ablation_inputs,
        ranked_inputs=ranked_inputs,
    )
    selected_features = add_ablation_columns(
        selected_features,
        ablation=ablation,
        candidate_features=candidate_features,
    )

    evaluation, threshold_value, train_normal_rows = evaluate_threshold_detector(
        ranked_inputs=ranked_inputs,
        threshold_quantile=threshold_quantile,
    )
    model_summary = summarize_threshold_result(
        prepared_data=prepared_data,
        ablation=ablation,
        ranked_inputs=ranked_inputs,
        evaluation=evaluation,
        threshold_quantile=threshold_quantile,
        threshold_value=threshold_value,
        train_normal_rows=train_normal_rows,
    )

    threshold_result = PredictionOnlyResult(
        model_name=THRESHOLD_MODEL_NAME,
        evaluation=evaluation,
    )
    test_predictions = build_test_prediction_table(
        config=prepared_data.config,
        test_metadata=prepared_data.test_features,
        tuned_results=[threshold_result],
    )
    test_predictions = add_ablation_columns(
        test_predictions,
        ablation=ablation,
        candidate_features=candidate_features,
    )

    predictions_with_impact, recall_by_impact, severity_recall_table = (
        build_impact_recall_outputs(
            config=prepared_data.config,
            predictions=test_predictions,
        )
    )
    predictions_with_impact = add_ablation_columns(
        predictions_with_impact,
        ablation=ablation,
        candidate_features=candidate_features,
    )
    recall_by_impact = add_ablation_columns(
        recall_by_impact,
        ablation=ablation,
        candidate_features=candidate_features,
    )
    severity_recall_table = add_ablation_columns(
        severity_recall_table,
        ablation=ablation,
        candidate_features=candidate_features,
    )

    return AblationRun(
        model_summary=model_summary,
        selected_features=selected_features,
        test_predictions=test_predictions,
        predictions_with_impact=predictions_with_impact,
        recall_by_impact=recall_by_impact,
        severity_recall_table=severity_recall_table,
    )


def sort_ablation_output_table(table_name: str, table: pd.DataFrame) -> pd.DataFrame:
    """Apply a stable, ablation-aware row order before writing CSV files."""

    ordered_table = table.copy()
    if "impact_pattern" in ordered_table.columns:
        impact_order = {
            impact_pattern: order
            for order, impact_pattern in enumerate(IMPACT_PATTERN_ORDER)
        }
        ordered_table["_impact_pattern_order"] = (
            ordered_table["impact_pattern"].map(impact_order).fillna(999).astype(int)
        )

    sort_columns_by_table = {
        "feature_counts": ["system", "scenario_family", "ablation"],
        "model_summary": ["system", "scenario_family", "ablation", "model"],
        "selected_features": ["system", "scenario_family", "ablation", "candidate_rank"],
        "test_predictions": [
            "system",
            "scenario_family",
            "ablation",
            "model",
            "test_row_position",
        ],
        "predictions_with_impact": [
            "system",
            "scenario_family",
            "ablation",
            "model",
            "test_row_position",
        ],
        "recall_by_impact": [
            "system",
            "scenario_family",
            "ablation",
            "model",
            "_impact_pattern_order",
        ],
        "severity_recall_table": ["system", "scenario_family", "ablation", "model"],
    }
    sort_columns = [
        column
        for column in sort_columns_by_table.get(table_name, [])
        if column in ordered_table.columns
    ]
    if sort_columns:
        ordered_table = ordered_table.sort_values(sort_columns, kind="mergesort")

    return ordered_table.drop(columns=["_impact_pattern_order"], errors="ignore").reset_index(
        drop=True
    )


def write_training_outputs(
    output_dir: Path,
    feature_counts: pd.DataFrame,
    ablation_runs: list[AblationRun],
) -> dict[str, Path]:
    """Write consolidated output CSVs for the ablations that were actually run."""

    if not ablation_runs:
        raise ValueError("No ablation runs were provided for output writing.")

    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    output_tables = {
        "feature_counts": feature_counts,
        "model_summary": pd.concat(
            [run.model_summary for run in ablation_runs],
            ignore_index=True,
        ),
        "selected_features": pd.concat(
            [run.selected_features for run in ablation_runs],
            ignore_index=True,
        ),
        "test_predictions": pd.concat(
            [run.test_predictions for run in ablation_runs],
            ignore_index=True,
        ),
        "predictions_with_impact": pd.concat(
            [run.predictions_with_impact for run in ablation_runs],
            ignore_index=True,
        ),
        "recall_by_impact": pd.concat(
            [run.recall_by_impact for run in ablation_runs],
            ignore_index=True,
        ),
        "severity_recall_table": pd.concat(
            [run.severity_recall_table for run in ablation_runs],
            ignore_index=True,
        ),
    }

    for table_name, table in output_tables.items():
        output_path = output_dir / TRAIN_OUTPUT_FILES[table_name]
        sort_ablation_output_table(table_name, table).to_csv(output_path, index=False)
        written[table_name] = output_path

    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for the feature-ablation run."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("feature-counts", "train"),
        default="feature-counts",
        help="Use feature-counts for a dry run or train for supervised ablations.",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        default=list(FINAL_SYSTEMS),
        help="Systems to inspect. Default: N14 N57 N300.",
    )
    parser.add_argument(
        "--ablations",
        nargs="+",
        default=["knn_logistic"],
        help="Ablations to train, or 'all'. Default: knn_logistic.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=None,
        help="RandomizedSearchCV iterations per supervised model.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for RandomizedSearchCV.",
    )
    parser.add_argument(
        "--threshold-quantile",
        type=float,
        default=0.95,
        help="Training-normal distance quantile for knn_threshold. Default: 0.95.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ABLATION_OUTPUT_DIR,
        help="Directory for future ablation outputs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run either a feature-count dry run or selected supervised ablations."""

    args = parse_args(argv)
    all_counts: list[pd.DataFrame] = []
    ablation_runs: list[AblationRun] = []
    selected_ablations = expand_requested_ablations(args.ablations)

    for system in args.systems:
        print(f"\n=== Preparing {system} feature ablations ===", flush=True)
        prepared_data = prepare_system_data(system)
        counts = summarize_ablation_feature_counts(prepared_data)
        all_counts.append(counts)
        print(counts[["ablation", "candidate_features", "models"]].to_string(index=False))

        if args.mode == "train":
            for ablation in selected_ablations:
                if ablation.name == "knn_threshold":
                    print(
                        f"\nRunning {system} / {ablation.name} / "
                        f"normal Q{int(round(args.threshold_quantile * 100))} threshold",
                        flush=True,
                    )
                    ablation_runs.append(
                        run_threshold_ablation(
                            prepared_data=prepared_data,
                            ablation=ablation,
                            threshold_quantile=args.threshold_quantile,
                        )
                    )
                    continue

                if not ablation.model_names:
                    raise ValueError(f"Ablation '{ablation.name}' has no model definition.")
                print(
                    f"\nTraining {system} / {ablation.name} / "
                    f"{', '.join(ablation.model_names)}",
                    flush=True,
                )
                ablation_runs.append(
                    run_supervised_ablation(
                        prepared_data=prepared_data,
                        ablation=ablation,
                        n_iter=args.n_iter,
                        n_jobs=args.n_jobs,
                    )
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_counts = pd.concat(all_counts, ignore_index=True)

    if args.mode == "feature-counts":
        output_path = args.output_dir / TRAIN_OUTPUT_FILES["feature_counts"]
        sort_ablation_output_table("feature_counts", feature_counts).to_csv(
            output_path,
            index=False,
        )
        print(f"\nFeature-count dry run written to: {output_path}", flush=True)
        return

    written_paths = write_training_outputs(
        output_dir=args.output_dir,
        feature_counts=feature_counts,
        ablation_runs=ablation_runs,
    )
    print("\n=== Ablation outputs ===", flush=True)
    for table_name, output_path in written_paths.items():
        print(f"{table_name}: {output_path}", flush=True)


if __name__ == "__main__":
    main()
