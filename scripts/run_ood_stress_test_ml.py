#!/usr/bin/env python3
"""
Run the N300 coordinated-attack stress test for the revision.

"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_severity_conditioned_ml import (  # noqa: E402
    FeatureRankedInputs,
    FinalMLConfig,
    ModelInputs,
    TunedModelResult,
    build_audited_train_test,
    build_model_inputs,
    evaluate_fitted_model,
    import_feature_module,
    load_final_ml_inputs,
    rank_features_by_information,
    tune_selected_models,
)

DEFAULT_OUTPUT_TABLE = (
    PROJECT_ROOT
    / "results"
    / "revision"
    / "ood_stress_test"
    / "table_coordinated_attack_stress_test.csv"
)
DEFAULT_MODEL_NAMES = ("CatBoost", "XGBoost", "Random Forest")
DEFAULT_NUMERIC_ZERO_TOLERANCE = 1e-12
DEFAULT_SEARCH_ITERATIONS = 500
MATCHED_NORMAL_FVAL_TOLERANCE = 1.0

OOD_METADATA_ROWS = (
    "fval",
    "ssn",
    "primary_attack_type",
    "primary_generator",
    "ood_scenario_code",
    "secondary_attack_type",
    "secondary_generator",
    "primary_alpha",
    "secondary_alpha",
    "matched_profile_id",
    "matched_raw_row_index",
)

EXPECTED_SEASON_BLOCK = np.array([1, 2, 3, 4, 1, 2, 3, 4])
SAMPLES_PER_LOAD_DRAW = 8
SEASONS_PER_NORMAL_OR_ATTACK_BLOCK = 4

ATTACK_TYPE_NAMES = {
    0: "normal",
    1: "ramp",
    2: "upper_limit",
    3: "lower_limit",
    4: "cost_bid",
}

OOD_SCENARIO_NAMES = {
    101: "two_generators_same_parameter",
    102: "one_generator_two_parameters",
}


@dataclass(frozen=True)
class OODStressData:
    """Parsed OOD file in both ML-row and paired-response forms."""

    path: Path
    scenario_name: str
    raw_samples: pd.DataFrame
    metadata: pd.DataFrame
    pair_summary: pd.DataFrame


@dataclass(frozen=True)
class OODModelData:
    """Original N300 training data and coordinated OOD data after feature preparation."""

    config: FinalMLConfig
    numeric_zero_tolerance: float
    train_raw: pd.DataFrame
    test_raw: pd.DataFrame
    ood_raw: pd.DataFrame
    pair_summary: pd.DataFrame
    train_features: pd.DataFrame
    test_features: pd.DataFrame
    ood_features: pd.DataFrame
    control_inputs: FeatureRankedInputs
    ood_inputs: FeatureRankedInputs


def resolve_project_path(path: Path) -> Path:
    """Interpret relative paths from the project root."""

    return path if path.is_absolute() else PROJECT_ROOT / path


def read_ood_stress_file(csv_path: Path) -> OODStressData:
    """
    Read one MATLAB OOD stress-test file and return validated sample rows.

    The MATLAB export stores one sample per column. The existing ML feature
    pipeline expects one sample per row, so this function transposes only the
    OPF trajectory block and then appends the metadata columns expected by the
    N300 feature module.
    """

    csv_path = resolve_project_path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"OOD stress-test file not found: {csv_path}")

    matrix = pd.read_csv(csv_path, header=None)
    if matrix.shape[0] <= len(OOD_METADATA_ROWS):
        raise ValueError(f"{csv_path.name} does not contain an OPF block plus metadata rows.")
    if matrix.shape[1] % SAMPLES_PER_LOAD_DRAW != 0:
        raise ValueError(
            f"{csv_path.name} has {matrix.shape[1]} columns, which is not divisible "
            f"by the expected 8-column load-draw block."
        )
    if not np.isfinite(matrix.to_numpy(dtype=float)).all():
        raise ValueError(f"{csv_path.name} contains NaN or infinite values.")

    opf_row_count = matrix.shape[0] - len(OOD_METADATA_ROWS)
    opf_columns = [f"ft{i}" for i in range(1, opf_row_count + 1)]

    opf_samples = matrix.iloc[:opf_row_count].T.reset_index(drop=True)
    opf_samples.columns = opf_columns

    metadata = matrix.iloc[opf_row_count:].T.reset_index(drop=True)
    metadata.columns = OOD_METADATA_ROWS
    metadata = metadata.apply(pd.to_numeric, errors="raise")
    if (metadata["fval"] <= -99999).any():
        raise ValueError(f"{csv_path.name} contains infeasible OPF sentinel objective values.")
    metadata.insert(0, "ood_sample_id", np.arange(len(metadata), dtype=int))
    metadata.insert(1, "load_draw", metadata["ood_sample_id"] // SAMPLES_PER_LOAD_DRAW + 1)
    metadata.insert(2, "column_in_draw", metadata["ood_sample_id"] % SAMPLES_PER_LOAD_DRAW)

    validate_ood_metadata(csv_path, metadata)
    metadata = add_human_readable_metadata(metadata)

    raw_samples = build_feature_pipeline_rows(opf_samples, metadata)
    pair_summary = build_paired_response_summary(raw_samples, metadata, opf_columns)

    scenario_names = metadata["ood_scenario_name"].unique()
    if len(scenario_names) != 1:
        raise ValueError(f"{csv_path.name} contains multiple OOD scenarios: {scenario_names}.")

    return OODStressData(
        path=csv_path,
        scenario_name=str(scenario_names[0]),
        raw_samples=raw_samples,
        metadata=metadata,
        pair_summary=pair_summary,
    )


def prepare_n300_ood_model_data(
    ood_files: list[Path],
    numeric_zero_tolerance: float,
) -> OODModelData:
    """
    Fit N300 feature baselines on the original training split and transform OOD rows.

    The OOD rows are treated strictly as test data. Normal operating limits, flow
    and angle baselines, nearest-neighbor models, and mutual-information feature
    selection are all learned from the original N300 training split only.
    """

    config = load_final_ml_inputs("N300")
    train_raw, test_raw = build_audited_train_test(config)
    ood_datasets = [read_ood_stress_file(path) for path in ood_files]
    ood_raw, pair_summary = combine_ood_datasets(ood_datasets)
    ood_raw, pair_summary = build_matched_ood_test_rows(test_raw, ood_raw, pair_summary)
    validate_ood_cost_regime(train_raw, ood_raw)

    train_raw = harmonize_numerical_zeros(train_raw, numeric_zero_tolerance)
    test_raw = harmonize_numerical_zeros(test_raw, numeric_zero_tolerance)
    ood_raw = harmonize_numerical_zeros(ood_raw, numeric_zero_tolerance)
    ood_feature_raw = build_feature_engineering_view(ood_raw, train_raw.columns, "OOD")

    feature_module = import_feature_module(config)
    feature_store: dict[str, object] = {}
    train_features, feature_store = feature_module.engineer_advanced_features(
        train_raw.copy(),
        is_training=True,
        feature_store=feature_store,
    )
    test_features, _ = feature_module.engineer_advanced_features(
        test_raw.copy(),
        is_training=False,
        feature_store=feature_store,
    )
    ood_features, _ = feature_module.engineer_advanced_features(
        ood_feature_raw.copy(),
        is_training=False,
        feature_store=feature_store,
    )

    validate_feature_rows("train", train_raw, train_features)
    validate_feature_rows("original test", test_raw, test_features)
    validate_feature_rows("OOD", ood_feature_raw, ood_features)

    control_model_inputs = build_model_inputs(train_features, test_features)
    control_inputs = rank_features_by_information(control_model_inputs)
    ood_model_inputs = build_model_inputs(train_features, ood_features)
    ood_inputs = apply_selected_features(ood_model_inputs, control_inputs)

    return OODModelData(
        config=config,
        numeric_zero_tolerance=numeric_zero_tolerance,
        train_raw=train_raw,
        test_raw=test_raw,
        ood_raw=ood_raw,
        pair_summary=pair_summary,
        train_features=train_features,
        test_features=test_features,
        ood_features=ood_features,
        control_inputs=control_inputs,
        ood_inputs=ood_inputs,
    )


def apply_selected_features(
    model_inputs: ModelInputs,
    selected_inputs: FeatureRankedInputs,
) -> FeatureRankedInputs:
    """
    Apply the training-selected feature list to a second test target.

    Feature ranking is learned once from the original N300 training split. The
    coordinated stress rows inherit that feature list; they do not influence
    which predictors are selected.
    """

    missing = [
        feature
        for feature in selected_inputs.feature_names
        if feature not in model_inputs.X_train.columns or feature not in model_inputs.X_test.columns
    ]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Selected features are missing from the requested test target: {preview}")

    return FeatureRankedInputs(
        X_train=model_inputs.X_train[selected_inputs.feature_names],
        X_test=model_inputs.X_test[selected_inputs.feature_names],
        y_train=model_inputs.y_train,
        y_test=model_inputs.y_test,
        feature_names=selected_inputs.feature_names,
        feature_scores=selected_inputs.feature_scores,
    )


def harmonize_numerical_zeros(raw_rows: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    """
    Treat solver roundoff near zero as zero before feature engineering.

    The same rule is applied to the original N300 train rows and to the
    coordinated stress-test rows. Values below this tolerance are numerical
    dust, not meaningful generation, angle, or flow magnitudes.
    """

    if tolerance < 0:
        raise ValueError("numeric zero tolerance must be nonnegative.")
    cleaned = raw_rows.copy()
    if tolerance == 0:
        return cleaned

    opf_columns = [column for column in cleaned.columns if column.startswith("ft")]
    if not opf_columns:
        return cleaned

    opf_values = cleaned[opf_columns].apply(pd.to_numeric, errors="raise")
    cleaned.loc[:, opf_columns] = opf_values.mask(opf_values.abs() < tolerance, 0.0)
    return cleaned


def build_feature_engineering_view(
    raw_rows: pd.DataFrame,
    training_columns: pd.Index,
    split_name: str,
) -> pd.DataFrame:
    """
    Keep feature engineering on the same raw schema as the original N300 data.

    The matched OOD rows carry extra descriptive columns such as scenario names
    and attack-combination labels. Those columns are useful for reporting, but
    they are not part of the original notebook input schema and must not enter
    feature engineering or model selection.
    """

    required_columns = {
        "attack",
        "type_of_attack",
        "gen_attacked",
        "fval",
        "ssn",
        "original_ssn",
        "raw_row_index",
        "temporal_index",
    }
    required_columns.update(column for column in training_columns if column.startswith("ft"))

    missing_required = sorted(column for column in required_columns if column not in raw_rows.columns)
    if missing_required:
        preview = ", ".join(missing_required[:5])
        raise ValueError(f"{split_name} rows are missing required raw columns: {preview}")

    feature_rows = raw_rows.reindex(columns=training_columns).copy()
    for column in feature_rows.columns:
        feature_rows[column] = pd.to_numeric(feature_rows[column], errors="raise")
    return feature_rows


def combine_ood_datasets(ood_datasets: list[OODStressData]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Concatenate OOD scenarios while keeping sample identifiers globally unique.

    Each MATLAB file numbers its own columns from zero. Before combining files,
    the local sample ids are shifted so that predictions can be joined back to
    the correct scenario and normal/attack pair without ambiguity.
    """

    raw_frames: list[pd.DataFrame] = []
    pair_frames: list[pd.DataFrame] = []
    row_offset = 0

    for data in ood_datasets:
        raw = data.raw_samples.copy()
        pairs = data.pair_summary.copy()

        raw["ood_file"] = data.path.name
        raw["ood_local_sample_id"] = raw["ood_sample_id"].astype(int)
        raw["ood_global_sample_id"] = raw["ood_local_sample_id"] + row_offset
        raw["raw_row_index"] = raw["ood_global_sample_id"]
        raw["temporal_index"] = raw["ood_global_sample_id"] + 1

        pairs["ood_file"] = data.path.name
        pairs["normal_raw_row_index"] = pairs["normal_ood_sample_id"].astype(int) + row_offset
        pairs["attacked_raw_row_index"] = pairs["attacked_ood_sample_id"].astype(int) + row_offset

        raw_frames.append(raw)
        pair_frames.append(pairs)
        row_offset += len(raw)

    ood_raw = pd.concat(raw_frames, ignore_index=True)
    pair_summary = pd.concat(pair_frames, ignore_index=True)

    if ood_raw["raw_row_index"].nunique() != len(ood_raw):
        raise ValueError("Combined OOD rows do not have unique raw_row_index values.")
    if pair_summary["attacked_raw_row_index"].nunique() != len(pair_summary):
        raise ValueError("Combined OOD pairs do not have unique attacked row identifiers.")

    return ood_raw, pair_summary


def build_matched_ood_test_rows(
    test_raw: pd.DataFrame,
    ood_raw: pd.DataFrame,
    pair_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Replace generated OOD normal rows with original held-out N300 normal controls.

    Coordinated attack rows remain from MATLAB. The normal rows are copied from
    the exact original held-out normal rows identified by matched_raw_row_index.
    This isolates attack-structure generalization from normal-data drift.
    """

    test_normals = test_raw[test_raw["attack"] == 0].copy()
    test_normal_by_raw = test_normals.set_index(test_normals["raw_row_index"].astype(int), drop=False)

    ood_by_raw = ood_raw.set_index(ood_raw["raw_row_index"].astype(int), drop=False)
    required_normal_ids = set(pair_summary["matched_raw_row_index"].astype(int))
    missing_normal_ids = sorted(required_normal_ids - set(test_normal_by_raw.index.astype(int)))
    if missing_normal_ids:
        preview = ", ".join(str(value) for value in missing_normal_ids[:5])
        raise ValueError(f"OOD pairs reference held-out normal rows that are not in the original test split: {preview}")

    required_attack_ids = set(pair_summary["attacked_raw_row_index"].astype(int))
    missing_attack_ids = sorted(required_attack_ids - set(ood_by_raw.index.astype(int)))
    if missing_attack_ids:
        preview = ", ".join(str(value) for value in missing_attack_ids[:5])
        raise ValueError(f"OOD pairs reference attack rows that are missing from the parsed OOD data: {preview}")

    matched_rows: list[pd.Series] = []
    matched_pairs: list[pd.Series] = []
    next_row_id = 0

    ordered_pairs = pair_summary.sort_values(
        ["ood_scenario_name", "load_draw", "ssn", "attacked_raw_row_index"]
    ).reset_index(drop=True)
    for _, pair in ordered_pairs.iterrows():
        matched_raw_row_index = int(pair["matched_raw_row_index"])
        normal_source = test_normal_by_raw.loc[matched_raw_row_index].copy()
        generated_normal = ood_by_raw.loc[int(pair["normal_raw_row_index"])]
        attack_source = ood_by_raw.loc[int(pair["attacked_raw_row_index"])].copy()

        if int(normal_source["original_ssn"]) != int(pair["ssn"]):
            raise ValueError(
                "Matched held-out normal season does not agree with the coordinated attack season."
            )
        fval_difference = abs(float(generated_normal["fval"]) - float(normal_source["fval"]))
        if fval_difference > MATCHED_NORMAL_FVAL_TOLERANCE:
            raise ValueError(
                "Generated matched-load normal objective does not agree with the original "
                f"held-out normal row {matched_raw_row_index}: difference={fval_difference:.4f}."
            )

        normal_row_id = next_row_id
        attack_row_id = next_row_id + 1
        next_row_id += 2

        normal_row = prepare_matched_normal_row(normal_source, pair, normal_row_id)
        attack_row = prepare_matched_attack_row(attack_source, pair, attack_row_id)
        matched_rows.extend([normal_row, attack_row])

        matched_pair = pair.copy()
        matched_pair["normal_raw_row_index"] = normal_row_id
        matched_pair["attacked_raw_row_index"] = attack_row_id
        matched_pairs.append(matched_pair)

    matched_ood_raw = pd.DataFrame(matched_rows).reset_index(drop=True)
    matched_pair_summary = pd.DataFrame(matched_pairs).reset_index(drop=True)

    if matched_ood_raw["raw_row_index"].nunique() != len(matched_ood_raw):
        raise ValueError("Matched OOD evaluation rows do not have unique raw_row_index values.")
    if matched_pair_summary["attacked_raw_row_index"].nunique() != len(matched_pair_summary):
        raise ValueError("Matched OOD pair summary has duplicated attacked row identifiers.")

    return matched_ood_raw, matched_pair_summary


def prepare_matched_normal_row(
    normal_source: pd.Series,
    pair: pd.Series,
    row_id: int,
) -> pd.Series:
    """Copy one original held-out normal row into a scenario-specific OOD control row."""

    normal_row = normal_source.copy()
    normal_row["raw_row_index"] = row_id
    normal_row["temporal_index"] = row_id + 1
    normal_row["attack"] = 0
    normal_row["type_of_attack"] = 0
    normal_row["gen_attacked"] = 0
    normal_row["ssn"] = int(pair["ssn"])
    normal_row["original_ssn"] = int(pair["ssn"])
    normal_row["ood_scenario_name"] = pair["ood_scenario_name"]
    normal_row["attack_combo"] = "normal"
    normal_row["primary_attack_name"] = "normal"
    normal_row["secondary_attack_name"] = "normal"
    normal_row["primary_generator"] = 0
    normal_row["secondary_generator"] = 0
    normal_row["matched_profile_id"] = int(pair["matched_profile_id"])
    normal_row["matched_raw_row_index"] = int(pair["matched_raw_row_index"])
    return normal_row


def prepare_matched_attack_row(
    attack_source: pd.Series,
    pair: pd.Series,
    row_id: int,
) -> pd.Series:
    """Copy one coordinated attack row and attach the matched original control id."""

    attack_row = attack_source.copy()
    attack_row["raw_row_index"] = row_id
    attack_row["temporal_index"] = row_id + 1
    attack_row["ssn"] = int(pair["ssn"])
    attack_row["original_ssn"] = int(pair["ssn"])
    attack_row["matched_profile_id"] = int(pair["matched_profile_id"])
    attack_row["matched_raw_row_index"] = int(pair["matched_raw_row_index"])
    return attack_row


def validate_ood_cost_regime(train_raw: pd.DataFrame, ood_raw: pd.DataFrame) -> None:
    """
    Stop early if OOD normal cases are not comparable with the original N300 data.

    The coordinated stress test is meaningful only if its normal columns are
    generated from the same case/load regime as the original N300 training data.
    A large normal-cost shift means the classifier is being tested on a new
    benchmark case, not only on a new attack structure.
    """

    train_normal = train_raw[train_raw["attack"] == 0]["fval"].astype(float)
    ood_normal = ood_raw[ood_raw["attack"] == 0]["fval"].astype(float)
    if train_normal.empty or ood_normal.empty:
        raise ValueError("Cannot compare normal-cost regimes without normal rows.")

    train_mean = float(train_normal.mean())
    ood_mean = float(ood_normal.mean())
    relative_shift = abs(ood_mean - train_mean) / max(abs(train_mean), 1.0)
    if relative_shift > 0.20:
        raise ValueError(
            "OOD normal objective values are far from the original N300 training regime. "
            f"Training-normal mean fval={train_mean:.2f}, OOD-normal mean fval={ood_mean:.2f} "
            f"({100 * relative_shift:.1f}% shift). Regenerate the OOD files with the same "
            "case used by data/N300_G69_transposed_new.csv, i.e. Cases/pglib_opf_case300_ieee.m."
        )


def validate_feature_rows(split_name: str, raw_rows: pd.DataFrame, feature_rows: pd.DataFrame) -> None:
    """Confirm feature engineering did not add, drop, or duplicate sample rows."""

    if len(raw_rows) != len(feature_rows):
        raise ValueError(
            f"{split_name} feature engineering changed row count from "
            f"{len(raw_rows)} to {len(feature_rows)}."
        )
    if "raw_row_index" not in feature_rows.columns:
        raise ValueError(f"{split_name} features lost raw_row_index.")

    raw_ids = pd.to_numeric(raw_rows["raw_row_index"], errors="raise")
    feature_ids = pd.to_numeric(feature_rows["raw_row_index"], errors="raise")
    if raw_ids.nunique() != len(raw_ids) or feature_ids.nunique() != len(feature_ids):
        raise ValueError(f"{split_name} rows contain duplicate raw_row_index values.")
    if set(raw_ids.tolist()) != set(feature_ids.tolist()):
        raise ValueError(f"{split_name} feature engineering lost or added raw rows.")


def tune_stress_test_models(
    model_data: OODModelData,
    model_names: list[str],
    n_iter: int,
    n_jobs: int,
) -> list[TunedModelResult]:
    """
    Tune models on the cleaned original N300 train/control split.

    The coordinated stress rows are not used for feature ranking, cross-validation,
    hyperparameter selection, or fitting. They are scored only after the final
    estimators have been selected on the original held-out control split.
    """

    tuned_results, _ = tune_selected_models(
        config=model_data.config,
        ranked_inputs=model_data.control_inputs,
        model_names=model_names,
        n_iter=n_iter,
        n_jobs=n_jobs,
    )
    return tuned_results


def build_ood_prediction_table(
    model_data: OODModelData,
    estimator: object,
    model_name: str,
) -> pd.DataFrame:
    """Score OOD rows and attach the metadata needed for the stress-test table."""

    evaluation = evaluate_fitted_model(model_name, estimator, model_data.ood_inputs)
    prediction_metadata = model_data.ood_features[["raw_row_index"]].merge(
        model_data.ood_raw[
            [
                "raw_row_index",
                "attack",
                "ssn",
                "ood_scenario_name",
                "attack_combo",
                "primary_attack_name",
                "secondary_attack_name",
                "primary_generator",
                "secondary_generator",
                "matched_profile_id",
                "matched_raw_row_index",
            ]
        ],
        how="left",
        on="raw_row_index",
        validate="one_to_one",
    )
    if prediction_metadata["attack"].isna().any():
        raise ValueError("Could not align OOD predictions with raw-row metadata.")

    predictions = prediction_metadata[
        [
            "raw_row_index",
            "attack",
            "ssn",
            "ood_scenario_name",
            "attack_combo",
            "primary_attack_name",
            "secondary_attack_name",
            "primary_generator",
            "secondary_generator",
            "matched_profile_id",
            "matched_raw_row_index",
        ]
    ].copy()
    predictions.insert(0, "model", model_name)
    predictions["predicted_attack"] = evaluation.test_predictions.to_numpy(dtype=int)
    predictions["attack_score"] = evaluation.test_attack_score.to_numpy(dtype=float)
    return predictions


def pct_true(values: pd.Series) -> float:
    """Return a percentage for a Boolean/int series, or NaN if no rows exist."""

    if len(values) == 0:
        return float("nan")
    return 100 * float(values.mean())


def build_control_sanity_table(
    model_data: OODModelData,
    tuned_results: list[TunedModelResult],
) -> pd.DataFrame:
    """
    Summarize performance on the original held-out N300 test split.

    This is the internal control for the OOD stress test. If the cleaned protocol
    performs poorly here, the coordinated-attack results should not be interpreted
    as a robustness result.
    """

    y_true = model_data.control_inputs.y_test.astype(int)
    normal_mask = y_true == 0
    attack_mask = y_true == 1

    rows: list[dict[str, object]] = []
    for result in tuned_results:
        y_pred = result.evaluation.test_predictions.astype(int)
        rows.append(
            {
                "model": result.model_name,
                "normal_samples": int(normal_mask.sum()),
                "attack_samples": int(attack_mask.sum()),
                "accuracy_pct": 100 * result.evaluation.test_accuracy,
                "precision_pct": 100 * result.evaluation.test_precision,
                "attack_recall_pct": pct_true(y_pred[attack_mask]),
                "normal_false_alarm_pct": pct_true(y_pred[normal_mask]),
                "f1_pct": 100 * result.evaluation.test_f1,
            }
        )

    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


def build_ood_stress_table(
    predictions: pd.DataFrame,
    pair_summary: pd.DataFrame,
    numeric_zero_tolerance: float,
) -> pd.DataFrame:
    """
    Summarize coordinated-attack detection by stress-test scenario.

    Recall is computed on all attacked OOD rows. Response-only recall excludes
    zero-response pairs where the coordinated attack did not change the OPF
    objective or state, because those cases have no observable downstream effect.
    """

    attack_predictions = predictions[predictions["attack"] == 1].copy()
    normal_predictions = predictions[predictions["attack"] == 0].copy()

    merged_attacks = attack_predictions.merge(
        pair_summary,
        how="left",
        left_on="raw_row_index",
        right_on="attacked_raw_row_index",
        validate="many_to_one",
        suffixes=("", "_pair"),
    )
    if merged_attacks["zero_response"].isna().any():
        raise ValueError("Some OOD attack predictions could not be matched to paired response rows.")

    try:
        from sklearn.metrics import accuracy_score, f1_score, precision_score
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("scikit-learn is required for OOD stress-test metrics.") from exc

    rows: list[dict[str, object]] = []
    group_columns = ["model", "ood_scenario_name"]
    for (model_name, scenario_name), scenario_attacks in merged_attacks.groupby(group_columns, sort=True):
        scenario_mask = (
            (predictions["model"] == model_name)
            & (predictions["ood_scenario_name"] == scenario_name)
        )
        scenario_predictions = predictions[scenario_mask]
        scenario_normals = normal_predictions[
            (normal_predictions["model"] == model_name)
            & (normal_predictions["ood_scenario_name"] == scenario_name)
        ]
        responsive_attacks = scenario_attacks[~scenario_attacks["zero_response"]]

        y_true = scenario_predictions["attack"].astype(int)
        y_pred = scenario_predictions["predicted_attack"].astype(int)

        rows.append(
            {
                "system": "N300",
                "training_protocol": "balanced_80_20_single_point",
                "stress_test_scenario": scenario_name,
                "model": model_name,
                "cost_changed_pct": pct_true(scenario_attacks["cost_changed"]),
                "opf_state_changed_pct": pct_true(scenario_attacks["opf_state_changed"]),
                "zero_response_pct": pct_true(scenario_attacks["zero_response"]),
                "attack_recall_pct": pct_true(scenario_attacks["predicted_attack"]),
                "response_only_recall_pct": pct_true(responsive_attacks["predicted_attack"]),
                "normal_false_alarm_pct": pct_true(scenario_normals["predicted_attack"]),
                "precision_pct": 100 * float(precision_score(y_true, y_pred, zero_division=0)),
                "f1_pct": 100 * float(f1_score(y_true, y_pred, zero_division=0)),
                "accuracy_pct": 100 * float(accuracy_score(y_true, y_pred)),
                "normal_samples": len(scenario_normals),
                "attack_samples": len(scenario_attacks),
                "numeric_zero_tolerance": numeric_zero_tolerance,
            }
        )

    return pd.DataFrame(rows).sort_values(["stress_test_scenario", "model"]).reset_index(drop=True)


def write_paper_table(stress_table: pd.DataFrame, output_table: Path) -> None:
    """Save the one OOD table intended for manuscript reporting."""

    output_table = resolve_project_path(output_table)
    output_table.parent.mkdir(parents=True, exist_ok=True)
    stress_table.to_csv(output_table, index=False)
    print(f"\nSaved OOD stress-test table: {output_table}")


def validate_ood_metadata(csv_path: Path, metadata: pd.DataFrame) -> None:
    """
    Check that the OOD file follows the intended paired seasonal layout.

    Each load draw must contain four normal seasons followed by the same four
    attacked seasons. Normal columns should have zero attack metadata; attacked
    columns should carry the coordinated-attack metadata.
    """

    if len(metadata) % SAMPLES_PER_LOAD_DRAW != 0:
        raise ValueError(f"{csv_path.name} has an incomplete 8-column load-draw block.")

    blocks = metadata.to_numpy().reshape(-1, SAMPLES_PER_LOAD_DRAW, metadata.shape[1])
    season_col = metadata.columns.get_loc("ssn")
    primary_type_col = metadata.columns.get_loc("primary_attack_type")
    primary_gen_col = metadata.columns.get_loc("primary_generator")
    secondary_type_col = metadata.columns.get_loc("secondary_attack_type")
    secondary_gen_col = metadata.columns.get_loc("secondary_generator")
    primary_alpha_col = metadata.columns.get_loc("primary_alpha")
    secondary_alpha_col = metadata.columns.get_loc("secondary_alpha")
    matched_profile_col = metadata.columns.get_loc("matched_profile_id")
    matched_raw_row_col = metadata.columns.get_loc("matched_raw_row_index")
    scenario_col = metadata.columns.get_loc("ood_scenario_code")

    seasons = blocks[:, :, season_col].astype(int)
    if not np.all(seasons == EXPECTED_SEASON_BLOCK):
        raise ValueError(f"{csv_path.name} does not repeat seasons as 1,2,3,4,1,2,3,4.")

    normal_block = blocks[:, :SEASONS_PER_NORMAL_OR_ATTACK_BLOCK, :]
    attack_block = blocks[:, SEASONS_PER_NORMAL_OR_ATTACK_BLOCK:, :]

    normal_primary = normal_block[:, :, primary_type_col]
    normal_generators = normal_block[:, :, primary_gen_col]
    normal_secondary = normal_block[:, :, secondary_type_col]
    normal_secondary_generators = normal_block[:, :, secondary_gen_col]
    normal_primary_alpha = normal_block[:, :, primary_alpha_col]
    normal_secondary_alpha = normal_block[:, :, secondary_alpha_col]
    normal_matched_profile = normal_block[:, :, matched_profile_col]
    normal_matched_raw_row = normal_block[:, :, matched_raw_row_col]
    if not (
        np.all(normal_primary == 0)
        and np.all(normal_generators == 0)
        and np.all(normal_secondary == 0)
        and np.all(normal_secondary_generators == 0)
        and np.all(normal_primary_alpha == 0)
        and np.all(normal_secondary_alpha == 0)
    ):
        raise ValueError(f"{csv_path.name} has nonzero attack metadata in normal columns.")

    attack_primary_type = attack_block[:, :, primary_type_col]
    attack_primary_gen = attack_block[:, :, primary_gen_col]
    attack_secondary_type = attack_block[:, :, secondary_type_col]
    attack_secondary_gen = attack_block[:, :, secondary_gen_col]
    attack_primary_alpha = attack_block[:, :, primary_alpha_col]
    attack_secondary_alpha = attack_block[:, :, secondary_alpha_col]
    attack_scenario_code = attack_block[:, :, scenario_col].astype(int)
    attack_matched_profile = attack_block[:, :, matched_profile_col]
    attack_matched_raw_row = attack_block[:, :, matched_raw_row_col]

    if np.any(attack_primary_type == 0):
        raise ValueError(f"{csv_path.name} has attacked columns without a primary attack type.")
    if np.any(attack_primary_gen == 0):
        raise ValueError(f"{csv_path.name} has attacked columns without a primary generator.")
    if np.any(attack_secondary_type == 0):
        raise ValueError(f"{csv_path.name} has attacked columns without a secondary attack type.")
    if np.any(attack_secondary_gen == 0):
        raise ValueError(f"{csv_path.name} has attacked columns without a secondary generator.")
    if np.any(attack_primary_alpha <= 0) or np.any(attack_secondary_alpha <= 0):
        raise ValueError(f"{csv_path.name} has attacked columns without positive alpha values.")
    if np.any(normal_matched_profile != attack_matched_profile):
        raise ValueError(f"{csv_path.name} does not preserve matched profile IDs across normal/attack pairs.")
    if np.any(normal_matched_raw_row != attack_matched_raw_row):
        raise ValueError(f"{csv_path.name} does not preserve matched raw-row IDs across normal/attack pairs.")
    if np.any(attack_matched_raw_row < 0):
        raise ValueError(f"{csv_path.name} is missing matched original normal raw-row IDs.")

    scenario_codes = sorted(metadata["ood_scenario_code"].astype(int).unique())
    unknown_codes = [code for code in scenario_codes if code not in OOD_SCENARIO_NAMES]
    if unknown_codes:
        raise ValueError(f"{csv_path.name} contains unknown OOD scenario codes: {unknown_codes}.")

    if scenario_codes == [101]:
        if np.any(attack_primary_gen == attack_secondary_gen):
            raise ValueError(f"{csv_path.name} should use two different generators for scenario 101.")
        if np.any(attack_primary_type != attack_secondary_type):
            raise ValueError(f"{csv_path.name} should use one shared attack type for scenario 101.")
    elif scenario_codes == [102]:
        if np.any(attack_primary_gen != attack_secondary_gen):
            raise ValueError(f"{csv_path.name} should use one shared generator for scenario 102.")
        if np.any(attack_primary_type == attack_secondary_type):
            raise ValueError(f"{csv_path.name} should use two different attack types for scenario 102.")
    elif len(set(attack_scenario_code.ravel().tolist())) > 1:
        raise ValueError(f"{csv_path.name} mixes OOD scenarios within one file.")


def add_human_readable_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    """Add names for scenario and attack codes without changing numeric columns."""

    enriched = metadata.copy()
    enriched["attack"] = (enriched["primary_attack_type"] != 0).astype(int)
    enriched["ood_scenario_name"] = enriched["ood_scenario_code"].astype(int).map(OOD_SCENARIO_NAMES)
    enriched["primary_attack_name"] = enriched["primary_attack_type"].astype(int).map(ATTACK_TYPE_NAMES)
    enriched["secondary_attack_name"] = enriched["secondary_attack_type"].astype(int).map(ATTACK_TYPE_NAMES)
    enriched["attack_combo"] = (
        enriched["primary_attack_name"] + "+" + enriched["secondary_attack_name"]
    )
    enriched.loc[enriched["attack"] == 0, "attack_combo"] = "normal"
    return enriched


def build_feature_pipeline_rows(opf_samples: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Build rows compatible with the existing N300 feature-engineering module.

    The coordinated OOD metadata is preserved for reporting, while
    type_of_attack/gen_attacked mirror the old single-point column names so the
    feature code can process the rows without a separate branch.
    """

    rows = pd.concat([opf_samples, metadata], axis=1)
    rows["fval"] = metadata["fval"]
    rows["ssn"] = metadata["ssn"].astype(int)
    rows["type_of_attack"] = metadata["primary_attack_type"].astype(int)
    rows["gen_attacked"] = metadata["primary_generator"].astype(int)
    rows["attack"] = metadata["attack"].astype(int)
    rows["raw_row_index"] = metadata["ood_sample_id"].astype(int)
    rows["original_ssn"] = rows["ssn"].astype(int)
    rows["year"] = ((metadata["load_draw"].astype(int) - 1) // 12) + 1
    rows["repetition"] = rows["year"].astype(int)
    rows["day_in_season"] = metadata["load_draw"].astype(int)
    rows["session_day"] = metadata["load_draw"].astype(int)
    rows["temporal_index"] = metadata["ood_sample_id"].astype(int) + 1
    return rows


def build_paired_response_summary(
    raw_samples: pd.DataFrame,
    metadata: pd.DataFrame,
    opf_columns: list[str],
) -> pd.DataFrame:
    """
    Pair each attacked season with its exact normal load-season counterpart.

    Within each 8-column load draw, columns 0--3 are normal winter/spring/
    summer/autumn and columns 4--7 are the corresponding attacked seasons.
    The pair summary is only descriptive; it is not used to fit or tune ML.
    """

    pairs: list[dict[str, object]] = []
    opf_values = raw_samples[opf_columns].to_numpy(dtype=float)
    fval = raw_samples["fval"].to_numpy(dtype=float)

    for block_start in range(0, len(raw_samples), SAMPLES_PER_LOAD_DRAW):
        load_draw = int(metadata.loc[block_start, "load_draw"])
        for season_offset in range(SEASONS_PER_NORMAL_OR_ATTACK_BLOCK):
            normal_row = block_start + season_offset
            attacked_row = block_start + SEASONS_PER_NORMAL_OR_ATTACK_BLOCK + season_offset
            state_difference = opf_values[attacked_row] - opf_values[normal_row]
            attacked_meta = metadata.loc[attacked_row]

            pairs.append(
                {
                    "ood_scenario_name": attacked_meta["ood_scenario_name"],
                    "load_draw": load_draw,
                    "ssn": int(attacked_meta["ssn"]),
                    "normal_ood_sample_id": int(metadata.loc[normal_row, "ood_sample_id"]),
                    "attacked_ood_sample_id": int(attacked_meta["ood_sample_id"]),
                    "primary_attack_type": int(attacked_meta["primary_attack_type"]),
                    "secondary_attack_type": int(attacked_meta["secondary_attack_type"]),
                    "primary_attack_name": attacked_meta["primary_attack_name"],
                    "secondary_attack_name": attacked_meta["secondary_attack_name"],
                    "attack_combo": attacked_meta["attack_combo"],
                    "primary_generator": int(attacked_meta["primary_generator"]),
                    "secondary_generator": int(attacked_meta["secondary_generator"]),
                    "primary_alpha": float(attacked_meta["primary_alpha"]),
                    "secondary_alpha": float(attacked_meta["secondary_alpha"]),
                    "matched_profile_id": int(attacked_meta["matched_profile_id"]),
                    "matched_raw_row_index": int(attacked_meta["matched_raw_row_index"]),
                    "delta_cost_abs": abs(float(fval[attacked_row] - fval[normal_row])),
                    "opf_state_l1": float(np.abs(state_difference).sum()),
                    "opf_state_max_abs": float(np.abs(state_difference).max()),
                }
            )

    summary = pd.DataFrame(pairs)
    summary["cost_changed"] = summary["delta_cost_abs"] > 1e-7
    summary["opf_state_changed"] = summary["opf_state_max_abs"] > 1e-7
    summary["zero_response"] = ~summary["cost_changed"] & ~summary["opf_state_changed"]
    return summary


def print_audit_summary(data: OODStressData) -> None:
    """Print the checks that matter before the ML stress-test step."""

    pairs = data.pair_summary
    attacks = data.metadata[data.metadata["attack"] == 1]
    normal = data.metadata[data.metadata["attack"] == 0]

    print(f"\n=== {data.path.name} ===")
    print(f"scenario: {data.scenario_name}")
    print(f"samples: {len(data.raw_samples)} total = {len(normal)} normal + {len(attacks)} attacked")
    print(f"paired seasonal cases: {len(pairs)}")
    print(f"cost changed: {pairs['cost_changed'].sum()} / {len(pairs)} ({100 * pairs['cost_changed'].mean():.1f}%)")
    print(
        "OPF state changed: "
        f"{pairs['opf_state_changed'].sum()} / {len(pairs)} "
        f"({100 * pairs['opf_state_changed'].mean():.1f}%)"
    )
    print(f"zero response: {pairs['zero_response'].sum()} / {len(pairs)} ({100 * pairs['zero_response'].mean():.1f}%)")
    print("attack combinations:")
    print(pairs["attack_combo"].value_counts().sort_index().to_string())


def print_stress_test_summary(
    model_data: OODModelData,
    control_table: pd.DataFrame,
    stress_table: pd.DataFrame,
) -> None:
    """Print the compact ML result intended to become the paper table."""

    print("\n=== N300 OOD stress-test model ===")
    print(f"training rows: {len(model_data.control_inputs.X_train)} original N300 balanced 80/20 samples")
    print(f"original test rows: {len(model_data.test_raw)} N300 balanced 80/20 samples")
    print(f"OOD test rows: {len(model_data.ood_inputs.X_test)} coordinated-stress samples")
    print(f"selected features: {len(model_data.control_inputs.feature_names)}")
    print(f"near-zero OPF tolerance: {model_data.numeric_zero_tolerance:g}")
    print("models: retuned on cleaned original N300 balanced 80/20 data")
    print("\nOriginal held-out control check:")
    print(control_table.round(2).to_string(index=False))
    print("\nStress-test detection table:")
    display_table = stress_table.drop(columns=["numeric_zero_tolerance"]).round(2)
    print(display_table.to_string(index=False))


def parse_args() -> argparse.Namespace:
    """Read command-line options for the OOD stress-test script."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ood-files",
        nargs="+",
        type=Path,
        required=True,
        help="OOD MATLAB CSV files to audit/evaluate.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODEL_NAMES),
        help="Models to tune on original N300 training rows before OOD scoring.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=DEFAULT_SEARCH_ITERATIONS,
        help="RandomizedSearchCV iterations per model.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel jobs passed to RandomizedSearchCV.",
    )
    parser.add_argument(
        "--numeric-zero-tolerance",
        type=float,
        default=DEFAULT_NUMERIC_ZERO_TOLERANCE,
        help=(
            "Set OPF values with absolute magnitude below this tolerance to zero "
            "before feature engineering. The same rule is applied to original "
            "N300 training rows and coordinated stress-test rows."
        ),
    )
    parser.add_argument(
        "--output-table",
        type=Path,
        default=DEFAULT_OUTPUT_TABLE,
        help="Where to save the single paper-ready OOD stress-test table.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Only parse and validate OOD files; skip model fitting and prediction.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the OOD parser audit and, unless skipped, the coordinated stress test."""

    args = parse_args()
    for csv_path in args.ood_files:
        data = read_ood_stress_file(csv_path)
        print_audit_summary(data)

    if args.audit_only:
        return

    model_data = prepare_n300_ood_model_data(
        args.ood_files,
        numeric_zero_tolerance=args.numeric_zero_tolerance,
    )
    tuned_results = tune_stress_test_models(
        model_data=model_data,
        model_names=args.models,
        n_iter=args.n_iter,
        n_jobs=args.n_jobs,
    )
    control_table = build_control_sanity_table(model_data, tuned_results)

    prediction_tables: list[pd.DataFrame] = []
    for result in tuned_results:
        prediction_tables.append(
            build_ood_prediction_table(
                model_data=model_data,
                estimator=result.best_estimator,
                model_name=result.model_name,
            )
        )

    predictions = pd.concat(prediction_tables, ignore_index=True)
    stress_table = build_ood_stress_table(
        predictions,
        model_data.pair_summary,
        numeric_zero_tolerance=args.numeric_zero_tolerance,
    )
    write_paper_table(stress_table, args.output_table)
    print_stress_test_summary(model_data, control_table, stress_table)


if __name__ == "__main__":
    main()
