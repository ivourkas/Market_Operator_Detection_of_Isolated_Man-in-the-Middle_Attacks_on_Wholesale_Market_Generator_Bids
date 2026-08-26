#!/usr/bin/env python3
"""
Audit whether scenario creation preserves raw row identity.

This script does not train models or engineer features. It recreates the
balanced 80/20 scenario construction used by the existing experiments and
checks whether each sampled row can be traced back to the raw transposed CSV.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "results" / "revision" / "ml_audit"
PAIR_CLASSIFICATION_PATH = PROJECT_ROOT / "results" / "revision" / "paired_impact_classification.csv"


@dataclass(frozen=True)
class SystemAuditConfig:
    system: str
    csv_path: Path
    scenario_family: str
    expected_train_attacks: int
    expected_train_normals: int
    expected_test_attacks: int
    expected_test_normals: int


SYSTEMS = (
    SystemAuditConfig(
        system="N14",
        csv_path=PROJECT_ROOT / "data" / "N14_G5_transposed.csv",
        scenario_family="n14_n57_balanced_80_20",
        expected_train_attacks=800,
        expected_train_normals=800,
        expected_test_attacks=200,
        expected_test_normals=200,
    ),
    SystemAuditConfig(
        system="N57",
        csv_path=PROJECT_ROOT / "data" / "N57_G7_transposed.csv",
        scenario_family="n14_n57_balanced_80_20",
        expected_train_attacks=800,
        expected_train_normals=800,
        expected_test_attacks=200,
        expected_test_normals=200,
    ),
    SystemAuditConfig(
        system="N300",
        csv_path=PROJECT_ROOT / "data" / "N300_G69_transposed_new.csv",
        scenario_family="n300_balanced_80_20",
        expected_train_attacks=768,
        expected_train_normals=768,
        expected_test_attacks=192,
        expected_test_normals=192,
    ),
)


def load_raw_with_identity(csv_path: Path) -> pd.DataFrame:
    """
    Load the raw transposed OPF export and attach its original row number.

    The inserted identifier is metadata. It is not a feature and must be excluded
    from any downstream model input.
    """

    df = pd.read_csv(
        csv_path,
        dtype={"ssn": "Int64", "type_of_attack": "Int64", "gen_attacked": "Int64"},
    )
    if "raw_row_index" in df.columns:
        raise ValueError(f"{csv_path} already contains raw_row_index")
    df = df.assign(raw_row_index=range(len(df))).copy()
    if "attack" not in df.columns:
        df = df.assign(attack=(df["type_of_attack"] != 0).astype(int))
    return df


def build_scenario_with_existing_logic(config: SystemAuditConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Recreate the existing balanced 80/20 split while preserving row identity.

    N14 and N57 follow the same scenario path. N300 follows its own path, matching
    the existing code.
    """

    df = load_raw_with_identity(config.csv_path)
    attack_data = df[df["attack"] == 1].copy()
    normal_data = df[df["attack"] == 0].copy()

    with contextlib.redirect_stdout(io.StringIO()):
        if config.scenario_family == "n14_n57_balanced_80_20":
            train_df, test_df = build_n14_n57_scenario(attack_data, normal_data)
        elif config.scenario_family == "n300_balanced_80_20":
            train_df, test_df = build_n300_scenario(attack_data, normal_data)
        else:
            raise ValueError(f"Unsupported scenario family: {config.scenario_family}")

    return train_df, test_df


def split_total_across_groups(total: int, groups: list[int]) -> dict[int, int]:
    """Split an integer total deterministically, giving remainders to earlier groups."""

    if not groups:
        return {}
    base, remainder = divmod(int(total), len(groups))
    return {group: base + (1 if idx < remainder else 0) for idx, group in enumerate(groups)}


def stratified_sample_n14_n57(
    attack_data: pd.DataFrame,
    normal_data: pd.DataFrame,
    attacks_needed_per_ssn: dict[int, int],
    normals_needed_per_ssn: dict[int, int],
    original_ssns: list[int],
) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
    """Sample N14/N57 attacks by season while keeping attack types balanced."""

    attack_df = attack_data.copy().reset_index(drop=True)
    normal_df = normal_data.copy().reset_index(drop=True)

    selected_attacks_by_ssn: dict[int, pd.DataFrame] = {}
    selected_normals_by_ssn: dict[int, pd.DataFrame] = {}
    total_attacks_needed_overall = int(sum(attacks_needed_per_ssn.get(ssn, 0) for ssn in original_ssns))

    for ssn in original_ssns:
        ssn_attacks = attack_df[attack_df["ssn"] == ssn].copy().reset_index(drop=True)
        ssn_normals = normal_df[normal_df["ssn"] == ssn].copy().reset_index(drop=True)

        total_attacks_needed = int(attacks_needed_per_ssn.get(ssn, 0))
        total_normals_needed = int(normals_needed_per_ssn.get(ssn, 0))

        selected_attacks_by_ssn[ssn] = pd.DataFrame(columns=ssn_attacks.columns)
        selected_normals_by_ssn[ssn] = pd.DataFrame(columns=ssn_normals.columns)

        if len(ssn_attacks) > 0 and total_attacks_needed > 0:
            ssn_attack_types = ssn_attacks["type_of_attack"].value_counts().sort_index()
            if len(ssn_attacks) >= total_attacks_needed:
                n_types = len(ssn_attack_types)
                base_per_type = total_attacks_needed // n_types
                remainder = total_attacks_needed % n_types

                sampled_parts: list[pd.DataFrame] = []
                actual_sampled = 0
                for idx, attack_type in enumerate(sorted(ssn_attack_types.index)):
                    target_count = base_per_type + (1 if idx < remainder else 0)
                    pool = ssn_attacks[ssn_attacks["type_of_attack"] == attack_type]
                    to_sample = min(target_count, len(pool))
                    if to_sample > 0:
                        sampled = pool.sample(n=to_sample, random_state=42)
                        sampled_parts.append(sampled)
                        actual_sampled += to_sample

                if actual_sampled < total_attacks_needed:
                    remaining_needed = total_attacks_needed - actual_sampled
                    already_ids = (
                        pd.concat(sampled_parts)["raw_row_index"].values if sampled_parts else []
                    )
                    remaining_pool = ssn_attacks[~ssn_attacks["raw_row_index"].isin(already_ids)]
                    if len(remaining_pool) >= remaining_needed:
                        sampled_parts.append(remaining_pool.sample(n=remaining_needed, random_state=42))

                if sampled_parts:
                    sampled_df = pd.concat(sampled_parts, ignore_index=True)
                    if len(sampled_df) > total_attacks_needed:
                        sampled_df = sampled_df.sample(n=total_attacks_needed, random_state=42)
                    selected_attacks_by_ssn[ssn] = sampled_df
            else:
                selected_attacks_by_ssn[ssn] = ssn_attacks.copy()

        if total_normals_needed > 0:
            if len(ssn_normals) >= total_normals_needed:
                selected_normals_by_ssn[ssn] = ssn_normals.sample(n=total_normals_needed, random_state=42)
            else:
                selected_normals_by_ssn[ssn] = ssn_normals.copy()

    all_selected = (
        pd.concat([df for df in selected_attacks_by_ssn.values() if not df.empty], ignore_index=True)
        if any(not df.empty for df in selected_attacks_by_ssn.values())
        else pd.DataFrame()
    )
    need_more_total = total_attacks_needed_overall - len(all_selected)

    if need_more_total > 0:
        already_ids = set(all_selected["raw_row_index"].values) if not all_selected.empty else set()
        remaining_pool = attack_df[~attack_df["raw_row_index"].isin(list(already_ids))].copy().reset_index(drop=True)
        if len(remaining_pool) >= need_more_total:
            extras = remaining_pool.sample(n=need_more_total, random_state=42)
            for _, row in extras.iterrows():
                ssn = int(row["ssn"])
                row_df = pd.DataFrame([row])
                if selected_attacks_by_ssn.get(ssn) is None or selected_attacks_by_ssn[ssn].empty:
                    selected_attacks_by_ssn[ssn] = row_df
                else:
                    selected_attacks_by_ssn[ssn] = pd.concat(
                        [selected_attacks_by_ssn[ssn], row_df], ignore_index=True
                    )

    return selected_attacks_by_ssn, selected_normals_by_ssn


def distribute_years_n14_n57(
    attacks_by_ssn: dict[int, pd.DataFrame],
    normals_by_ssn: dict[int, pd.DataFrame],
    all_years: list[int],
    attacks_targets_by_year_ssn: dict[int, dict[int, int]],
    normals_targets_by_year_ssn: dict[int, dict[int, int]],
    original_ssns: list[int],
) -> dict[int, dict[int, dict[str, object]]]:
    """Distribute sampled N14/N57 rows into year-season slots."""

    distributed_data = {
        year: {ssn: {"attacks": [], "normals": pd.DataFrame()} for ssn in original_ssns}
        for year in all_years
    }

    for ssn in original_ssns:
        ssn_attacks_df = attacks_by_ssn.get(ssn, pd.DataFrame()).copy().reset_index(drop=True)
        ssn_normals_df = normals_by_ssn.get(ssn, pd.DataFrame()).copy().reset_index(drop=True)

        if not ssn_attacks_df.empty:
            type_queues = {
                attack_type: group.sample(frac=1, random_state=42).reset_index(drop=True)
                for attack_type, group in ssn_attacks_df.groupby("type_of_attack")
            }
            type_indices = {attack_type: 0 for attack_type in type_queues}
        else:
            type_queues = {}
            type_indices = {}

        normal_idx = 0
        for year in all_years:
            attacks_needed = int(attacks_targets_by_year_ssn.get(year, {}).get(ssn, 0))
            normals_needed = int(normals_targets_by_year_ssn.get(year, {}).get(ssn, 0))

            year_attacks_list: list[pd.DataFrame] = []
            attacks_assigned = 0
            while attacks_assigned < attacks_needed:
                made_progress = False
                for attack_type in sorted(type_queues.keys()):
                    if attacks_assigned >= attacks_needed:
                        break
                    if type_indices[attack_type] < len(type_queues[attack_type]):
                        row = type_queues[attack_type].iloc[[type_indices[attack_type]]]
                        year_attacks_list.append(row)
                        type_indices[attack_type] += 1
                        attacks_assigned += 1
                        made_progress = True
                if not made_progress:
                    break

            distributed_data[year][ssn]["attacks"] = year_attacks_list

            if normals_needed > 0 and normal_idx < len(ssn_normals_df):
                year_normals = ssn_normals_df.iloc[normal_idx : normal_idx + normals_needed]
                distributed_data[year][ssn]["normals"] = year_normals
                normal_idx += normals_needed
            else:
                distributed_data[year][ssn]["normals"] = pd.DataFrame()

    return distributed_data


def stratified_sample_n300(
    attack_data: pd.DataFrame,
    normal_data: pd.DataFrame,
    n_attacks_per_ssn_per_year: int,
    n_normals_per_ssn_per_year: int,
    original_ssns: list[int],
    n_years: int,
) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
    """Sample N300 rows using the existing proportional attack-type logic."""

    selected_attacks_by_ssn: dict[int, pd.DataFrame] = {}
    selected_normals_by_ssn: dict[int, pd.DataFrame] = {}

    for ssn in original_ssns:
        ssn_attacks = attack_data[attack_data["ssn"] == ssn].copy()
        ssn_normals = normal_data[normal_data["ssn"] == ssn].copy()

        total_attacks_needed = n_attacks_per_ssn_per_year * n_years
        total_normals_needed = n_normals_per_ssn_per_year * n_years

        if len(ssn_attacks) > 0:
            ssn_attack_types = ssn_attacks["type_of_attack"].value_counts().sort_index()
            if len(ssn_attacks) >= total_attacks_needed:
                sampled_attacks: list[pd.DataFrame] = []
                for attack_type, count in ssn_attack_types.items():
                    proportion = count / len(ssn_attacks)
                    n_to_sample = max(1, int(total_attacks_needed * proportion))
                    type_attacks = ssn_attacks[ssn_attacks["type_of_attack"] == attack_type]
                    sampled = (
                        type_attacks.sample(n=n_to_sample, random_state=42)
                        if len(type_attacks) >= n_to_sample
                        else type_attacks
                    )
                    sampled_attacks.append(sampled)
                selected_attacks_by_ssn[ssn] = pd.concat(sampled_attacks, ignore_index=True)
            else:
                selected_attacks_by_ssn[ssn] = ssn_attacks
        else:
            selected_attacks_by_ssn[ssn] = pd.DataFrame()

        if len(ssn_normals) >= total_normals_needed:
            selected_normals_by_ssn[ssn] = ssn_normals.sample(n=total_normals_needed, random_state=42)
        else:
            selected_normals_by_ssn[ssn] = ssn_normals

    return selected_attacks_by_ssn, selected_normals_by_ssn


def distribute_years_n300(
    attacks_by_ssn: dict[int, pd.DataFrame],
    normals_by_ssn: dict[int, pd.DataFrame],
    all_years: list[int],
    attacks_per_ssn_per_train_year: int,
    attacks_per_ssn_per_test_year: int,
    normals_per_ssn_per_train_year: int,
    normals_per_ssn_per_test_year: int,
    train_years: list[int],
    original_ssns: list[int],
) -> dict[int, dict[int, dict[str, object]]]:
    """Distribute sampled N300 rows into year-season slots."""

    distributed_data = {
        year: {ssn: {"attacks": [], "normals": []} for ssn in original_ssns}
        for year in all_years
    }

    for ssn in original_ssns:
        ssn_attacks = attacks_by_ssn[ssn]
        ssn_normals = normals_by_ssn[ssn]

        if len(ssn_attacks) > 0:
            type_queues = {
                attack_type: group.sample(frac=1, random_state=42).reset_index(drop=True)
                for attack_type, group in ssn_attacks.groupby("type_of_attack")
            }
            type_indices = {attack_type: 0 for attack_type in type_queues}

            for year in all_years:
                attacks_needed = (
                    attacks_per_ssn_per_train_year
                    if year in train_years
                    else attacks_per_ssn_per_test_year
                )

                year_attacks: list[pd.DataFrame] = []
                attacks_assigned = 0
                while attacks_assigned < attacks_needed:
                    made_progress = False
                    for attack_type in sorted(type_queues.keys()):
                        if attacks_assigned >= attacks_needed:
                            break
                        if type_indices[attack_type] < len(type_queues[attack_type]):
                            row = type_queues[attack_type].iloc[[type_indices[attack_type]]]
                            year_attacks.append(row)
                            type_indices[attack_type] += 1
                            attacks_assigned += 1
                            made_progress = True
                    if not made_progress:
                        break
                distributed_data[year][ssn]["attacks"] = year_attacks

        if len(ssn_normals) > 0:
            normal_indices = 0
            ssn_normals_shuffled = ssn_normals.sample(frac=1, random_state=42).reset_index(drop=True)
            for year in all_years:
                normals_needed = (
                    normals_per_ssn_per_train_year
                    if year in train_years
                    else normals_per_ssn_per_test_year
                )
                if normals_needed > 0 and normal_indices < len(ssn_normals_shuffled):
                    year_normals = ssn_normals_shuffled.iloc[normal_indices : normal_indices + normals_needed]
                    distributed_data[year][ssn]["normals"] = year_normals
                    normal_indices += normals_needed
                else:
                    distributed_data[year][ssn]["normals"] = pd.DataFrame()

    return distributed_data


def build_n14_n57_scenario(attack_data: pd.DataFrame, normal_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Recreate the N14/N57 balanced 80/20 scenario.

    This path uses per-year and per-season target dictionaries. That is different
    from the N300 path and is preserved intentionally.
    """

    train_years = [1, 2, 3, 4]
    test_years = [5]
    all_years = train_years + test_years
    days_per_ssn_per_year = 100

    train_attacks_total = 800
    train_normals_total = 800
    test_attacks_total = 200
    test_normals_total = 200

    original_ssns = sorted(
        set(attack_data["ssn"].dropna().astype(int).unique())
        | set(normal_data["ssn"].dropna().astype(int).unique())
    )
    num_ssns = len(original_ssns)

    attacks_train_year = split_total_across_groups(train_attacks_total, train_years)
    normals_train_year = split_total_across_groups(train_normals_total, train_years)
    attacks_test_year = split_total_across_groups(test_attacks_total, test_years)
    normals_test_year = split_total_across_groups(test_normals_total, test_years)

    attacks_by_year = {**attacks_train_year, **attacks_test_year}
    normals_by_year = {**normals_train_year, **normals_test_year}

    attacks_targets = {
        year: split_total_across_groups(attacks_by_year[year], original_ssns)
        for year in all_years
    }
    normals_targets = {
        year: split_total_across_groups(normals_by_year[year], original_ssns)
        for year in all_years
    }

    total_attacks_per_ssn = {
        ssn: sum(attacks_targets[year][ssn] for year in all_years)
        for ssn in original_ssns
    }
    total_normals_per_ssn = {
        ssn: sum(normals_targets[year][ssn] for year in all_years)
        for ssn in original_ssns
    }

    attacks_by_ssn, normals_by_ssn = stratified_sample_n14_n57(
        attack_data,
        normal_data,
        total_attacks_per_ssn,
        total_normals_per_ssn,
        original_ssns,
    )
    distributed_data = distribute_years_n14_n57(
        attacks_by_ssn,
        normals_by_ssn,
        all_years,
        attacks_targets,
        normals_targets,
        original_ssns,
    )

    final_data: list[pd.DataFrame] = []
    for year in all_years:
        for ssn in original_ssns:
            year_attacks = distributed_data[year][ssn]["attacks"]
            year_normals = distributed_data[year][ssn]["normals"]

            ssn_year_attacks = (
                pd.concat(year_attacks, ignore_index=True)
                if year_attacks
                else pd.DataFrame(columns=attack_data.columns)
            )
            ssn_year_normals = (
                year_normals.copy()
                if isinstance(year_normals, pd.DataFrame) and len(year_normals) > 0
                else pd.DataFrame(columns=normal_data.columns)
            )

            ssn_year_data = pd.concat([ssn_year_attacks, ssn_year_normals], ignore_index=True, sort=False)
            if ssn_year_data.empty:
                continue

            ssn_year_data = ssn_year_data.sample(
                frac=1,
                random_state=42 + int(year) + int(ssn),
            ).reset_index(drop=True)

            ssn_year_data["year"] = year
            ssn_year_data["repetition"] = year
            ssn_year_data["original_ssn"] = ssn
            ssn_year_data["ssn"] = int(year) * 10 + int(ssn)
            ssn_year_data["day_in_season"] = range(1, len(ssn_year_data) + 1)
            ssn_year_data["session_day"] = ssn_year_data["day_in_season"]
            base_idx = ((int(year) - 1) * days_per_ssn_per_year * num_ssns) + (
                (int(ssn) - 1) * days_per_ssn_per_year
            )
            ssn_year_data["temporal_index"] = base_idx + ssn_year_data["day_in_season"]
            final_data.append(ssn_year_data)

    final_df = pd.concat(final_data, ignore_index=True, sort=False)
    final_df = final_df.sort_values("temporal_index").reset_index(drop=True)
    train_df = final_df[final_df["repetition"].isin(train_years)].copy().reset_index(drop=True)
    test_df = final_df[final_df["repetition"].isin(test_years)].copy().reset_index(drop=True)
    return train_df, test_df


def build_n300_scenario(attack_data: pd.DataFrame, normal_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Recreate the N300 balanced 80/20 scenario.

    This path uses scalar per-season-per-year targets and a 96-day season block,
    matching the existing N300 behavior.
    """

    train_years = [1, 2, 3, 4]
    test_years = [5]
    all_years = train_years + test_years
    days_per_ssn_per_year = 96

    train_attacks_total = 768
    train_normals_total = 768
    test_attacks_total = 192
    test_normals_total = 192

    original_ssns = sorted(
        set(attack_data["ssn"].dropna().astype(int).unique())
        | set(normal_data["ssn"].dropna().astype(int).unique())
    )
    num_ssns = len(original_ssns)

    attacks_per_ssn_per_train_year = (train_attacks_total // len(train_years)) // num_ssns
    normals_per_ssn_per_train_year = (train_normals_total // len(train_years)) // num_ssns
    attacks_per_ssn_per_test_year = (test_attacks_total // len(test_years)) // num_ssns
    normals_per_ssn_per_test_year = (test_normals_total // len(test_years)) // num_ssns

    total_attacks_per_ssn = (
        attacks_per_ssn_per_train_year * len(train_years)
        + attacks_per_ssn_per_test_year * len(test_years)
    )
    total_normals_per_ssn = (
        normals_per_ssn_per_train_year * len(train_years)
        + normals_per_ssn_per_test_year * len(test_years)
    )

    attacks_by_ssn, normals_by_ssn = stratified_sample_n300(
        attack_data,
        normal_data,
        total_attacks_per_ssn,
        total_normals_per_ssn,
        original_ssns,
        1,
    )
    distributed_data = distribute_years_n300(
        attacks_by_ssn,
        normals_by_ssn,
        all_years,
        attacks_per_ssn_per_train_year,
        attacks_per_ssn_per_test_year,
        normals_per_ssn_per_train_year,
        normals_per_ssn_per_test_year,
        train_years,
        original_ssns,
    )

    final_data: list[pd.DataFrame] = []
    for year in all_years:
        for ssn in original_ssns:
            year_attacks = distributed_data[year][ssn]["attacks"]
            year_normals = distributed_data[year][ssn]["normals"]

            ssn_year_attacks = (
                pd.concat(year_attacks, ignore_index=True)
                if year_attacks
                else pd.DataFrame(columns=attack_data.columns)
            )
            ssn_year_normals = (
                year_normals.copy()
                if isinstance(year_normals, pd.DataFrame) and len(year_normals) > 0
                else pd.DataFrame(columns=normal_data.columns)
            )

            ssn_year_data = pd.concat([ssn_year_attacks, ssn_year_normals], ignore_index=True)
            if ssn_year_data.empty:
                continue

            ssn_year_data = ssn_year_data.sample(
                frac=1,
                random_state=42 + int(year) + int(ssn),
            ).reset_index(drop=True)

            ssn_year_data["year"] = year
            ssn_year_data["repetition"] = year
            ssn_year_data["original_ssn"] = ssn
            ssn_year_data["ssn"] = int(year) * 10 + int(ssn)
            ssn_year_data["day_in_season"] = range(1, len(ssn_year_data) + 1)
            ssn_year_data["session_day"] = ssn_year_data["day_in_season"]
            base_idx = ((int(year) - 1) * days_per_ssn_per_year * num_ssns) + (
                (int(ssn) - 1) * days_per_ssn_per_year
            )
            ssn_year_data["temporal_index"] = base_idx + ssn_year_data["day_in_season"]
            final_data.append(ssn_year_data)

    final_df = pd.concat(final_data, ignore_index=True)
    final_df = final_df.sort_values("temporal_index").reset_index(drop=True)
    train_df = final_df[final_df["repetition"].isin(train_years)].copy().reset_index(drop=True)
    test_df = final_df[final_df["repetition"].isin(test_years)].copy().reset_index(drop=True)
    return train_df, test_df


def summarize_split(config: SystemAuditConfig, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, object]:
    """Create one row describing whether the scenario split matches expectations."""

    train_attacks = int(train_df["attack"].sum())
    train_normals = int((train_df["attack"] == 0).sum())
    test_attacks = int(test_df["attack"].sum())
    test_normals = int((test_df["attack"] == 0).sum())

    scenario = pd.concat(
        [train_df.assign(split="train"), test_df.assign(split="test")],
        ignore_index=True,
        sort=False,
    )

    return {
        "system": config.system,
        "scenario_family": config.scenario_family,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_attacks": train_attacks,
        "expected_train_attacks": config.expected_train_attacks,
        "train_normals": train_normals,
        "expected_train_normals": config.expected_train_normals,
        "test_attacks": test_attacks,
        "expected_test_attacks": config.expected_test_attacks,
        "test_normals": test_normals,
        "expected_test_normals": config.expected_test_normals,
        "counts_match": bool(
            train_attacks == config.expected_train_attacks
            and train_normals == config.expected_train_normals
            and test_attacks == config.expected_test_attacks
            and test_normals == config.expected_test_normals
        ),
        "raw_row_index_nulls": int(scenario["raw_row_index"].isna().sum()),
        "raw_row_index_duplicates": int(scenario["raw_row_index"].duplicated().sum()),
        "unique_raw_rows": int(scenario["raw_row_index"].nunique()),
    }


def summarize_attack_types(config: SystemAuditConfig, train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Count attack types by split and original season."""

    scenario = pd.concat(
        [train_df.assign(split="train"), test_df.assign(split="test")],
        ignore_index=True,
        sort=False,
    )
    attacks = scenario[scenario["attack"] == 1].copy()
    grouped = (
        attacks.groupby(["split", "original_ssn", "type_of_attack"], dropna=False)
        .size()
        .reset_index(name="n_rows")
    )
    grouped.insert(0, "system", config.system)
    grouped.insert(1, "scenario_family", config.scenario_family)
    return grouped


def summarize_pair_merge(config: SystemAuditConfig, train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Check how many sampled attacked rows can be joined to impact classifications."""

    paired = pd.read_csv(PAIR_CLASSIFICATION_PATH)
    paired_system = paired[paired["system"] == config.system].copy()

    rows: list[dict[str, object]] = []
    for split_name, split_df in (("train", train_df), ("test", test_df)):
        attacked = split_df[split_df["attack"] == 1].copy()
        merged = attacked.merge(
            paired_system[["attacked_raw_row_index", "pair_id", "impact_pattern"]],
            left_on="raw_row_index",
            right_on="attacked_raw_row_index",
            how="left",
            validate="many_to_one",
        )
        rows.append(
            {
                "system": config.system,
                "scenario_family": config.scenario_family,
                "split": split_name,
                "attacked_rows": len(attacked),
                "matched_to_pair_classification": int(merged["pair_id"].notna().sum()),
                "unmatched_attacked_rows": int(merged["pair_id"].isna().sum()),
                "match_rate_pct": 100.0 * float(merged["pair_id"].notna().mean()) if len(merged) else 0.0,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    split_rows: list[dict[str, object]] = []
    attack_type_tables: list[pd.DataFrame] = []
    merge_tables: list[pd.DataFrame] = []

    for config in SYSTEMS:
        print(f"Auditing {config.system} scenario identity...")
        train_df, test_df = build_scenario_with_existing_logic(config)
        split_rows.append(summarize_split(config, train_df, test_df))
        attack_type_tables.append(summarize_attack_types(config, train_df, test_df))
        merge_tables.append(summarize_pair_merge(config, train_df, test_df))

    split_summary = pd.DataFrame(split_rows)
    attack_type_summary = pd.concat(attack_type_tables, ignore_index=True)
    pair_merge_summary = pd.concat(merge_tables, ignore_index=True)

    split_summary.to_csv(OUTPUT_DIR / "scenario_identity_summary.csv", index=False)
    attack_type_summary.to_csv(OUTPUT_DIR / "scenario_attack_type_distribution.csv", index=False)
    pair_merge_summary.to_csv(OUTPUT_DIR / "scenario_pair_merge_summary.csv", index=False)

    print("\nWrote:")
    print(f"  {OUTPUT_DIR / 'scenario_identity_summary.csv'}")
    print(f"  {OUTPUT_DIR / 'scenario_attack_type_distribution.csv'}")
    print(f"  {OUTPUT_DIR / 'scenario_pair_merge_summary.csv'}")


if __name__ == "__main__":
    main()
