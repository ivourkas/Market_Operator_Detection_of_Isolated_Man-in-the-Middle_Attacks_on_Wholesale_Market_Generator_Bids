#!/usr/bin/env python3
"""
Validate normal/attack OPF pairs.

This script reconstructs pair_id values from the raw transposed OPF exports,
before any ML sampling or shuffling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


SENTINEL = -100000
SENTINEL_THRESHOLD = -99999
SEASONS = (1, 2, 3, 4)
META_COLUMNS = ("fval", "ssn", "type_of_attack", "gen_attacked")
PAIR_COLUMNS = (
    "pair_id",
    "system",
    "opf_iteration",
    "season",
    "normal_raw_row_index",
    "attacked_raw_row_index",
    "attack_type",
    "attack_type_name",
    "target_generator",
    "fval_normal",
    "fval_attacked",
)
MISSING_COLUMNS = ("system", "opf_iteration", "season", "normal_raw_row_index")
ERROR_COLUMNS = (
    "system",
    "position",
    "raw_row_index",
    "opf_iteration",
    "message",
    "found_ssn",
    "found_type_of_attack",
    "expected_type_of_attack",
    "found_type_of_attack",
    "expected_gen_attacked",
    "found_gen_attacked",
)
INFEASIBLE_COLUMNS = (
    "system",
    "raw_row_index",
    "fval",
    "ssn",
    "type_of_attack",
    "gen_attacked",
    "fval_sentinel",
    "all_opf_sentinel",
)


@dataclass(frozen=True)
class SystemConfig:
    system: str
    csv_path: Path


SYSTEMS = (
    SystemConfig("N14", Path("data/N14_G5_transposed.csv")),
    SystemConfig("N57", Path("data/N57_G7_transposed.csv")),
    SystemConfig("N300", Path("data/N300_G69_transposed_new.csv")),
)


def _as_int(value: object) -> int:
    """Read MATLAB-exported metadata values such as 1.0 as clean integers."""
    if pd.isna(value):
        raise ValueError("metadata value is NaN")
    return int(float(value))


def _attack_type_name(attack_type: int) -> str:
    return {
        1: "ramp_rate",
        2: "upper_limit",
        3: "lower_limit",
        4: "economic_bid",
    }.get(attack_type, f"unknown_{attack_type}")


def scan_infeasible_rows(csv_path: Path, chunksize: int = 128) -> pd.DataFrame:
    """
    Find infeasible OPF sentinel rows.

    The OPF code writes fval = -100000 and all OPF variables = -100000 when
    linprog returns no feasible solution. We record both tests explicitly.
    """
    rows: list[dict[str, object]] = []

    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        ft_cols = [col for col in chunk.columns if str(col).startswith("ft")]

        fval_bad = chunk["fval"].le(SENTINEL_THRESHOLD)
        if ft_cols:
            all_opf_bad = chunk[ft_cols].le(SENTINEL_THRESHOLD).all(axis=1)
        else:
            all_opf_bad = pd.Series(False, index=chunk.index)

        bad = fval_bad | all_opf_bad
        for raw_idx, row in chunk.loc[bad, META_COLUMNS].iterrows():
            rows.append(
                {
                    "raw_row_index": int(raw_idx),
                    "fval": row["fval"],
                    "ssn": _as_int(row["ssn"]),
                    "type_of_attack": _as_int(row["type_of_attack"]),
                    "gen_attacked": _as_int(row["gen_attacked"]),
                    "fval_sentinel": bool(fval_bad.loc[raw_idx]),
                    "all_opf_sentinel": bool(all_opf_bad.loc[raw_idx]),
                }
            )

    return pd.DataFrame(rows)


def load_clean_metadata(csv_path: Path, infeasible_rows: Iterable[int]) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=list(META_COLUMNS))
    df.insert(0, "raw_row_index", range(len(df)))

    bad = set(int(i) for i in infeasible_rows)
    clean = df.loc[~df["raw_row_index"].isin(bad)].copy().reset_index(drop=True)

    for col in ("ssn", "type_of_attack", "gen_attacked"):
        clean[col] = clean[col].map(_as_int)

    return clean


def parse_pairs(system: str, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """
    Parse blocks produced by spaOPFsmlF.m.

    For each OPF iteration the raw export contains normal seasons 1..4, followed
    by the attacked seasons that remained feasible. If an attacked season is
    missing, that season is excluded from paired-impact analysis and recorded.
    """
    pairs: list[dict[str, object]] = []
    missing: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []

    i = 0
    opf_iteration = 0
    n_rows = len(meta)

    while i < n_rows:
        normal_pos_by_season: dict[int, int] = {}

        for season in SEASONS:
            if i >= n_rows:
                errors.append(
                    {
                        "system": system,
                        "position": i,
                        "opf_iteration": opf_iteration,
                        "message": f"Expected normal season {season}, reached end of file",
                    }
                )
                return pd.DataFrame(pairs), pd.DataFrame(missing), pd.DataFrame(errors), opf_iteration

            row = meta.iloc[i]
            row_season = _as_int(row["ssn"])
            row_attack_type = _as_int(row["type_of_attack"])

            if row_season != season or row_attack_type != 0:
                errors.append(
                    {
                        "system": system,
                        "position": i,
                        "raw_row_index": int(row["raw_row_index"]),
                        "opf_iteration": opf_iteration,
                        "message": f"Expected normal season {season}",
                        "found_ssn": row_season,
                        "found_type_of_attack": row_attack_type,
                    }
                )
                return pd.DataFrame(pairs), pd.DataFrame(missing), pd.DataFrame(errors), opf_iteration

            normal_pos_by_season[season] = i
            i += 1

        attack_pos_by_season: dict[int, int] = {}
        attack_type: int | None = None
        target_generator: int | None = None

        while i < n_rows:
            row = meta.iloc[i]
            row_season = _as_int(row["ssn"])
            row_attack_type = _as_int(row["type_of_attack"])
            row_target = _as_int(row["gen_attacked"])

            if row_attack_type == 0 and row_season == 1:
                break

            if row_attack_type == 0:
                errors.append(
                    {
                        "system": system,
                        "position": i,
                        "raw_row_index": int(row["raw_row_index"]),
                        "opf_iteration": opf_iteration,
                        "message": "Unexpected normal row inside attacked block",
                        "found_ssn": row_season,
                    }
                )
                return pd.DataFrame(pairs), pd.DataFrame(missing), pd.DataFrame(errors), opf_iteration

            if row_season not in SEASONS:
                errors.append(
                    {
                        "system": system,
                        "position": i,
                        "raw_row_index": int(row["raw_row_index"]),
                        "opf_iteration": opf_iteration,
                        "message": "Unexpected attacked season value",
                        "found_ssn": row_season,
                    }
                )
                return pd.DataFrame(pairs), pd.DataFrame(missing), pd.DataFrame(errors), opf_iteration

            if attack_type is None:
                attack_type = row_attack_type
                target_generator = row_target
            elif row_attack_type != attack_type or row_target != target_generator:
                errors.append(
                    {
                        "system": system,
                        "position": i,
                        "raw_row_index": int(row["raw_row_index"]),
                        "opf_iteration": opf_iteration,
                        "message": "Attack metadata changed within one OPF iteration",
                        "expected_type_of_attack": attack_type,
                        "found_type_of_attack": row_attack_type,
                        "expected_gen_attacked": target_generator,
                        "found_gen_attacked": row_target,
                    }
                )
                return pd.DataFrame(pairs), pd.DataFrame(missing), pd.DataFrame(errors), opf_iteration

            if row_season in attack_pos_by_season:
                errors.append(
                    {
                        "system": system,
                        "position": i,
                        "raw_row_index": int(row["raw_row_index"]),
                        "opf_iteration": opf_iteration,
                        "message": "Duplicate attacked season within one OPF iteration",
                        "found_ssn": row_season,
                    }
                )
                return pd.DataFrame(pairs), pd.DataFrame(missing), pd.DataFrame(errors), opf_iteration

            attack_pos_by_season[row_season] = i
            i += 1

        for season in SEASONS:
            normal = meta.iloc[normal_pos_by_season[season]]
            if season not in attack_pos_by_season:
                missing.append(
                    {
                        "system": system,
                        "opf_iteration": opf_iteration,
                        "season": season,
                        "normal_raw_row_index": int(normal["raw_row_index"]),
                    }
                )
                continue

            attacked = meta.iloc[attack_pos_by_season[season]]
            current_attack_type = _as_int(attacked["type_of_attack"])
            current_target = _as_int(attacked["gen_attacked"])

            pairs.append(
                {
                    "pair_id": f"{system}_iter{opf_iteration:05d}_ssn{season}",
                    "system": system,
                    "opf_iteration": opf_iteration,
                    "season": season,
                    "normal_raw_row_index": int(normal["raw_row_index"]),
                    "attacked_raw_row_index": int(attacked["raw_row_index"]),
                    "attack_type": current_attack_type,
                    "attack_type_name": _attack_type_name(current_attack_type),
                    "target_generator": current_target,
                    "fval_normal": float(normal["fval"]),
                    "fval_attacked": float(attacked["fval"]),
                }
            )

        opf_iteration += 1

    return pd.DataFrame(pairs), pd.DataFrame(missing), pd.DataFrame(errors), opf_iteration


def summarize_system(
    config: SystemConfig,
    raw_rows: int,
    clean_meta: pd.DataFrame,
    infeasible: pd.DataFrame,
    pairs: pd.DataFrame,
    missing: pd.DataFrame,
    errors: pd.DataFrame,
    parsed_iterations: int,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "system": config.system,
        "source_csv": str(config.csv_path),
        "raw_rows": raw_rows,
        "rows_removed_infeasible": len(infeasible),
        "fval_sentinel_rows": int(infeasible["fval_sentinel"].sum()) if not infeasible.empty else 0,
        "all_opf_sentinel_rows": int(infeasible["all_opf_sentinel"].sum()) if not infeasible.empty else 0,
        "rows_after_filter": len(clean_meta),
        "normal_rows_after_filter": int((clean_meta["type_of_attack"] == 0).sum()),
        "attack_rows_after_filter": int((clean_meta["type_of_attack"] != 0).sum()),
        "parsed_opf_iterations": parsed_iterations,
        "valid_pairs": len(pairs),
        "missing_attacked_season_pairs": len(missing),
        "parse_errors": len(errors),
    }

    for season in SEASONS:
        summary[f"valid_pairs_season_{season}"] = int((pairs["season"] == season).sum()) if not pairs.empty else 0

    for attack_type in (1, 2, 3, 4):
        summary[f"valid_pairs_attack_type_{attack_type}"] = (
            int((pairs["attack_type"] == attack_type).sum()) if not pairs.empty else 0
        )

    return summary


def main() -> None:
    output_dir = Path("results/revision")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_pairs: list[pd.DataFrame] = []
    all_missing: list[pd.DataFrame] = []
    all_errors: list[pd.DataFrame] = []
    all_infeasible: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []

    for config in SYSTEMS:
        print(f"\n=== {config.system}: validating {config.csv_path} ===")
        with config.csv_path.open("rb") as fh:
            raw_rows = sum(1 for _ in fh) - 1

        infeasible = scan_infeasible_rows(config.csv_path)
        if not infeasible.empty:
            infeasible.insert(0, "system", config.system)
        all_infeasible.append(infeasible)

        clean_meta = load_clean_metadata(config.csv_path, infeasible["raw_row_index"] if not infeasible.empty else [])
        pairs, missing, errors, parsed_iterations = parse_pairs(config.system, clean_meta)

        all_pairs.append(pairs)
        if not missing.empty:
            all_missing.append(missing)
        if not errors.empty:
            all_errors.append(errors)

        summary = summarize_system(
            config=config,
            raw_rows=raw_rows,
            clean_meta=clean_meta,
            infeasible=infeasible,
            pairs=pairs,
            missing=missing,
            errors=errors,
            parsed_iterations=parsed_iterations,
        )
        summaries.append(summary)

        print(
            f"{config.system}: {summary['valid_pairs']} valid pairs, "
            f"{summary['rows_removed_infeasible']} infeasible rows removed, "
            f"{summary['missing_attacked_season_pairs']} missing attacked seasonal pairs, "
            f"{summary['parse_errors']} parse errors"
        )

    pair_index = (
        pd.concat(all_pairs, ignore_index=True)
        if all_pairs
        else pd.DataFrame(columns=PAIR_COLUMNS)
    )
    missing_index = (
        pd.concat(all_missing, ignore_index=True)
        if all_missing
        else pd.DataFrame(columns=MISSING_COLUMNS)
    )
    errors = (
        pd.concat(all_errors, ignore_index=True)
        if all_errors
        else pd.DataFrame(columns=ERROR_COLUMNS)
    )
    infeasible_rows = (
        pd.concat(all_infeasible, ignore_index=True)
        if all_infeasible
        else pd.DataFrame(columns=INFEASIBLE_COLUMNS)
    )
    summary_df = pd.DataFrame(summaries)

    pair_index.to_csv(output_dir / "paired_rows_index.csv", index=False)
    summary_df.to_csv(output_dir / "pair_validation_summary.csv", index=False)
    missing_index.to_csv(output_dir / "missing_attacked_pairs.csv", index=False)
    infeasible_rows.to_csv(output_dir / "infeasible_rows.csv", index=False)
    errors.to_csv(output_dir / "pair_parse_errors.csv", index=False)

    print("\nWrote:")
    print(f"  {output_dir / 'paired_rows_index.csv'}")
    print(f"  {output_dir / 'pair_validation_summary.csv'}")
    print(f"  {output_dir / 'missing_attacked_pairs.csv'}")
    print(f"  {output_dir / 'infeasible_rows.csv'}")
    print(f"  {output_dir / 'pair_parse_errors.csv'}")

    if not errors.empty:
        raise SystemExit("Pair parsing produced errors; inspect pair_parse_errors.csv before continuing.")


if __name__ == "__main__":
    main()
