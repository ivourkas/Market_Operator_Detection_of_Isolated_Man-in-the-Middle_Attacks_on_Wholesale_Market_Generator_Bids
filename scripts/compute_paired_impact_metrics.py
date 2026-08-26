#!/usr/bin/env python3
"""
Compute paired physical and economic impact metrics.

Each output row compares one normal OPF solution with its attacked counterpart
for the same system, OPF iteration, season, and load realization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


N_HOURS = 24


@dataclass(frozen=True)
class SystemConfig:
    system: str
    csv_path: Path
    case_path: Path


@dataclass(frozen=True)
class CaseLayout:
    base_mva: float
    n_buses: int
    features_per_hour: int
    n_flows: int
    active_generator_rows: tuple[int, ...]
    active_generator_buses_internal: tuple[int, ...]
    generator_row_to_bus_internal: dict[int, int]
    generator_row_to_bus_external: dict[int, int]


SYSTEMS = (
    SystemConfig("N14", Path("data/N14_G5_transposed.csv"), Path("Cases/pglib_opf_case14_ieee.m")),
    SystemConfig("N57", Path("data/N57_G7_transposed.csv"), Path("Cases/pglib_opf_case57_ieee.m")),
    SystemConfig("N300", Path("data/N300_G69_transposed_new.csv"), Path("Cases/pglib_opf_case300_ieee.m")),
)


METRIC_DEFINITIONS = (
    {
        "metric": "delta_cost_signed",
        "plain_language": "Direction of the cost change. Positive means the attacked OPF is more expensive than the matching normal OPF; negative means it is cheaper.",
        "formula": "fval_attacked - fval_normal",
    },
    {
        "metric": "delta_cost_abs",
        "plain_language": "Size of the cost change caused by the attack, ignoring whether the cost went up or down.",
        "formula": "abs(fval_attacked - fval_normal)",
    },
    {
        "metric": "delta_cost_pct",
        "plain_language": "Cost change scaled by the normal cost for the same load profile, so systems and seasons are easier to compare.",
        "formula": "delta_cost_abs / abs(fval_normal)",
    },
    {
        "metric": "dispatch_l1",
        "plain_language": "Total fleet redispatch over the 24-hour day. This is the main proof that a one-generator attack can move the wider system.",
        "formula": "sum_g sum_h abs(Pg_attacked(g,h) - Pg_normal(g,h))",
    },
    {
        "metric": "target_dispatch_change",
        "plain_language": "How much the attacked generator itself moved across the day.",
        "formula": "sum_h abs(Pg_attacked(target,h) - Pg_normal(target,h))",
    },
    {
        "metric": "non_target_dispatch_change",
        "plain_language": "How much all other generators moved to compensate. This is the clean system-wide consequence metric.",
        "formula": "sum_{g != target} sum_h abs(Pg_attacked(g,h) - Pg_normal(g,h))",
    },
    {
        "metric": "flow_l1",
        "plain_language": "Total change in network flows over all modeled directed line-flow variables and hours.",
        "formula": "sum_l sum_h abs(flow_attacked(l,h) - flow_normal(l,h))",
    },
    {
        "metric": "ramp_l1",
        "plain_language": "Total change in hour-to-hour generator movement across the fleet. It separates a simple output shift from an attack that changes operating movement over the day.",
        "formula": "sum_g sum_h abs(ramp_attacked(g,h) - ramp_normal(g,h))",
    },
)


def parse_base_mva(case_path: Path) -> float:
    for line in case_path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("mpc.baseMVA"):
            return float(stripped.split("=")[1].strip(" ;"))
    raise ValueError(f"Could not find mpc.baseMVA in {case_path}")


def parse_matpower_matrix(case_path: Path, matrix_name: str) -> np.ndarray:
    """Read a simple MATPOWER matrix block such as mpc.bus or mpc.gen."""
    lines = case_path.read_text().splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line.strip().startswith(f"mpc.{matrix_name} = ["):
            start = idx + 1
            break
    if start is None:
        raise ValueError(f"Could not find mpc.{matrix_name} in {case_path}")

    rows: list[list[float]] = []
    for line in lines[start:]:
        clean = line.split("%", 1)[0].strip()
        if clean.startswith("];"):
            break
        clean = clean.rstrip(";").strip()
        if not clean:
            continue
        rows.append([float(part) for part in clean.split()])

    if not rows:
        raise ValueError(f"mpc.{matrix_name} was empty in {case_path}")
    return np.asarray(rows, dtype=float)


def build_case_layout(config: SystemConfig, ft_column_count: int) -> CaseLayout:
    bus = parse_matpower_matrix(config.case_path, "bus")
    gen = parse_matpower_matrix(config.case_path, "gen")
    base_mva = parse_base_mva(config.case_path)

    if ft_column_count % N_HOURS != 0:
        raise ValueError(f"{config.system}: raw feature count is not divisible by {N_HOURS}")

    features_per_hour = ft_column_count // N_HOURS
    n_buses = len(bus)
    n_flows = features_per_hour - 2 * n_buses

    if n_flows <= 0:
        raise ValueError(f"{config.system}: invalid flow count inferred from feature layout")

    # spaOPFsmlF.m chooses attgen from generator rows whose Pmax is nonzero.
    # The OPF variable itself is bus-level active power, so we map the generator
    # row to the internal bus-power column used in the raw ft* matrix.
    external_bus_to_internal = {int(row[0]): idx + 1 for idx, row in enumerate(bus)}
    generator_row_to_bus_external: dict[int, int] = {}
    generator_row_to_bus_internal: dict[int, int] = {}
    active_generator_rows: list[int] = []
    active_generator_buses_internal: list[int] = []

    for row_idx, row in enumerate(gen, start=1):
        pmax = row[8]
        if pmax <= 0:
            continue

        external_bus = int(row[0])
        if external_bus not in external_bus_to_internal:
            raise ValueError(f"{config.system}: generator row {row_idx} maps to missing bus {external_bus}")

        internal_bus = external_bus_to_internal[external_bus]
        active_generator_rows.append(row_idx)
        active_generator_buses_internal.append(internal_bus)
        generator_row_to_bus_external[row_idx] = external_bus
        generator_row_to_bus_internal[row_idx] = internal_bus

    if not active_generator_rows:
        raise ValueError(f"{config.system}: no active generators found in case file")

    # The OPF formulation is bus-level. If multiple active generator rows share
    # one bus, the bus still appears once in the dispatch sum.
    active_generator_buses_internal = sorted(set(active_generator_buses_internal))

    return CaseLayout(
        base_mva=base_mva,
        n_buses=n_buses,
        features_per_hour=features_per_hour,
        n_flows=n_flows,
        active_generator_rows=tuple(active_generator_rows),
        active_generator_buses_internal=tuple(active_generator_buses_internal),
        generator_row_to_bus_internal=generator_row_to_bus_internal,
        generator_row_to_bus_external=generator_row_to_bus_external,
    )


def bus_power_columns(layout: CaseLayout, bus_internal: int) -> list[int]:
    """Column positions, zero-based within ft*, for one bus over the full day."""
    bus_offset = bus_internal - 1
    return [hour * layout.features_per_hour + bus_offset for hour in range(N_HOURS)]


def all_generator_dispatch_columns(layout: CaseLayout) -> list[int]:
    cols: list[int] = []
    for hour in range(N_HOURS):
        hour_start = hour * layout.features_per_hour
        cols.extend(hour_start + bus - 1 for bus in layout.active_generator_buses_internal)
    return cols


def all_flow_columns(layout: CaseLayout) -> list[int]:
    cols: list[int] = []
    for hour in range(N_HOURS):
        flow_start = hour * layout.features_per_hour + 2 * layout.n_buses
        cols.extend(range(flow_start, flow_start + layout.n_flows))
    return cols


def load_raw_ft_matrix(csv_path: Path) -> tuple[pd.DataFrame, list[str]]:
    columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
    ft_cols = [col for col in columns if col.startswith("ft")]
    df = pd.read_csv(csv_path, usecols=ft_cols)
    return df, ft_cols


def compute_system_metrics(config: SystemConfig, pair_index: pd.DataFrame) -> pd.DataFrame:
    pairs = pair_index[pair_index["system"] == config.system].copy().reset_index(drop=True)
    if pairs.empty:
        return pd.DataFrame()

    raw_ft, ft_cols = load_raw_ft_matrix(config.csv_path)
    layout = build_case_layout(config, len(ft_cols))
    ft = raw_ft.to_numpy(dtype=np.float64)

    normal_idx = pairs["normal_raw_row_index"].to_numpy(dtype=int)
    attacked_idx = pairs["attacked_raw_row_index"].to_numpy(dtype=int)

    gen_cols = all_generator_dispatch_columns(layout)
    flow_cols = all_flow_columns(layout)

    n_pairs = len(pairs)
    n_gens = len(layout.active_generator_buses_internal)

    normal_dispatch = ft[normal_idx[:, None], gen_cols].reshape(n_pairs, N_HOURS, n_gens)
    attacked_dispatch = ft[attacked_idx[:, None], gen_cols].reshape(n_pairs, N_HOURS, n_gens)
    dispatch_delta = np.abs(attacked_dispatch - normal_dispatch)

    # Total fleet movement: if this is large, the single compromised bid caused
    # the OPF to reshuffle generation beyond the attacked unit.
    dispatch_l1 = dispatch_delta.sum(axis=(1, 2))

    target_dispatch_change: list[float] = []
    target_bus_internal: list[int] = []
    target_bus_external: list[int] = []

    for _, pair in pairs.iterrows():
        target_row = int(pair["target_generator"])
        if target_row not in layout.generator_row_to_bus_internal:
            raise ValueError(f"{config.system}: target generator row {target_row} is not active in the case file")

        bus_internal = layout.generator_row_to_bus_internal[target_row]
        bus_external = layout.generator_row_to_bus_external[target_row]
        target_cols = bus_power_columns(layout, bus_internal)

        normal_target = ft[int(pair["normal_raw_row_index"]), target_cols]
        attacked_target = ft[int(pair["attacked_raw_row_index"]), target_cols]

        target_bus_internal.append(bus_internal)
        target_bus_external.append(bus_external)
        target_dispatch_change.append(float(np.abs(attacked_target - normal_target).sum()))

    target_dispatch_change_arr = np.asarray(target_dispatch_change, dtype=np.float64)

    # The non-target term is the important propagation signal: it measures how
    # much the rest of the fleet had to move after one generator was attacked.
    non_target_dispatch_change = np.maximum(dispatch_l1 - target_dispatch_change_arr, 0.0)

    normal_flow = ft[normal_idx[:, None], flow_cols]
    attacked_flow = ft[attacked_idx[:, None], flow_cols]

    # Flow movement tells us whether the attack changed where power travels in
    # the network, even when the cost signal alone looks modest.
    flow_l1 = np.abs(attacked_flow - normal_flow).sum(axis=1)

    normal_ramp = np.diff(normal_dispatch, axis=1)
    attacked_ramp = np.diff(attacked_dispatch, axis=1)

    # Ramping movement shows whether the attack changes hour-to-hour operating
    # movement, not just the total energy dispatched.
    ramp_l1 = np.abs(attacked_ramp - normal_ramp).sum(axis=(1, 2))

    result = pairs.copy()
    result["base_mva"] = layout.base_mva
    result["target_bus_internal"] = target_bus_internal
    result["target_bus_external"] = target_bus_external
    result["n_active_generator_buses"] = n_gens
    result["n_flow_variables"] = layout.n_flows

    result["delta_cost_signed"] = result["fval_attacked"] - result["fval_normal"]
    result["delta_cost_abs"] = result["delta_cost_signed"].abs()
    result["delta_cost_pct"] = result["delta_cost_abs"] / result["fval_normal"].abs().replace(0, np.nan)
    result["dispatch_l1"] = dispatch_l1
    result["target_dispatch_change"] = target_dispatch_change_arr
    result["non_target_dispatch_change"] = non_target_dispatch_change
    result["flow_l1"] = flow_l1
    result["ramp_l1"] = ramp_l1

    return result


def build_system_layout_rows(config: SystemConfig) -> list[dict[str, object]]:
    columns = pd.read_csv(config.csv_path, nrows=0).columns.tolist()
    ft_cols = [col for col in columns if col.startswith("ft")]
    layout = build_case_layout(config, len(ft_cols))

    rows: list[dict[str, object]] = []
    for gen_row in layout.active_generator_rows:
        rows.append(
            {
                "system": config.system,
                "base_mva": layout.base_mva,
                "n_buses": layout.n_buses,
                "features_per_hour": layout.features_per_hour,
                "n_flows": layout.n_flows,
                "active_generator_row": gen_row,
                "active_generator_bus_external": layout.generator_row_to_bus_external[gen_row],
                "active_generator_bus_internal": layout.generator_row_to_bus_internal[gen_row],
            }
        )
    return rows


def summarize_by_attack_type(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "delta_cost_abs",
        "delta_cost_pct",
        "dispatch_l1",
        "target_dispatch_change",
        "non_target_dispatch_change",
        "flow_l1",
        "ramp_l1",
    ]

    rows: list[dict[str, object]] = []
    grouped = metrics.groupby(["system", "attack_type", "attack_type_name"], sort=True)
    for (system, attack_type, attack_type_name), group in grouped:
        row: dict[str, object] = {
            "system": system,
            "attack_type": attack_type,
            "attack_type_name": attack_type_name,
            "n_pairs": len(group),
        }
        for metric in metric_cols:
            # Median is the paper-friendly center here: it says what a typical
            # attack does without letting a few extreme OPF outcomes dominate.
            row[f"median_{metric}"] = group[metric].median()
            row[f"mean_{metric}"] = group[metric].mean()
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_metric_distributions(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "delta_cost_abs",
        "delta_cost_pct",
        "dispatch_l1",
        "target_dispatch_change",
        "non_target_dispatch_change",
        "flow_l1",
        "ramp_l1",
    ]

    rows: list[dict[str, object]] = []
    for (system, season), group in metrics.groupby(["system", "season"], sort=True):
        for metric in metric_cols:
            values = group[metric].dropna()
            rows.append(
                {
                    "system": system,
                    "season": season,
                    "metric": metric,
                    "n_pairs": len(values),
                    "mean": values.mean(),
                    "std": values.std(ddof=1),
                    "median": values.median(),
                    "q90": values.quantile(0.90),
                    "q95": values.quantile(0.95),
                    "max": values.max(),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    output_dir = Path("results/revision")
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_index_path = output_dir / "paired_rows_index.csv"
    pair_index = pd.read_csv(pair_index_path)

    all_metrics: list[pd.DataFrame] = []
    layout_rows: list[dict[str, object]] = []
    for config in SYSTEMS:
        print(f"\n=== {config.system}: computing paired impact metrics ===")
        system_metrics = compute_system_metrics(config, pair_index)
        print(f"{config.system}: computed {len(system_metrics)} paired metric rows")
        all_metrics.append(system_metrics)
        layout_rows.extend(build_system_layout_rows(config))

    metrics = pd.concat(all_metrics, ignore_index=True)
    by_attack = summarize_by_attack_type(metrics)
    distribution_summary = summarize_metric_distributions(metrics)
    metric_definitions = pd.DataFrame(METRIC_DEFINITIONS)
    system_layout = pd.DataFrame(layout_rows)

    metrics.to_csv(output_dir / "paired_impact_metrics.csv", index=False)
    by_attack.to_csv(output_dir / "impact_by_system_attack_type.csv", index=False)
    distribution_summary.to_csv(output_dir / "impact_metric_summary.csv", index=False)
    metric_definitions.to_csv(output_dir / "impact_metric_definitions.csv", index=False)
    system_layout.to_csv(output_dir / "impact_system_layout.csv", index=False)

    print("\nWrote:")
    print(f"  {output_dir / 'paired_impact_metrics.csv'}")
    print(f"  {output_dir / 'impact_by_system_attack_type.csv'}")
    print(f"  {output_dir / 'impact_metric_summary.csv'}")
    print(f"  {output_dir / 'impact_metric_definitions.csv'}")
    print(f"  {output_dir / 'impact_system_layout.csv'}")


if __name__ == "__main__":
    main()
