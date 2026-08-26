#!/usr/bin/env python3
"""
Plot normalized attack impacts in cost-dispatch quadrants.

Each point is one validated normal-attack OPF pair. Cost and dispatch are
normalized by the corresponding system-season Q90 thresholds, so the threshold
lines x=1 and y=1 have the same interpretation across systems and seasons.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path("results/revision/paired_impact_classification.csv")
OUTPUT_DIR = Path("results/revision/figures")
MPL_CACHE_DIR = Path("results/revision/.matplotlib_cache")

MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CACHE_DIR / "xdg"))

import matplotlib.pyplot as plt

SYSTEM_ORDER = ["N14", "N57", "N300"]
ATTACK_ORDER = ["ramp_rate", "upper_limit", "lower_limit", "economic_bid"]
ATTACK_LABELS = {
    "ramp_rate": "Ramp rate",
    "upper_limit": "Upper limit",
    "lower_limit": "Lower limit",
    "economic_bid": "Economic bid",
}
ATTACK_MARKERS = {
    "ramp_rate": "o",
    "upper_limit": "s",
    "lower_limit": "^",
    "economic_bid": "D",
}
ATTACK_COLORS = {
    "ramp_rate": "#4C78A8",
    "upper_limit": "#F58518",
    "lower_limit": "#54A24B",
    "economic_bid": "#B279A2",
}

ZERO_TOLERANCE = 1e-12
MAX_AXIS_VALUE = 4.0
FIGURE_BASENAME = "figure_impact_quadrants"
BUBBLE_FIGURE_BASENAME = "figure_impact_quadrant_summary"
HEATMAP_FIGURE_BASENAME = "figure_impact_quadrant_heatmap"
PATTERN_CENTERS = {
    "weak_or_nonbinding_attack": (0.42, 0.42),
    "economic_impact_attack": (1.62, 0.42),
    "hidden_redispatch_attack": (0.42, 1.62),
    "critical_attack": (1.62, 1.62),
}
PATTERN_LABELS = {
    "weak_or_nonbinding_attack": "Weak/\nnon-binding",
    "economic_impact_attack": "Economic\nimpact",
    "hidden_redispatch_attack": "Hidden\nredispatch",
    "critical_attack": "Critical",
}


def prepare_plot_data(classified: pd.DataFrame) -> pd.DataFrame:
    """Normalize cost and dispatch by the system-season Q90 thresholds."""

    required = [
        "system",
        "attack_type_name",
        "delta_cost_abs",
        "dispatch_l1",
        "cost_q90",
        "dispatch_q90",
    ]
    missing = [col for col in required if col not in classified.columns]
    if missing:
        raise ValueError(f"{INPUT_PATH} is missing required columns: {missing}")

    plot_data = classified.copy()
    plot_data["cost_q90_safe"] = plot_data["cost_q90"].where(plot_data["cost_q90"] > ZERO_TOLERANCE, np.nan)
    plot_data["dispatch_q90_safe"] = plot_data["dispatch_q90"].where(
        plot_data["dispatch_q90"] > ZERO_TOLERANCE, np.nan
    )
    plot_data["normalized_cost_impact"] = plot_data["delta_cost_abs"] / plot_data["cost_q90_safe"]
    plot_data["normalized_dispatch_impact"] = plot_data["dispatch_l1"] / plot_data["dispatch_q90_safe"]

    plot_data["normalized_cost_impact"] = plot_data["normalized_cost_impact"].replace([np.inf, -np.inf], np.nan)
    plot_data["normalized_dispatch_impact"] = plot_data["normalized_dispatch_impact"].replace(
        [np.inf, -np.inf], np.nan
    )
    return plot_data.dropna(subset=["normalized_cost_impact", "normalized_dispatch_impact"])


def cap_for_display(values: pd.Series) -> pd.Series:
    """
    Cap extreme values for readable plotting.

    The raw values remain in the CSV outputs. The cap only prevents a few very
    large tail events from flattening the rest of the figure.
    """

    return values.clip(upper=MAX_AXIS_VALUE)


def add_quadrant_guides(ax: plt.Axes) -> None:
    """Draw threshold lines and unobtrusive quadrant labels."""

    ax.axvline(1.0, color="0.2", linewidth=0.8, linestyle="--")
    ax.axhline(1.0, color="0.2", linewidth=0.8, linestyle="--")

    label_style = {
        "fontsize": 6.8,
        "color": "0.25",
        "ha": "left",
        "va": "top",
    }
    ax.text(0.08, 0.92, "Hidden redispatch", transform=ax.transAxes, **label_style)
    ax.text(0.62, 0.92, "Critical", transform=ax.transAxes, **label_style)
    ax.text(0.08, 0.24, "Weak/non-binding", transform=ax.transAxes, **label_style)
    ax.text(0.62, 0.24, "Economic impact", transform=ax.transAxes, **label_style)


def plot_quadrants(plot_data: pd.DataFrame) -> plt.Figure:
    """Create one IEEE-friendly three-panel quadrant figure."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.35), sharex=True, sharey=True)

    for ax, system in zip(axes, SYSTEM_ORDER):
        system_data = plot_data[plot_data["system"] == system]
        add_quadrant_guides(ax)

        for attack in ATTACK_ORDER:
            subset = system_data[system_data["attack_type_name"] == attack]
            if subset.empty:
                continue
            ax.scatter(
                cap_for_display(subset["normalized_cost_impact"]),
                cap_for_display(subset["normalized_dispatch_impact"]),
                s=12,
                marker=ATTACK_MARKERS[attack],
                color=ATTACK_COLORS[attack],
                alpha=0.42,
                linewidths=0,
                label=ATTACK_LABELS[attack],
            )

        ax.set_title(system)
        ax.set_xlim(-0.05, MAX_AXIS_VALUE)
        ax.set_ylim(-0.05, MAX_AXIS_VALUE)
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_yticks([0, 1, 2, 3, 4])
        ax.grid(color="0.9", linewidth=0.5)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Dispatch impact / system-season Q90")
    axes[1].set_xlabel("Cost impact / system-season Q90")

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
        handletextpad=0.35,
        columnspacing=1.1,
    )
    fig.text(
        0.995,
        0.01,
        f"Values above {MAX_AXIS_VALUE:g} are clipped for display.",
        ha="right",
        va="bottom",
        fontsize=6.8,
        color="0.35",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.92), w_pad=0.8)
    return fig


def plot_quadrant_summary(classified: pd.DataFrame) -> plt.Figure:
    """
    Summarize quadrant membership with attack-type bubbles.

    This figure sacrifices row-level scatter detail for readability: bubble area
    is proportional to the percentage of an attack type falling in a quadrant.
    """

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.35), sharex=True, sharey=True)
    attack_offsets = {
        "ramp_rate": (-0.13, 0.12),
        "upper_limit": (0.13, 0.12),
        "lower_limit": (-0.13, -0.12),
        "economic_bid": (0.13, -0.12),
    }

    for ax, system in zip(axes, SYSTEM_ORDER):
        system_data = classified[classified["system"] == system]
        ax.axvline(1.0, color="0.2", linewidth=0.8, linestyle="--")
        ax.axhline(1.0, color="0.2", linewidth=0.8, linestyle="--")

        for pattern, (x_center, y_center) in PATTERN_CENTERS.items():
            ax.text(
                x_center,
                y_center + 0.33,
                PATTERN_LABELS[pattern],
                ha="center",
                va="center",
                fontsize=6.7,
                color="0.25",
            )

        for attack in ATTACK_ORDER:
            attack_data = system_data[system_data["attack_type_name"] == attack]
            if attack_data.empty:
                continue

            shares = attack_data["impact_pattern"].value_counts(normalize=True) * 100.0
            dx, dy = attack_offsets[attack]
            for pattern, pct in shares.items():
                if pattern not in PATTERN_CENTERS or pct <= 0:
                    continue
                x_center, y_center = PATTERN_CENTERS[pattern]
                ax.scatter(
                    x_center + dx,
                    y_center + dy,
                    s=10.0 + 8.0 * pct,
                    marker=ATTACK_MARKERS[attack],
                    color=ATTACK_COLORS[attack],
                    alpha=0.58,
                    linewidths=0.25,
                    edgecolors="white",
                )

        ax.set_title(system)
        ax.set_xlim(0.0, 2.05)
        ax.set_ylim(0.0, 2.05)
        ax.set_xticks([0.5, 1.5])
        ax.set_yticks([0.5, 1.5])
        ax.set_xticklabels(["Low cost", "High cost"])
        ax.set_yticklabels(["Low dispatch", "High dispatch"])
        ax.grid(color="0.92", linewidth=0.5)
        ax.set_axisbelow(True)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker=ATTACK_MARKERS[attack],
            color="none",
            markerfacecolor=ATTACK_COLORS[attack],
            markeredgecolor="white",
            markersize=6,
            label=ATTACK_LABELS[attack],
        )
        for attack in ATTACK_ORDER
    ]
    size_handles = [
        plt.scatter([], [], s=10.0 + 8.0 * pct, color="0.65", alpha=0.58, edgecolors="white", label=f"{pct}%")
        for pct in (10, 30, 60)
    ]

    legend1 = fig.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.47, 1.04),
        handletextpad=0.35,
        columnspacing=1.1,
    )
    fig.add_artist(legend1)
    fig.legend(
        size_handles,
        [handle.get_label() for handle in size_handles],
        loc="upper right",
        ncol=3,
        title="Share",
        frameon=False,
        bbox_to_anchor=(1.0, 1.055),
        handletextpad=0.4,
        columnspacing=0.7,
    )

    fig.tight_layout(rect=(0, 0.02, 1, 0.90), w_pad=0.8)
    return fig


def plot_quadrant_heatmap(classified: pd.DataFrame) -> plt.Figure:
    """
    Show quadrant shares as a compact heatmap.

    This is the most space-efficient view for the manuscript because it avoids
    overplotting while preserving the four-class interpretation.
    """

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    pattern_order = [
        "weak_or_nonbinding_attack",
        "economic_impact_attack",
        "hidden_redispatch_attack",
        "critical_attack",
    ]
    pattern_labels = [
        "Weak/non-binding\nlow C, low D",
        "Economic impact\nhigh C, low D",
        "Hidden redispatch\nlow C, high D",
        "Critical\nhigh C, high D",
    ]

    rows: list[dict[str, float | str]] = []
    row_labels: list[str] = []

    for system in SYSTEM_ORDER:
        for attack in ATTACK_ORDER:
            group = classified[
                (classified["system"] == system) & (classified["attack_type_name"] == attack)
            ]
            if group.empty:
                continue

            shares = group["impact_pattern"].value_counts(normalize=True) * 100.0
            rows.append({pattern: float(shares.get(pattern, 0.0)) for pattern in pattern_order})
            row_labels.append(f"{system} - {ATTACK_LABELS[attack]}")

    matrix = pd.DataFrame(rows, columns=pattern_order).to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.16, 3.75))
    image = ax.imshow(matrix, cmap="Greys", vmin=0.0, vmax=100.0, aspect="auto")

    ax.set_xticks(range(len(pattern_labels)))
    ax.set_xticklabels(pattern_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            text_color = "white" if value >= 55 else "0.15"
            ax.text(
                col_idx,
                row_idx,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=7.2,
                color=text_color,
            )

    for boundary in [3.5, 7.5]:
        ax.axhline(boundary, color="white", linewidth=2.0)

    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("Share (%)")
    colorbar.ax.tick_params(labelsize=7)

    fig.tight_layout(rect=(0, 0.01, 1, 1))
    return fig


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    classified = pd.read_csv(INPUT_PATH)
    plot_data = prepare_plot_data(classified)
    fig = plot_quadrants(plot_data)
    bubble_fig = plot_quadrant_summary(classified)
    heatmap_fig = plot_quadrant_heatmap(classified)

    pdf_path = OUTPUT_DIR / f"{FIGURE_BASENAME}.pdf"
    png_path = OUTPUT_DIR / f"{FIGURE_BASENAME}.png"
    data_path = OUTPUT_DIR / f"{FIGURE_BASENAME}_data.csv"
    bubble_pdf_path = OUTPUT_DIR / f"{BUBBLE_FIGURE_BASENAME}.pdf"
    bubble_png_path = OUTPUT_DIR / f"{BUBBLE_FIGURE_BASENAME}.png"
    heatmap_pdf_path = OUTPUT_DIR / f"{HEATMAP_FIGURE_BASENAME}.pdf"
    heatmap_png_path = OUTPUT_DIR / f"{HEATMAP_FIGURE_BASENAME}.png"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=350, bbox_inches="tight")
    bubble_fig.savefig(bubble_pdf_path, bbox_inches="tight")
    bubble_fig.savefig(bubble_png_path, dpi=350, bbox_inches="tight")
    heatmap_fig.savefig(heatmap_pdf_path, bbox_inches="tight")
    heatmap_fig.savefig(heatmap_png_path, dpi=350, bbox_inches="tight")
    plot_data.to_csv(data_path, index=False)
    plt.close(fig)
    plt.close(bubble_fig)
    plt.close(heatmap_fig)

    print("Wrote:")
    print(f"  {pdf_path}")
    print(f"  {png_path}")
    print(f"  {data_path}")
    print(f"  {bubble_pdf_path}")
    print(f"  {bubble_png_path}")
    print(f"  {heatmap_pdf_path}")
    print(f"  {heatmap_png_path}")


if __name__ == "__main__":
    main()
