"""Recreate the direct-W locality plot and the 3x2 W--J panel figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


SPIN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = SPIN_ROOT / "results" / "direct_w_evaluation"
PAIR_NAMES = ("AA", "BB", "AB", "AA", "BB", "AB")
DISTANCES = np.asarray([1, 1, 1, 2, 2, 2])
COLORS = {"AA": "#2F6F9F", "BB": "#C46A38", "AB": "#4F8B59"}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
        }
    )


def locality(metrics: dict, output: Path) -> None:
    rows = metrics["distance_statistics"]
    x = np.asarray([row["distance"] for row in rows])
    y = np.asarray([row["mean_abs_W"] for row in rows])
    q25 = np.asarray([row["q25_abs_W"] for row in rows])
    q75 = np.asarray([row["q75_abs_W"] for row in rows])
    error = np.vstack((y - q25, q75 - y))
    blue = "#2F6F9F"
    red = "#B3473D"
    fig, ax = plt.subplots(figsize=(3.45, 3.10), constrained_layout=True)
    profile = ax.errorbar(
        x,
        y,
        yerr=error,
        fmt="-o",
        color=blue,
        ecolor=blue,
        lw=1.45,
        elinewidth=0.85,
        ms=3.8,
        capsize=2.2,
        capthick=0.75,
        label=r"mean $|W_{ij}|$ (IQR)",
        zorder=4,
    )
    cutoff = ax.axvline(
        2.5,
        color=red,
        lw=1.15,
        ls=(0, (4, 2.5)),
        label=r"bare cutoff ($r=2$)",
        zorder=3,
    )
    ax.set_xlabel(r"Sequence distance $r=|i-j|$")
    ax.set_ylabel(r"Mean absolute coefficient $|W_{ij}|$")
    ax.set_xlim(0.5, 15.5)
    ax.set_ylim(0.0, 11.2)
    ax.set_xticks(np.arange(1, 16, 2))
    ax.grid(color="#B8B8B8", alpha=0.42, lw=0.45, ls=(0, (1.5, 2.2)))
    ax.legend(
        handles=[cutoff, profile],
        loc="upper right",
        fontsize=7.0,
        frameon=True,
        facecolor="white",
        edgecolor="#C8C8C8",
        framealpha=0.94,
        borderpad=0.35,
        handlelength=2.3,
    )
    ratio = 100.0 * float(metrics["far_to_support_absolute_mass_ratio"])
    ax.text(
        0.965,
        0.055,
        rf"$\sum_{{r\geq3}}|W|/\sum_{{r\leq2}}|W|={ratio:.2f}\%$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.8,
        color="#202020",
        bbox={"boxstyle": "square,pad=0.14", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        zorder=8,
    )
    inset = ax.inset_axes([0.43, 0.27, 0.53, 0.43])
    inset.errorbar(
        x,
        y,
        yerr=error,
        fmt="-o",
        color=blue,
        ecolor=blue,
        lw=0.95,
        elinewidth=0.65,
        ms=2.5,
        capsize=1.6,
        capthick=0.55,
    )
    inset.axvline(2.5, color=red, lw=0.9, ls=(0, (4, 2.5)))
    inset.set_yscale("log")
    inset.set_xlim(0.5, 15.5)
    inset.set_ylim(2.0e-3, 1.6e1)
    inset.set_xticks([1, 5, 10, 15])
    inset.tick_params(axis="both", which="major", labelsize=6.2, width=0.55)
    inset.grid(color="#B8B8B8", alpha=0.35, lw=0.35, ls=(0, (1.5, 2.2)))
    inset.set_title("log scale", fontsize=6.8, pad=2)
    fig.savefig(output / "Fig2_Weight_Decay_PRL.pdf", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(output / "Fig2_Weight_Decay_PRL.png", dpi=420, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def wj_panels(observations: dict[str, np.ndarray], metrics: dict, output: Path) -> None:
    rng = np.random.default_rng(20260801)
    w = observations["W_support"]
    group = observations["group"]
    spin_dot = observations["spin_dot"]
    a_star = float(metrics["a_star"])
    j_values = np.asarray([row["J"] for row in metrics["classes"]])
    order = np.asarray([[0, 3], [1, 4], [2, 5]])
    fig, axes = plt.subplots(3, 2, figsize=(3.35, 5.15), sharex=True, sharey="col", constrained_layout=True)
    for panel, (row, col) in enumerate(np.ndindex(axes.shape)):
        index = int(order[row, col])
        ax = axes[row, col]
        selected = np.flatnonzero(group == index)
        if len(selected) > 9000:
            selected = rng.choice(selected, 9000, replace=False)
        pair = PAIR_NAMES[index]
        ax.scatter(spin_dot[selected], w[selected], s=2.7, alpha=0.075, color=COLORS[pair], edgecolors="none", rasterized=True)
        ax.axhline(
            a_star * j_values[index],
            color="#202020",
            lw=1.0,
            ls=(0, (4, 2.6)),
            zorder=5,
        )
        ax.set_title(rf"{pair}, $r={DISTANCES[index]}$  ($\mathcal{{J}}={j_values[index]:.2f}$)", pad=3)
        target = a_star * j_values[index]
        annotation_y = 0.055 if col == 0 else 0.955
        ax.text(
            0.035,
            annotation_y,
            rf"$a_\star\mathcal{{J}}={target:.2f}$",
            transform=ax.transAxes,
            ha="left",
            va="bottom" if col == 0 else "top",
            fontsize=6.9,
            color="#202020",
            bbox={"boxstyle": "square,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.72},
        )
        ax.text(0.965, 0.955, f"({chr(ord('a') + panel)})", transform=ax.transAxes, ha="right", va="top", fontsize=7.8)
        ax.set_xlim(-1.02, 1.02)
        ax.grid(color="#B8B8B8", alpha=0.34, lw=0.45)
    for ax in axes[:, 0]:
        ax.set_ylim(0.0, 13.0)
    for ax in axes[:, 1]:
        ax.set_ylim(-7.0, 0.0)
    fig.supxlabel(r"Local configuration $\mathbf{S}_i\!\cdot\!\mathbf{S}_j$", fontsize=9)
    fig.supylabel(r"Learned coefficient $W_{ij}$", fontsize=9)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#777777", alpha=0.45, markersize=4, label=r"OOD $W_{ij}$"),
        Line2D([0], [0], color="#202020", lw=1.0, ls=(0, (4, 2.6)), label=r"class-balanced $a_\star\mathcal{J}$"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=2, frameon=False, handlelength=2.2)
    fig.savefig(output / "Fig3_Epoch26_Direct_WJ_SixPanel_PRL.pdf", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(output / "Fig3_Epoch26_Direct_WJ_SixPanel_PRL.png", dpi=420, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output-dir", type=Path, default=SPIN_ROOT / "figures")
    args = parser.parse_args()
    result = args.result_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((result / "metrics.json").read_text(encoding="utf-8"))
    with np.load(result / "observations.npz", allow_pickle=False) as data:
        observations = {key: np.asarray(data[key]) for key in data.files}
    style()
    locality(metrics, output)
    wj_panels(observations, metrics, output)


if __name__ == "__main__":
    main()
