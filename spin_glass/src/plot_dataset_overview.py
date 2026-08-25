"""Recreate Supplemental Fig. S1 with the canonical independent test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np


SPIN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = SPIN_ROOT / "data" / "train_40000"
DEFAULT_TEST = SPIN_ROOT / "data" / "test_2000"
DEFAULT_OUTPUT = SPIN_ROOT / "figures" / "FigS1_Dataset_Overview_PRL.pdf"


def training_statistics(directory: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    lengths: list[int] = []
    dot_parts: dict[str, list[np.ndarray]] = {"AA": [], "BB": [], "AB": []}
    chunks = sorted(directory.glob("chunk_*.npz"))
    if len(chunks) != 40:
        raise RuntimeError(f"expected 40 training chunks, found {len(chunks)}")
    for path in chunks:
        with np.load(path, allow_pickle=True) as data:
            for sequence_value, snapshots_value in zip(data["sequences"], data["spins"]):
                sequence = str(sequence_value)
                snapshots = np.asarray(snapshots_value, dtype=np.float64)
                if snapshots.shape != (10, len(sequence), 3):
                    raise RuntimeError(f"unexpected training shape in {path}: {snapshots.shape}")
                lengths.append(len(sequence))
                dots = np.sum(snapshots[:, :-1] * snapshots[:, 1:], axis=-1)
                pair_types = np.asarray(
                    ["".join(sorted(sequence[index : index + 2])) for index in range(len(sequence) - 1)]
                )
                for pair_type in dot_parts:
                    dot_parts[pair_type].append(dots[:, pair_types == pair_type].reshape(-1))
    if len(lengths) != 40_000:
        raise RuntimeError(f"expected 40,000 training sequences, found {len(lengths)}")
    return np.asarray(lengths), {
        pair_type: np.concatenate(parts) for pair_type, parts in dot_parts.items()
    }


def test_statistics(directory: Path) -> tuple[np.ndarray, list[np.ndarray]]:
    force_parts: list[np.ndarray] = []
    first_snapshots: list[np.ndarray] = []
    sequences = 0
    for path in sorted(directory.glob("chunk_*.npz")):
        with np.load(path, allow_pickle=True) as data:
            required = {"sequences", "spins", "tangent_forces"}
            if not required.issubset(data.files):
                raise RuntimeError(f"missing canonical arrays in {path}")
            for sequence_value, snapshots_value, forces_value in zip(
                data["sequences"], data["spins"], data["tangent_forces"]
            ):
                sequence = str(sequence_value)
                snapshots = np.asarray(snapshots_value, dtype=np.float64)
                forces = np.asarray(forces_value, dtype=np.float64)
                if snapshots.shape != (5, len(sequence), 3) or forces.shape != snapshots.shape:
                    raise RuntimeError(f"unexpected canonical test shape in {path}")
                first_snapshots.append(snapshots[0])
                force_parts.append(np.linalg.vector_norm(forces, axis=-1).reshape(-1))
                sequences += 1
    if sequences != 2_000:
        raise RuntimeError(f"expected 2,000 canonical test sequences, found {sequences}")
    return np.concatenate(force_parts), first_snapshots


def draw(
    lengths: np.ndarray,
    dot_products: dict[str, np.ndarray],
    force_magnitudes: np.ndarray,
    first_snapshots: list[np.ndarray],
    output: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 13,
            "axes.labelsize": 14,
            "axes.titlesize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(12, 9))
    grid = GridSpec(2, 2, figure=figure, wspace=0.25, hspace=0.35)

    ax1 = figure.add_subplot(grid[0, 0])
    ax1.hist(
        lengths,
        bins=np.arange(19.5, 51.5, 1),
        color="#4c72b0",
        edgecolor="black",
        linewidth=0.8,
        alpha=0.85,
    )
    ax1.set_title(r"$\mathbf{(a)}$ Sequence Length Distribution", loc="left")
    ax1.set_xlabel("Chain Length $N$")
    ax1.set_ylabel("Frequency")
    ax1.grid(axis="y", linestyle=":", alpha=0.6)

    ax2 = figure.add_subplot(grid[0, 1])
    colors = {"AA": "#d62728", "BB": "#1f77b4", "AB": "#2ca02c"}
    labels = {"AA": "A-A (Rigid)", "BB": "B-B (Flexible)", "AB": "A-B (Moderate)"}
    for pair_type in ("AA", "BB", "AB"):
        ax2.hist(
            dot_products[pair_type],
            bins=80,
            range=(-0.7, 1.0),
            alpha=0.6,
            histtype="stepfilled",
            edgecolor="black",
            linewidth=0.5,
            label=f"Bond: {labels[pair_type]}",
            color=colors[pair_type],
            density=True,
        )
    ax2.set_title(r"$\mathbf{(b)}$ Local Frustration by Bond Type", loc="left")
    ax2.set_xlabel(r"Nearest-Neighbor Dot Product ($\mathbf{S}_i \cdot \mathbf{S}_{i+1}$)")
    ax2.set_ylabel("Probability Density")
    ax2.legend(loc="upper left", framealpha=0.9)
    ax2.axvline(1.0, color="black", linestyle="--", alpha=0.4, linewidth=1.5)
    ax2.grid(True, linestyle=":", alpha=0.6)

    ax3 = figure.add_subplot(grid[1, 0])
    ax3.hist(
        force_magnitudes,
        bins=80,
        range=(0, np.percentile(force_magnitudes, 99.9)),
        color="#9467bd",
        edgecolor="black",
        linewidth=0.5,
        alpha=0.85,
    )
    ax3.set_title(r"$\mathbf{(c)}$ Tangent Force Magnitudes", loc="left")
    ax3.set_xlabel(r"Force Magnitude $|\mathbf{F}_{\mathrm{tan}}|$")
    ax3.set_ylabel("Frequency")
    ax3.grid(axis="y", linestyle=":", alpha=0.6)

    ax4_base = figure.add_subplot(grid[1, 1])
    ax4_base.set_axis_off()
    ax4_base.set_title(r"$\mathbf{(d)}$ Emergent 3D Conformations", loc="left", pad=10)
    inner = grid[1, 1].subgridspec(2, 2, wspace=0.0, hspace=0.0)
    selected = np.linspace(0, len(first_snapshots) - 1, 4).astype(int)
    chain_colors = ("#ff7f0e", "#2ca02c", "#d62728", "#9467bd")
    for panel, (index, color) in enumerate(zip(selected, chain_colors)):
        axis = figure.add_subplot(inner[panel // 2, panel % 2], projection="3d")
        coordinates = np.vstack(([0.0, 0.0, 0.0], np.cumsum(first_snapshots[index], axis=0)))
        coordinates -= coordinates.mean(axis=0, keepdims=True)
        axis.plot(
            coordinates[:, 0],
            coordinates[:, 1],
            coordinates[:, 2],
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=4.0,
            markeredgecolor="black",
            markeredgewidth=0.3,
            alpha=0.9,
        )
        midpoint = 0.5 * (coordinates.max(axis=0) + coordinates.min(axis=0))
        margin = 1.05 * 0.5 * np.ptp(coordinates, axis=0).max()
        axis.set_xlim(midpoint[0] - margin, midpoint[0] + margin)
        axis.set_ylim(midpoint[1] - margin, midpoint[1] + margin)
        axis.set_zlim(midpoint[2] - margin, midpoint[2] + margin)
        axis.view_init(elev=20, azim=45 + panel * 25)
        # The canonical independent examples have different extents from the
        # legacy display examples; keep enough margin that every chain remains
        # inside its mini-panel.
        axis.set_box_aspect(None, zoom=1.35)
        axis.set_axis_off()

    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    lengths, dot_products = training_statistics(args.train_dir.resolve())
    force_magnitudes, first_snapshots = test_statistics(args.test_dir.resolve())
    output = args.output.resolve()
    draw(lengths, dot_products, force_magnitudes, first_snapshots, output)
    summary = {
        "training_sequences": int(len(lengths)),
        "canonical_test_sequences": int(len(first_snapshots)),
        "training_snapshots_per_sequence": 10,
        "canonical_test_snapshots_per_sequence": 5,
        "nearest_neighbor_observations": {
            key: int(len(value)) for key, value in dot_products.items()
        },
        "force_observations": int(len(force_magnitudes)),
        "force_mean": float(np.mean(force_magnitudes)),
        "force_std": float(np.std(force_magnitudes)),
        "force_q50": float(np.quantile(force_magnitudes, 0.5)),
        "force_q999": float(np.quantile(force_magnitudes, 0.999)),
    }
    summary_path = args.summary.resolve() if args.summary else output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

