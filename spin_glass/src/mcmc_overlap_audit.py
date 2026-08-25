"""Aligned-overlap decorrelation audit used in Supplementary Sec. S1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def load_sequences(directory: Path) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for path in sorted(directory.glob("chunk_*.npz")):
        with np.load(path, allow_pickle=True) as data:
            for sequence, spins in zip(data["sequences"], data["spins"]):
                output[str(sequence)] = np.asarray(spins, dtype=np.float64)
    if not output:
        raise FileNotFoundError(f"no chunk_*.npz files in {directory}")
    return output


def aligned_overlap(first: np.ndarray, second: np.ndarray, device: torch.device) -> np.ndarray:
    """Maximum mean spin overlap after one common O(3) alignment."""
    covariance = np.einsum("ani,bnj->abij", first, second)
    matrix = torch.as_tensor(covariance, dtype=torch.float64, device=device)
    singular = torch.linalg.svdvals(matrix)
    return singular.sum(dim=-1).cpu().numpy() / float(first.shape[1])


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, replicates: int) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    draws = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 200):
        stop = min(replicates, start + 200)
        index = rng.integers(0, values.size, size=(stop - start, values.size))
        draws[start:stop] = values[index].mean(axis=1)
    return {
        "point": float(values.mean()),
        "q025": float(np.quantile(draws, 0.025)),
        "q50": float(np.quantile(draws, 0.5)),
        "q975": float(np.quantile(draws, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True, help="40,000-sequence training REMC archive")
    parser.add_argument("--independent-dir", type=Path, required=True, help="separately initialized run for 2,000 of the same sequences")
    parser.add_argument("--output", type=Path, default=Path("aligned_overlap_audit.json"))
    parser.add_argument("--bootstrap-replicates", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=26081701)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    training = load_sequences(args.training_dir.resolve())
    independent = load_sequences(args.independent_dir.resolve())
    shared = sorted(set(training).intersection(independent))
    if len(training) != 40_000 or len(independent) != 2_000 or len(shared) != 2_000:
        raise RuntimeError("expected 40,000 training sequences and 2,000 shared independently resampled sequences")

    device = torch.device(args.device)
    within = {lag: [] for lag in range(1, 10)}
    baseline = []
    for number, sequence in enumerate(shared, start=1):
        first = training[sequence]
        second = independent[sequence]
        if first.shape[0] != 10 or second.shape[0] != 5:
            raise RuntimeError("the audit requires ten within-run and five independent-run snapshots")
        q_within = aligned_overlap(first, first, device)
        for lag in range(1, 10):
            within[lag].append(float(np.diag(q_within, k=lag).mean()))
        baseline.append(float(aligned_overlap(first, second, device).mean()))
        if number == 1 or number % 200 == 0:
            print(f"processed {number}/{len(shared)} sequences", flush=True)

    rng = np.random.default_rng(args.seed)
    baseline_array = np.asarray(baseline)
    report = {
        "schema_version": 1,
        "definition": "maximum mean spin overlap after one common O(3) alignment",
        "saved_snapshot_spacing_sweeps": 100,
        "sequences": len(shared),
        "independent_run_baseline": bootstrap_mean(baseline_array, rng, args.bootstrap_replicates),
        "within_run_by_saved_lag": {},
    }
    for lag, values_list in within.items():
        values = np.asarray(values_list)
        report["within_run_by_saved_lag"][str(lag)] = {
            "sweeps": 100 * lag,
            "mean_overlap": bootstrap_mean(values, rng, args.bootstrap_replicates),
            "paired_difference_from_independent_run": bootstrap_mean(values - baseline_array, rng, args.bootstrap_replicates),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
