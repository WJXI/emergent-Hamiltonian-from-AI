"""Directly read the frozen pair matrix W on the independent 2,000-sequence set.

This is the release implementation of the spin-glass experiment reported in
the paper.  It evaluates the epoch-26 model at t=1e-4, without OLS score
reconstruction or any post-training residual model.  The single display scale
``a_star`` gives each of the six nonzero coupling classes equal weight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from model import MiniAF3ScoreModel


SPIN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = SPIN_ROOT / "data" / "test_2000"
DEFAULT_CHECKPOINT = SPIN_ROOT / "checkpoints" / "epoch_26.pt"
DEFAULT_OUTPUT = SPIN_ROOT / "results" / "direct_w_evaluation"
EXPECTED_CHECKPOINT_SHA256 = "880adc11f48253e85fdbd6205f8dafad5f1cadc877e02675337b4a21932ea048"

J_VALUES = np.asarray([1.0, 0.5, 0.8, -0.45, -0.25, -0.35], dtype=np.float64)
CLASS_LABELS = ("AA_r1", "BB_r1", "AB_r1", "AA_r2", "BB_r2", "AB_r2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_and_audit_data(data_dir: Path) -> tuple[list[tuple[int, int, str, np.ndarray]], dict[str, Any]]:
    manifest_path = data_dir / "CANONICAL_TESTSET_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["release_manifest_sha256"] = sha256(manifest_path)
    expected = {
        "canonical_status": "formal independent sequence-OOD test set",
        "num_sequences": 2000,
        "snapshots_per_sequence": 5,
        "training_sequence_overlap": 0,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"canonical manifest mismatch for {key}: {manifest.get(key)!r}")
    for item in manifest["chunks"]:
        path = data_dir / item["name"]
        if sha256(path) != item["sha256"]:
            raise RuntimeError(f"test chunk hash mismatch: {path}")

    records: list[tuple[int, int, str, np.ndarray]] = []
    sequences: list[str] = []
    for path in sorted(data_dir.glob("chunk_*.npz")):
        with np.load(path, allow_pickle=True) as data:
            for sequence, snapshots in zip(data["sequences"], data["spins"]):
                sequence_text = str(sequence)
                sequence_index = len(sequences)
                sequences.append(sequence_text)
                if len(snapshots) != 5:
                    raise RuntimeError("every sequence must have five saved configurations")
                for snapshot_index, spins in enumerate(snapshots):
                    records.append(
                        (
                            sequence_index,
                            snapshot_index,
                            sequence_text,
                            np.asarray(spins, dtype=np.float32),
                        )
                    )
    if len(sequences) != 2000 or len(set(sequences)) != 2000 or len(records) != 10000:
        raise RuntimeError("unexpected canonical test-set cardinality")
    return records, manifest


def load_state(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location=device, weights_only=False)
    return payload["model_state_dict"] if "model_state_dict" in payload else payload


def coupling_class(first: int, second: int, distance: int) -> int:
    if first == 0 and second == 0:
        pair_type = 0
    elif first == 1 and second == 1:
        pair_type = 1
    else:
        pair_type = 2
    return pair_type if distance == 1 else pair_type + 3


def extract(
    records: list[tuple[int, int, str, np.ndarray]],
    checkpoint: Path,
    device: torch.device,
    batch_size: int,
    t_eval: float,
) -> dict[str, np.ndarray]:
    model = MiniAF3ScoreModel(c_s=64, c_z=32, num_blocks=4).to(device)
    model.load_state_dict(load_state(checkpoint, device))
    model.eval()
    captured: dict[str, torch.Tensor] = {}

    def capture_head(_module, _inputs, output):
        captured["raw"] = output

    handle = model.weight_head.register_forward_hook(capture_head)
    support_w: list[np.ndarray] = []
    support_group: list[np.ndarray] = []
    support_dot: list[np.ndarray] = []
    support_sequence: list[np.ndarray] = []
    all_w: list[np.ndarray] = []
    all_distance: list[np.ndarray] = []
    try:
        with torch.inference_mode():
            for start in range(0, len(records), batch_size):
                batch = records[start : start + batch_size]
                size = len(batch)
                sequence = torch.zeros((size, 50), dtype=torch.long, device=device)
                spins = torch.zeros((size, 50, 3), dtype=torch.float32, device=device)
                mask = torch.zeros((size, 50), dtype=torch.bool, device=device)
                lengths: list[int] = []
                for row, (_seq, _snap, sequence_text, values) in enumerate(batch):
                    length = len(sequence_text)
                    lengths.append(length)
                    sequence[row, :length] = torch.as_tensor(
                        [0 if token == "A" else 1 for token in sequence_text],
                        dtype=torch.long,
                        device=device,
                    )
                    spins[row, :length] = torch.as_tensor(values, device=device)
                    mask[row, :length] = True
                time = torch.full((size, 1), t_eval, dtype=torch.float32, device=device)
                model(sequence, spins, time, mask)
                raw = captured.pop("raw").squeeze(-1)
                weights = 0.5 * (raw + raw.transpose(1, 2))
                weights = weights.cpu().numpy()
                sequence_np = sequence.cpu().numpy()
                spins_np = spins.cpu().numpy()

                for row, (sequence_index, _snapshot, _text, _values) in enumerate(batch):
                    length = lengths[row]
                    ii, jj = np.triu_indices(length, k=1)
                    distance = jj - ii
                    values = weights[row, ii, jj].astype(np.float32)
                    all_w.append(values)
                    all_distance.append(distance.astype(np.int8))

                    keep = distance <= 2
                    ii_s, jj_s, distance_s = ii[keep], jj[keep], distance[keep]
                    groups = np.fromiter(
                        (
                            coupling_class(
                                int(sequence_np[row, first]),
                                int(sequence_np[row, second]),
                                int(separation),
                            )
                            for first, second, separation in zip(ii_s, jj_s, distance_s)
                        ),
                        dtype=np.int8,
                        count=len(ii_s),
                    )
                    support_w.append(weights[row, ii_s, jj_s].astype(np.float32))
                    support_group.append(groups)
                    support_dot.append(
                        np.einsum("ij,ij->i", spins_np[row, ii_s], spins_np[row, jj_s]).astype(np.float32)
                    )
                    support_sequence.append(np.full(len(ii_s), sequence_index, dtype=np.int16))
                if start == 0 or (start // batch_size) % 20 == 0:
                    print(f"processed {min(start + batch_size, len(records))}/{len(records)}", flush=True)
    finally:
        handle.remove()

    groups = np.concatenate(support_group)
    return {
        "W_support": np.concatenate(support_w),
        "group": groups,
        "J": J_VALUES[groups].astype(np.float32),
        "spin_dot": np.concatenate(support_dot),
        "sequence_index": np.concatenate(support_sequence),
        "W_all": np.concatenate(all_w),
        "distance": np.concatenate(all_distance),
    }


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64) - float(np.mean(x))
    y = np.asarray(y, dtype=np.float64) - float(np.mean(y))
    return float(np.sum(x * y) / np.sqrt(np.sum(x * x) * np.sum(y * y)))


def summarize(data: dict[str, np.ndarray], checkpoint: Path, manifest: dict[str, Any], t_eval: float) -> dict[str, Any]:
    w = data["W_support"].astype(np.float64)
    j = data["J"].astype(np.float64)
    group = data["group"]
    group_means = np.asarray([np.mean(w[group == index]) for index in range(6)], dtype=np.float64)
    class_ratios = group_means / J_VALUES
    a_star = float(np.mean(class_ratios))

    configuration_residual = 0.0
    configuration_magnitude = 0.0
    for sequence_index in np.unique(data["sequence_index"]):
        selected = np.flatnonzero(data["sequence_index"] == sequence_index)
        values = w[selected].reshape(5, len(selected) // 5)
        conditional_mean = np.mean(values, axis=0, keepdims=True)
        configuration_residual += float(np.sum(np.square(values - conditional_mean)))
        configuration_magnitude += float(np.sum(np.square(values)))

    w_all = data["W_all"].astype(np.float64)
    distance = data["distance"]
    support_abs = np.abs(w_all[distance <= 2])
    far_abs = np.abs(w_all[distance >= 3])
    mass_ratio = float(np.sum(far_abs) / np.sum(support_abs))
    distance_statistics = []
    for separation in range(1, 16):
        values = np.abs(w_all[distance == separation])
        q25, q50, q75 = np.percentile(values, [25, 50, 75])
        distance_statistics.append(
            {
                "distance": separation,
                "count": int(len(values)),
                "mean_abs_W": float(np.mean(values)),
                "q25_abs_W": float(q25),
                "median_abs_W": float(q50),
                "q75_abs_W": float(q75),
            }
        )

    classes = []
    for index, label in enumerate(CLASS_LABELS):
        values = w[group == index]
        classes.append(
            {
                "label": label,
                "J": float(J_VALUES[index]),
                "count": int(len(values)),
                "W_mean": float(np.mean(values)),
                "W_std": float(np.std(values)),
                "a_star_J": float(a_star * J_VALUES[index]),
            }
        )

    return {
        "schema_version": 1,
        "experiment": "direct frozen W readout",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "canonical_manifest_sha256": manifest["release_manifest_sha256"],
        "test_sequences": int(manifest["num_sequences"]),
        "test_configurations": int(manifest["num_sequences"] * manifest["snapshots_per_sequence"]),
        "t_eval": t_eval,
        "support_pair_observations": int(len(w)),
        "all_offdiagonal_pair_observations": int(len(w_all)),
        "pearson_W_J": pearson(w, j),
        "a_star": a_star,
        "a_star_definition": "equal-weight mean of the six class ratios <W>_c/J_c",
        "pointwise_relative_rmse": float(np.sqrt(np.mean(np.square(w - a_star * j)) / np.mean(np.square(w)))),
        "class_ratio_relative_rms": float(np.sqrt(np.mean(np.square(class_ratios / a_star - 1.0)))),
        "configuration_relative_rms": float(np.sqrt(configuration_residual / configuration_magnitude)),
        "far_to_support_mean_abs_ratio": float(np.mean(far_abs) / np.mean(support_abs)),
        "far_to_support_absolute_mass_ratio": mass_ratio,
        "support_absolute_weight_fraction": float(1.0 / (1.0 + mass_ratio)),
        "classes": classes,
        "distance_statistics": distance_statistics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--t-eval", type=float, default=1.0e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-observations", action="store_true")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    if sha256(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("the supplied checkpoint is not the frozen epoch-26 model")
    records, manifest = load_and_audit_data(data_dir)
    device = torch.device(args.device)
    observations = extract(records, checkpoint, device, args.batch_size, args.t_eval)
    summary = summarize(observations, checkpoint, manifest, args.t_eval)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.save_observations:
        np.savez_compressed(output_dir / "observations.npz", **observations)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
