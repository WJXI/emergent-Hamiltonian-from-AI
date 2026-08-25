"""Fit and freeze the sector-preserving constant projection of Arm 3.

This coordinate-only stage never opens physical force or component labels.
Each sector is projected onto the corresponding frozen dynamic Network 3
sector score before the six constant sectors are assembled.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import projection_utils as emergence  # noqa: E402
import pairformer_arm3 as arm3  # noqa: E402
import sector_scores as components  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "configs" / "block_projection_protocol_v1.json"


def load_projection_protocol(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        protocol = json.load(handle)
    expected = ["bond", "angle", "proper", "out_of_plane", "graph_distance_3", "graph_distance_gt_3"]
    if protocol["projection"]["blocks"] != expected:
        raise RuntimeError("sector-preserving block order changed")
    if protocol["projection"]["physical_labels_used_for_fit_or_selection"]:
        raise RuntimeError("physical labels cannot enter the block projection")
    return protocol


def verify_frozen_inputs(config: Mapping[str, Any], result_dir: Path) -> dict[str, Any]:
    manifest_path = arm3.EXPERIMENT_ROOT / "frozen_models.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    network3 = [item for item in manifest["artifacts"] if item["network"] == "Network 3"]
    checkpoints = [item for item in network3 if item["path"].endswith(".pt")]
    if len(checkpoints) != 9:
        raise RuntimeError("the frozen Network 3 ensemble is incomplete")
    for item in checkpoints:
        checkpoint = arm3.EXPERIMENT_ROOT / str(item["path"])
        if arm3.controlled.sha256_file(checkpoint) != item["sha256"]:
            raise RuntimeError("a frozen Network 3 checkpoint changed")
    return {
        "manifest": str(manifest_path),
        "clean_readout": {
            "frozen_readout": "three-seed mean at 0.000375 nm",
            "gate_passed": False,
        },
    }


def constant_block_score(
    model: arm3.TrueTopologyBasisPairformer,
    frames: np.ndarray,
    coefficient: np.ndarray,
    block_slice: tuple[int, int],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Cartesian score from one constant semantic block."""
    left, right = block_slice
    alpha = torch.as_tensor(coefficient, dtype=torch.float32, device=device)
    output = []
    for start in range(0, frames.shape[0], batch_size):
        point = torch.as_tensor(frames[start : start + batch_size], dtype=torch.float32, device=device)
        point = point.requires_grad_(True)
        basis = model.standardized_basis(point)[:, left:right]
        action = basis @ alpha
        score = -torch.autograd.grad(action.sum(), point)[0]
        output.append(score.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(output)


def exact_sum_audit(reconstructed: np.ndarray, direct: np.ndarray) -> dict[str, Any]:
    difference = np.asarray(reconstructed, dtype=np.float64) - np.asarray(direct, dtype=np.float64)
    denominator = max(float(np.sqrt(np.mean(np.square(direct)))), 1.0e-300)
    return {
        "maximum_absolute_score_error": float(np.max(np.abs(difference))),
        "rms_error_over_direct_score_rms": float(np.sqrt(np.mean(np.square(difference))) / denominator),
        "reconstructed_vs_direct": arm3.arm1.score_metric(reconstructed, direct),
    }


def gate_passed(metric: Mapping[str, float], gate: Mapping[str, float]) -> bool:
    return bool(
        metric["cosine"] >= float(gate["minimum_score_cosine"])
        and metric["rms_difference_over_reference"] <= float(gate["maximum_rms_difference_over_dynamic_score"])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=ROOT / "results" / "protocol_v1")
    parser.add_argument("--output-dir", type=Path, default=arm3.EXPERIMENT_ROOT / "reproduced" / "network_3_P")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    projection_protocol = load_projection_protocol(args.protocol.resolve())
    config = arm3.load_config(ROOT / "configs" / "protocol_v1.json")
    frozen = verify_frozen_inputs(config, result_dir)
    train, train_family, validation, validation_family, _ = arm3.load_coordinate_splits(config)
    if sorted(map(int, np.unique(train_family))) != projection_protocol["projection"]["fit_families"]:
        raise RuntimeError("fit families differ from the frozen P_block protocol")
    if sorted(map(int, np.unique(validation_family))) != projection_protocol["projection"]["selection_families"]:
        raise RuntimeError("selection families differ from the frozen P_block protocol")
    transform, blocks, center, scale = emergence.load_basis(result_dir)
    batch_size = int(args.batch_size)

    print("[P_block] computing frozen Arm 3 block scores on fit families 00-04", flush=True)
    dynamic_train, dynamic_train_sum_audit = components.ensemble_dynamic_block_scores(
        config, result_dir, train, train, transform, blocks, center, scale, batch_size
    )
    print("[P_block] computing frozen Arm 3 block scores on selection family 05", flush=True)
    dynamic_validation, dynamic_validation_sum_audit = components.ensemble_dynamic_block_scores(
        config, result_dir, train, validation, transform, blocks, center, scale, batch_size
    )

    device = torch.device(str(config["dsm"]["device"]) if torch.cuda.is_available() else "cpu")
    low = float(config["clean_readout"]["low_sigma_nm"])
    reference, _ = arm3.load_model(
        arm3.checkpoint_path(result_dir, 0, low), train, transform, blocks, center, scale, device
    )
    full_gram = emergence.standardized_gram(config, transform, scale)
    ridges = [float(value) for value in projection_protocol["projection"]["ridges"]]
    gate = projection_protocol["projection"]["collapse_gate"]
    full_alpha = np.zeros(sum(reference.block_sizes), dtype=np.float64)
    selected_train: dict[str, np.ndarray] = {}
    selected_validation: dict[str, np.ndarray] = {}
    block_reports: dict[str, Any] = {}
    selected_ridges = []

    for name, block_slice in zip(reference.block_names, reference.block_slices):
        left, right = block_slice
        print(f"[P_block] fitting {name} ({right-left} retained coefficients)", flush=True)
        directional = emergence.basis_directional_derivative(
            reference, train, dynamic_train[name], batch_size, device
        )[:, left:right]
        rhs = -directional.mean(axis=0)
        block_gram = full_gram[left:right, left:right]
        candidates = emergence.fit_constant_candidates(block_gram, rhs, ridges)
        candidate_metrics = []
        best = None
        for ridge, alpha in candidates:
            predicted = constant_block_score(reference, validation, alpha, block_slice, batch_size, device)
            metric = arm3.arm1.score_metric(predicted, dynamic_validation[name])
            candidate_metrics.append({"ridge": ridge, **metric})
            if best is None or metric["rms_difference_over_reference"] < best[0]:
                best = (metric["rms_difference_over_reference"], ridge, alpha, predicted, metric)
        assert best is not None
        _, selected_ridge, selected_alpha, validation_score, validation_metric = best
        train_score = constant_block_score(reference, train, selected_alpha, block_slice, batch_size, device)
        train_metric = arm3.arm1.score_metric(train_score, dynamic_train[name])
        full_alpha[left:right] = selected_alpha
        selected_train[name] = train_score
        selected_validation[name] = validation_score
        selected_ridges.append(selected_ridge)
        block_reports[name] = {
            "slice": [left, right],
            "coefficient_count": right - left,
            "selected_ridge": selected_ridge,
            "candidate_metrics_on_family_05": candidate_metrics,
            "fit_dynamic_to_constant": train_metric,
            "family_05_dynamic_to_constant": validation_metric,
            "family_05_collapse_gate_passed": gate_passed(validation_metric, gate),
        }

    constant_train = np.sum(np.stack(list(selected_train.values())), axis=0)
    constant_validation = np.sum(np.stack(list(selected_validation.values())), axis=0)
    direct_train = emergence.constant_score(reference, train, full_alpha, batch_size, device)
    direct_validation = emergence.constant_score(reference, validation, full_alpha, batch_size, device)
    train_total_metric = arm3.arm1.score_metric(constant_train, dynamic_train["total"])
    validation_total_metric = arm3.arm1.score_metric(constant_validation, dynamic_validation["total"])

    artifact_path = output_dir / "frozen_block_projection.npz"
    arm3.controlled.atomic_npz(
        artifact_path,
        standardized_coefficient=full_alpha,
        block_name=np.asarray(reference.block_names),
        block_slice=np.asarray(reference.block_slices, dtype=np.int64),
        selected_ridge=np.asarray(selected_ridges, dtype=np.float64),
        dynamic_train_score=dynamic_train["total"].astype(np.float32),
        constant_train_score=constant_train.astype(np.float32),
        dynamic_validation_score=dynamic_validation["total"].astype(np.float32),
        constant_validation_score=constant_validation.astype(np.float32),
    )

    report = {
        "schema_version": 1,
        "status": "coordinate-only sector-preserving projection frozen before P_block physical evaluation",
        "definition": projection_protocol["definition"],
        "projected_quantity": projection_protocol["controlled_change"]["projected_quantity"],
        "fit_families": projection_protocol["projection"]["fit_families"],
        "selection_families": projection_protocol["projection"]["selection_families"],
        "fit_frames": int(train.shape[0]),
        "selection_frames": int(validation.shape[0]),
        "batch_size": batch_size,
        "dynamic_readout": frozen["clean_readout"]["frozen_readout"],
        "dynamic_exact_sum_audit": {"fit": dynamic_train_sum_audit, "selection": dynamic_validation_sum_audit},
        "blocks": block_reports,
        "assembled_total": {
            "fit_dynamic_to_constant": train_total_metric,
            "family_05_dynamic_to_constant": validation_total_metric,
            "family_05_collapse_gate_passed": gate_passed(validation_total_metric, gate),
            "fit_exact_sum_audit": exact_sum_audit(constant_train, direct_train),
            "family_05_exact_sum_audit": exact_sum_audit(constant_validation, direct_validation),
        },
        "coordinate_only_collapse_gate": gate,
        "all_block_gates_passed_on_family_05": bool(
            all(value["family_05_collapse_gate_passed"] for value in block_reports.values())
        ),
        "artifact": {
            "path": str(artifact_path.resolve()),
            "sha256": arm3.controlled.sha256_file(artifact_path),
        },
        "protocol": {
            "path": str(args.protocol.resolve()),
            "sha256": arm3.controlled.sha256_file(args.protocol.resolve()),
        },
        "physical_force_or_component_labels_used_for_fit_or_selection": False,
    }
    arm3.atomic_json(output_dir / "coordinate_only_block_emergence_audit.json", report)
    print(json.dumps({"blocks": block_reports, "assembled_total": report["assembled_total"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
