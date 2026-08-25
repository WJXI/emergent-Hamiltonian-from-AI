"""Recompute the complete Phenol table from the released data and weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.func import jacrev, vmap


ROOT = Path(__file__).resolve().parent
for source in (
    ROOT / "src",
    ROOT / "phenol_dsm_controlled" / "src",
    ROOT / "phenol_pairformer_arm1" / "src",
    ROOT / "phenol_pairformer_arm3" / "src",
    ROOT / "src",
):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

import independent_sigma_dsm_experiment as network2_training  # noqa: E402
import pairformer_arm1 as network1  # noqa: E402
import pairformer_arm3 as network3  # noqa: E402
import sector_scores as sectors  # noqa: E402
import unified_dsm_experiment as structured  # noqa: E402


NETWORK1_RESULT = ROOT / "phenol_pairformer_arm1" / "results" / "protocol_v1"
NETWORK2_RESULT = (
    ROOT
    / "phenol_dsm_controlled"
    / "results"
    / "clean_limit_v2"
    / "true_topology"
    / "frozen_coordinate_only_ensemble.npz"
)
NETWORK3_RESULT = ROOT / "phenol_pairformer_arm3" / "results" / "protocol_v1"
HOLDOUT = ROOT / "phenol_dsm_controlled" / "results" / "final_holdout_v1"


def network1_scores(frames: np.ndarray, batch_size: int, device: torch.device) -> dict[str, np.ndarray]:
    sigma = 0.000375
    predictions = []
    for seed in range(3):
        path = NETWORK1_RESULT / "checkpoints" / f"seed_{seed:02d}_sigma_0p00037500.pt"
        model, _ = network1.load_model(path, device)
        predictions.append(network1.score_frames(model, frames, sigma, batch_size, device))
        del model
    return {"total": np.mean(np.stack(predictions), axis=0)}


def network2_scores(
    protocol: Mapping[str, Any],
    train: np.ndarray,
    frames: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    with np.load(NETWORK2_RESULT, allow_pickle=False) as values:
        coefficient = np.mean(
            np.asarray(values["raw_eta_lowest_sigma_seed"], dtype=np.float64), axis=0
        )
    dictionary, _ = structured.build_dictionary("true_topology", protocol, train)
    dictionary = dictionary.to(device=device, dtype=torch.float64)
    coefficient_tensor = torch.as_tensor(coefficient, dtype=torch.float64, device=device)
    if coefficient_tensor.shape != (dictionary.basis_count,):  # type: ignore[attr-defined]
        raise RuntimeError("Network 2 coefficient dimension changed")
    blocks = list(dict.fromkeys(dictionary.basis_block))  # type: ignore[attr-defined]
    block_index = {
        block: torch.as_tensor(
            [index for index, value in enumerate(dictionary.basis_block) if value == block],  # type: ignore[attr-defined]
            dtype=torch.long,
            device=device,
        )
        for block in blocks
    }

    def basis_single(single: torch.Tensor) -> torch.Tensor:
        return dictionary(single.unsqueeze(0)).squeeze(0)

    jacobian_single = jacrev(basis_single)
    output: dict[str, list[np.ndarray]] = {name: [] for name in ("total", *blocks)}
    for start in range(0, frames.shape[0], batch_size):
        batch = torch.as_tensor(frames[start : start + batch_size], dtype=torch.float64, device=device)
        jacobian = vmap(jacobian_single)(batch)
        output["total"].append(
            (-torch.einsum("bpnc,p->bnc", jacobian, coefficient_tensor)).detach().cpu().numpy()
        )
        for block, index in block_index.items():
            output[block].append(
                (-torch.einsum("bpnc,p->bnc", jacobian[:, index], coefficient_tensor[index]))
                .detach()
                .cpu()
                .numpy()
            )
    return sectors.add_combined({name: np.concatenate(parts) for name, parts in output.items()})


def load_network3_basis() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(NETWORK3_RESULT / "basis_preconditioner.npz", allow_pickle=False) as values:
        return (
            np.asarray(values["transform"], dtype=np.float64),
            np.asarray(values["block"]).astype(str),
            np.asarray(values["center"], dtype=np.float64),
            np.asarray(values["scale"], dtype=np.float64),
        )


def sector_point_metrics(
    scores: Mapping[str, np.ndarray], targets: Mapping[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    comparisons = {
        "total": ("total", "total"),
        "bond": ("bond", "bond"),
        "angle": ("angle", "angle"),
        "proper": ("proper", "proper"),
        "out_of_plane_vs_improper": ("out_of_plane", "improper"),
        "torsional_combined": ("torsional_combined", "torsional_combined"),
        "graph_distance_3_vs_exact_exception": ("graph_distance_3", "exception_dg3"),
        "graph_distance_gt_3_vs_pme_real": ("graph_distance_gt_3", "direct_pme_real_gt3"),
        "pair_combined_vs_pme_real_intramolecular": ("pair_combined", "pair_pme_real"),
        "four_atom_combined": ("four_atom_combined", "four_atom_physical"),
    }
    return {
        name: sectors.physical.point_metric(scores[prediction], targets[target])
        for name, (prediction, target) in comparisons.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "paper_evaluation.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    network3_config = network3.load_config(ROOT / "phenol_pairformer_arm3" / "configs" / "protocol_v1.json")
    protocol = network2_training.load_protocol(
        ROOT / "phenol_dsm_controlled" / "configs" / "clean_limit_protocol_v2.json"
    )
    train, _, _, _, _ = structured.load_coordinate_splits(protocol)
    with np.load(HOLDOUT / "selected_coordinates.npz", allow_pickle=False) as values:
        frames = np.asarray(values["phenol_coordinates_nm"], dtype=np.float32)
        family = np.asarray(values["family_id"], dtype=np.int16)
    if frames.shape != (1200, 13, 3) or sorted(map(int, np.unique(family))) != list(range(8)):
        raise RuntimeError("the released physical holdout changed")

    targets, target_audit = sectors.physical_targets(network3_config, frames, family)
    first = network1_scores(frames, args.batch_size, device)
    second = network2_scores(protocol, train, frames, args.batch_size, device)
    transform, blocks, center, scale = load_network3_basis()
    third, third_sum_audit = sectors.ensemble_dynamic_block_scores(
        network3_config,
        NETWORK3_RESULT,
        train,
        frames,
        transform,
        blocks,
        center,
        scale,
        args.batch_size,
    )
    third = sectors.add_combined(third)
    with np.load(NETWORK3_RESULT / "frozen_block_projection.npz", allow_pickle=False) as values:
        coefficient = np.asarray(values["standardized_coefficient"], dtype=np.float64)
    reference, _ = network3.load_model(
        network3.checkpoint_path(NETWORK3_RESULT, 0, 0.000375),
        train,
        transform,
        blocks,
        center,
        scale,
        device,
    )
    projected, projected_sum_audit = sectors.constant_block_scores(
        reference, frames, coefficient, args.batch_size, device
    )
    projected = sectors.add_combined(projected)

    first_metric = {
        "total": sectors.physical.point_metric(first["total"], targets["total"])
    }
    second_metric = sector_point_metrics(second, targets)
    third_metric = sector_point_metrics(third, targets)
    projected_metric = sector_point_metrics(projected, targets)
    constant_collapse = sectors.physical.point_metric(projected["total"], third["total"])

    report = {
        "schema_version": 1,
        "definition": "frozen common-low-noise evaluation; Network 3-P is the sector-preserving constant projection",
        "frames": int(frames.shape[0]),
        "families": sorted(map(int, np.unique(family))),
        "sigma_nm": 0.000375,
        "metrics": {
            "network_1": first_metric,
            "network_2": second_metric,
            "network_3": third_metric,
            "network_3_P": projected_metric,
        },
        "network_3_P_R_const_squared": constant_collapse["risk_reduction"],
        "network_3_dynamic_sum_audit": third_sum_audit,
        "network_3_P_sum_audit": projected_sum_audit,
        "physical_target_audit": target_audit,
        "physical_labels_used_for_training_projection_or_selection": False,
    }
    args.output.resolve().write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": report["metrics"], "R_const_squared": report["network_3_P_R_const_squared"]}, indent=2))


if __name__ == "__main__":
    main()
