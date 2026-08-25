"""Post-training physical targets and metrics for the phenol experiment."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import torch

import pair_force_targets as pair_targets


def project_internal_force(positions: torch.Tensor, forces: torch.Tensor) -> torch.Tensor:
    centered = positions - positions.mean(dim=1, keepdim=True)
    internal = forces - forces.mean(dim=1, keepdim=True)
    axes = torch.eye(3, dtype=positions.dtype, device=positions.device)
    modes = torch.cross(
        axes.reshape(1, 3, 1, 3).expand(positions.shape[0], -1, positions.shape[1], -1),
        centered.unsqueeze(1).expand(-1, 3, -1, -1),
        dim=-1,
    )
    gram = torch.einsum("bkni,blni->bkl", modes, modes)
    rhs = torch.einsum("bkni,bni->bk", modes, internal)
    coefficient = torch.linalg.solve(gram, rhs.unsqueeze(-1)).squeeze(-1)
    return internal - torch.einsum("bk,bkni->bni", coefficient, modes)


def project_beta_targets(
    coordinates: Mapping[str, np.ndarray], raw: Mapping[str, np.ndarray], temperature: float
) -> tuple[dict[str, np.ndarray], float]:
    beta = 1.0 / (0.00831446261815324 * float(temperature))
    positions = torch.as_tensor(coordinates["phenol_coordinates_nm"], dtype=torch.float64)
    projected = {}
    for name, value in raw.items():
        force = torch.as_tensor(beta * np.asarray(value), dtype=torch.float64)
        projected[name] = project_internal_force(positions, force).numpy()
    return projected, beta


def matched_pair_targets(
    config: Mapping[str, Any], frames: np.ndarray, beta: float, device: torch.device
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    parameters = pair_targets.extract_nonbonded_parameters(config)
    maximum = max(
        float(np.max(np.linalg.norm(frames[:, second] - frames[:, first], axis=-1)))
        for first, second, *_ in parameters["direct_rows"]
    )
    if maximum >= float(parameters["cutoff_nm"]):
        raise RuntimeError("a solute pair crossed the PME real-space cutoff")
    coulomb = float(config["coulomb_constant_kj_mol_nm_e2"])
    exception = beta * pair_targets.pair_force_targets(
        frames, parameters["exception_rows"], coulomb, "bare", device=device
    )
    direct = beta * pair_targets.pair_force_targets(
        frames,
        parameters["direct_rows"],
        coulomb,
        "pme_real",
        pme_alpha=float(parameters["pme_alpha_per_nm"]),
        device=device,
    )
    return {
        "exception_dg3": exception,
        "direct_pme_real_gt3": direct,
    }, {"parameter_audit": parameters, "maximum_gt3_distance_nm": maximum}


def point_metric(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    dot = float(np.sum(prediction * target))
    prediction_square = float(np.sum(prediction * prediction))
    target_square = float(np.sum(target * target))
    error_square = float(np.sum((prediction - target) ** 2))
    return {
        "cosine": dot / math.sqrt(max(prediction_square * target_square, 1e-300)),
        "norm_ratio": math.sqrt(prediction_square / max(target_square, 1e-300)),
        "nrmse": math.sqrt(error_square / max(target_square, 1e-300)),
        "risk_reduction": 1.0 - error_square / max(target_square, 1e-300),
    }


def hierarchical_interval(
    prediction: np.ndarray,
    target: np.ndarray,
    family_id: np.ndarray,
    bootstrap: Mapping[str, Any],
    seed_offset: int,
) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    per_frame = {
        "dot": np.sum(prediction * target, axis=(1, 2)),
        "pred_sq": np.sum(prediction * prediction, axis=(1, 2)),
        "target_sq": np.sum(target * target, axis=(1, 2)),
        "error_sq": np.sum((prediction - target) ** 2, axis=(1, 2)),
    }
    block_frames = int(bootstrap["block_frames"])
    blocks = {}
    for family in sorted(map(int, np.unique(family_id))):
        indices = np.flatnonzero(family_id == family)
        if len(indices) % block_frames:
            raise RuntimeError("family length is not divisible by the bootstrap block")
        blocks[family] = [
            indices[start : start + block_frames]
            for start in range(0, len(indices), block_frames)
        ]
    rng = np.random.default_rng(int(bootstrap["seed"]) + int(seed_offset))
    values = {
        name: np.empty(int(bootstrap["replicates"]), dtype=np.float64)
        for name in ("cosine", "norm_ratio", "nrmse", "risk_reduction")
    }
    families = sorted(blocks)
    for replicate in range(int(bootstrap["replicates"])):
        selected = []
        for family in rng.choice(families, size=len(families), replace=True):
            family_blocks = blocks[int(family)]
            selected.extend(
                family_blocks[int(choice)]
                for choice in rng.integers(0, len(family_blocks), size=len(family_blocks))
            )
        index = np.concatenate(selected)
        dot = float(per_frame["dot"][index].sum())
        prediction_square = float(per_frame["pred_sq"][index].sum())
        target_square = float(per_frame["target_sq"][index].sum())
        error_square = float(per_frame["error_sq"][index].sum())
        values["cosine"][replicate] = dot / math.sqrt(max(prediction_square * target_square, 1e-300))
        values["norm_ratio"][replicate] = math.sqrt(prediction_square / max(target_square, 1e-300))
        values["nrmse"][replicate] = math.sqrt(error_square / max(target_square, 1e-300))
        values["risk_reduction"][replicate] = 1.0 - error_square / max(target_square, 1e-300)
    return {
        "point": point_metric(prediction, target),
        "two_way_family_time_block_interval": {
            name: {
                "q025": float(np.quantile(array, 0.025)),
                "q50": float(np.quantile(array, 0.5)),
                "q975": float(np.quantile(array, 0.975)),
            }
            for name, array in values.items()
        },
        "independent_families": len(families),
        "time_blocks_per_family": len(blocks[families[0]]),
        "replicates": int(bootstrap["replicates"]),
    }
