"""Exact chain-rule sector scores and blinded OpenMM evaluation utilities."""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT.parent
for source in (
    ROOT / "src",
    EXPERIMENT_ROOT / "phenol_dsm_controlled" / "src",
    EXPERIMENT_ROOT / "src",
):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

import pairformer_arm3 as arm3  # noqa: E402
import physical_targets as physical  # noqa: E402


def score_metric(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return arm3.arm1.point_metric(prediction, target)


def model_block_scores(
    model: arm3.TrueTopologyBasisPairformer,
    frames: np.ndarray,
    sigma_value: float,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    model.eval()
    output: dict[str, list[np.ndarray]] = {name: [] for name in ("total", *model.block_names)}
    direct_parts = []
    for start in range(0, frames.shape[0], batch_size):
        positions = torch.as_tensor(frames[start : start + batch_size], dtype=torch.float32, device=device)
        positions = positions.requires_grad_(True)
        sigma = torch.full((positions.shape[0],), sigma_value, dtype=torch.float32, device=device)
        basis = model.standardized_basis(positions)
        action = model.action_from_basis(basis, sigma)
        eta = torch.autograd.grad(action.sum(), basis, retain_graph=True, create_graph=False)[0].detach()
        direct = -torch.autograd.grad(action.sum(), positions, retain_graph=True, create_graph=False)[0]
        blocks = []
        for number, (name, (left, right)) in enumerate(zip(model.block_names, model.block_slices)):
            scalar = (basis[:, left:right] * eta[:, left:right]).sum()
            retain = number + 1 < len(model.block_slices)
            component = -torch.autograd.grad(scalar, positions, retain_graph=retain, create_graph=False)[0]
            blocks.append(component)
            output[name].append(component.detach().cpu().numpy().astype(np.float64))
        reconstructed = torch.stack(blocks, dim=0).sum(dim=0)
        output["total"].append(reconstructed.detach().cpu().numpy().astype(np.float64))
        direct_parts.append(direct.detach().cpu().numpy().astype(np.float64))
    result = {name: np.concatenate(parts) for name, parts in output.items()}
    direct = np.concatenate(direct_parts)
    difference = result["total"] - direct
    audit = {
        "maximum_absolute_score_error": float(np.max(np.abs(difference))),
        "rms_error_over_direct_score_rms": float(
            np.sqrt(np.mean(np.square(difference))) / max(np.sqrt(np.mean(np.square(direct))), 1.0e-300)
        ),
        "reconstructed_vs_direct": arm3.arm1.score_metric(result["total"], direct),
    }
    return result, audit


def ensemble_dynamic_block_scores(
    config: Mapping[str, Any],
    result_dir: Path,
    train: np.ndarray,
    frames: np.ndarray,
    transform: np.ndarray,
    blocks: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    device = torch.device(str(config["dsm"]["device"]) if torch.cuda.is_available() else "cpu")
    sigma = float(config["clean_readout"]["low_sigma_nm"])
    seed_outputs = []
    audits = []
    for seed in range(len(config["dsm"]["fit_noise_seeds"])):
        model, _ = arm3.load_model(
            arm3.checkpoint_path(result_dir, seed, sigma), train, transform, blocks, center, scale, device
        )
        values, audit = model_block_scores(model, frames, sigma, batch_size, device)
        seed_outputs.append(values)
        audits.append({"seed": seed, **audit})
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    names = seed_outputs[0].keys()
    ensemble = {name: np.mean(np.stack([item[name] for item in seed_outputs]), axis=0) for name in names}
    return ensemble, {"per_seed": audits}


def constant_block_scores(
    model: arm3.TrueTopologyBasisPairformer,
    frames: np.ndarray,
    coefficient: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    alpha = torch.as_tensor(coefficient, dtype=torch.float32, device=device)
    output: dict[str, list[np.ndarray]] = {name: [] for name in ("total", *model.block_names)}
    direct_parts = []
    for start in range(0, frames.shape[0], batch_size):
        positions = torch.as_tensor(frames[start : start + batch_size], dtype=torch.float32, device=device)
        positions = positions.requires_grad_(True)
        basis = model.standardized_basis(positions)
        direct_action = (basis * alpha).sum()
        direct = -torch.autograd.grad(direct_action, positions, retain_graph=True)[0]
        components = []
        for number, (name, (left, right)) in enumerate(zip(model.block_names, model.block_slices)):
            action = (basis[:, left:right] * alpha[left:right]).sum()
            retain = number + 1 < len(model.block_slices)
            component = -torch.autograd.grad(action, positions, retain_graph=retain)[0]
            components.append(component)
            output[name].append(component.detach().cpu().numpy().astype(np.float64))
        reconstructed = torch.stack(components, dim=0).sum(dim=0)
        output["total"].append(reconstructed.detach().cpu().numpy().astype(np.float64))
        direct_parts.append(direct.detach().cpu().numpy().astype(np.float64))
    result = {name: np.concatenate(parts) for name, parts in output.items()}
    direct = np.concatenate(direct_parts)
    difference = result["total"] - direct
    audit = {
        "maximum_absolute_score_error": float(np.max(np.abs(difference))),
        "rms_error_over_direct_score_rms": float(
            np.sqrt(np.mean(np.square(difference))) / max(np.sqrt(np.mean(np.square(direct))), 1.0e-300)
        ),
        "reconstructed_vs_direct": arm3.arm1.score_metric(result["total"], direct),
    }
    return result, audit


def add_combined(scores: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = dict(scores)
    result["torsional_combined"] = result["proper"] + result["out_of_plane"]
    result["pair_combined"] = result["graph_distance_3"] + result["graph_distance_gt_3"]
    result["four_atom_combined"] = result["proper"] + result["out_of_plane"] + result["graph_distance_3"]
    return result


def physical_targets(
    config: Mapping[str, Any], frames: np.ndarray, family: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    holdout_config_path = arm3.CONTROLLED_ROOT / "configs" / "final_holdout_v1.json"
    with holdout_config_path.open("r", encoding="utf-8") as handle:
        holdout_config = json.load(handle)
    coordinates = {"phenol_coordinates_nm": np.asarray(frames), "family_id": np.asarray(family)}
    raw_path = arm3.CONTROLLED_ROOT / "results" / "final_holdout_v1" / "raw_openmm_force_components.npz"
    with np.load(raw_path, allow_pickle=False) as values:
        raw = {name: np.asarray(values[name], dtype=np.float64) for name in ("bond", "angle", "proper", "improper", "nonbonded", "total")}
    projected, beta = physical.project_beta_targets(
        coordinates, raw, float(holdout_config["targets"]["temperature_kelvin"])
    )
    device = torch.device(str(holdout_config["runtime"]["device"]) if torch.cuda.is_available() else "cpu")
    pair_targets, pair_audit = physical.matched_pair_targets(
        holdout_config, coordinates["phenol_coordinates_nm"], beta, device
    )
    targets = dict(projected)
    targets.update(pair_targets)
    targets["torsional_combined"] = targets["proper"] + targets["improper"]
    targets["pair_pme_real"] = targets["exception_dg3"] + targets["direct_pme_real_gt3"]
    targets["four_atom_physical"] = targets["torsional_combined"] + targets["exception_dg3"]
    return targets, {
        "temperature_kelvin": float(holdout_config["targets"]["temperature_kelvin"]),
        "beta_mol_per_kj": beta,
        "raw_force_sha256": arm3.controlled.sha256_file(raw_path),
        "pair_target_audit": pair_audit,
        "bootstrap": holdout_config["bootstrap"],
    }


def component_metrics(
    scores: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    family: np.ndarray,
    bootstrap: Mapping[str, Any],
    offset: int,
) -> dict[str, Any]:
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
    result = {}
    for number, (name, (prediction_name, target_name)) in enumerate(comparisons.items()):
        result[name] = physical.hierarchical_interval(
            scores[prediction_name], targets[target_name], family, bootstrap, offset + number
        )
    return result


def concise(metrics: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    return {
        name: {
            "cosine": float(value["point"]["cosine"]),
            "norm_ratio": float(value["point"]["norm_ratio"]),
            "nrmse": float(value["point"]["nrmse"]),
            "risk_reduction": float(value["point"]["risk_reduction"]),
        }
        for name, value in metrics.items()
    }

