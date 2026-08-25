"""Coordinate-only low-noise DSM utilities for Phenol Network 2.

Fit and selection read coordinate-only arrays.  The legacy test split and all
force, energy, solvent, force-field, and component arrays are deliberately not
opened by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.func import jacrev, vmap


DSM_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = DSM_ROOT.parent
SHARED_SRC = EXPERIMENT_ROOT / "src"
if str(SHARED_SRC) not in sys.path:
    sys.path.insert(0, str(SHARED_SRC))
import structured_basis as v2  # noqa: E402


DEFAULT_CONFIG = DSM_ROOT / "configs" / "protocol_v1.json"
ARM_NAMES = ("true_topology",)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".tmp", delete=False, dir=path.parent) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False, dir=path.parent) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_protocol(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_json(path)
    if tuple(config["arms"]) != ARM_NAMES:
        raise RuntimeError("the Network 2 arm definition changed")
    if int(config["common_model"]["raw_basis_count"]) != 270:
        raise RuntimeError("raw capacity control changed")
    if config["splits"]["fit"] != "train" or config["splits"]["selection"] != "validation":
        raise RuntimeError("fit/selection split changed")
    for key in ("legacy_test_used", "previous_final_holdout_used", "new_dsm_final_holdout_generated"):
        if bool(config["splits"][key]):
            raise RuntimeError(f"holdout firewall disabled: {key}")
    guardrails = config["guardrails"]
    forbidden = (
        "energy_or_force_labels_for_fit_selection",
        "OpenMM_or_GAFF_parameters_for_basis",
        "OpenMM_force_term_lists_for_basis",
        "event_or_atom_name_embeddings",
        "shared_law_residual_decomposition",
    )
    if any(bool(guardrails[key]) for key in forbidden):
        raise RuntimeError("a forbidden oracle or residual input was enabled")
    if list(guardrails["fit_and_selection_arrays"]) != ["phenol_coordinates_nm", "family_id"]:
        raise RuntimeError("coordinate-only array firewall changed")
    return config


def load_coordinate_splits(config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    manifest_path = EXPERIMENT_ROOT / str(config["dataset_manifest"])
    manifest = load_json(manifest_path)
    if not bool(manifest["coordinate_only"]) or bool(manifest["q1_derived_coordinates_used"]) or int(manifest["dummy_atoms"]) != 0:
        raise RuntimeError("coordinate manifest failed the physical-solute firewall")
    output: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    audit: dict[str, Any] = {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "legacy_test_opened": False,
        "force_energy_or_component_arrays_opened": False,
    }
    for role in ("fit", "selection"):
        split = str(config["splits"][role])
        item = manifest["split_files"][split]
        path = Path(item["path"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        if sha256_file(path) != str(item["sha256"]):
            raise RuntimeError(f"coordinate hash mismatch for {role}")
        with np.load(path, allow_pickle=False) as values:
            if sorted(values.files) != ["family_id", "phenol_coordinates_nm"]:
                raise RuntimeError(f"forbidden array found in {role}")
            coordinates = np.asarray(values["phenol_coordinates_nm"], dtype=np.float64)
            family = np.asarray(values["family_id"], dtype=np.int64)
        if coordinates.shape[1:] != (int(config["molecule"]["atoms"]), 3):
            raise RuntimeError(f"unexpected coordinate shape for {role}")
        if float(np.max(np.abs(coordinates.mean(axis=1)))) > 2.0e-6:
            raise RuntimeError(f"translation gauge not fixed for {role}")
        output[role] = (coordinates, family)
        audit[f"{role}_path"] = str(path.resolve())
        audit[f"{role}_sha256"] = str(item["sha256"])
        audit[f"{role}_frames"] = int(coordinates.shape[0])
        audit[f"{role}_families"] = [int(x) for x in np.unique(family)]
    return (*output["fit"], *output["selection"], audit)


def array_norm(array: np.ndarray) -> float:
    return float(math.sqrt(float(np.sum(np.square(np.asarray(array))))))


def build_dictionary(arm: str, config: Mapping[str, Any], train: np.ndarray) -> tuple[nn.Module, dict[str, Any]]:
    if arm != "true_topology":
        raise ValueError(f"Network 2 supports only the fixed structured basis, not {arm!r}")
    parent_path = EXPERIMENT_ROOT / "configs" / "structured_basis.json"
    parent = v2.load_protocol(parent_path)
    groups, audit = v2.build_dictionary_groups(parent, train)
    basis = parent["dictionary"]["radial_and_angle_basis"]
    dictionary = v2.GeometryDictionary(
        groups,
        basis["gaussian_centers_standardized"],
        float(basis["gaussian_width_standardized"]),
        parent["dictionary"]["proper_fourier_orders"],
    )
    return dictionary, {
        **audit,
        "kind": "fixed structured geometric basis",
        "basis_count": int(dictionary.basis_count),
        "basis_protocol": str(parent_path.resolve()),
        "molecular_graph_used": True,
    }


def dictionary_device(dictionary: nn.Module) -> torch.device:
    return next(dictionary.buffers()).device


def compute_score_gram(dictionary: nn.Module, coordinates: np.ndarray, batch_size: int) -> np.ndarray:
    basis_count = int(dictionary.basis_count)  # type: ignore[attr-defined]
    total = np.zeros((basis_count, basis_count), dtype=np.float64)

    def basis_single(single: Tensor) -> Tensor:
        return dictionary(single.unsqueeze(0)).squeeze(0)

    jacobian_single = jacrev(basis_single)
    device = dictionary_device(dictionary)
    for start in range(0, coordinates.shape[0], int(batch_size)):
        batch = torch.as_tensor(coordinates[start : start + int(batch_size)], dtype=torch.float64, device=device)
        jacobian = vmap(jacobian_single)(batch).flatten(2)
        total += torch.einsum("brd,bsd->rs", jacobian, jacobian).detach().cpu().numpy()
    return total / float(coordinates.shape[0])


def blockwise_transform(
    dictionary: nn.Module,
    gram: np.ndarray,
    relative_cutoff: float,
    absolute_cutoff: float,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    basis_blocks = list(dictionary.basis_block)  # type: ignore[attr-defined]
    ordered_blocks = list(dict.fromkeys(basis_blocks))
    pieces = []
    transformed_blocks: list[str] = []
    block_summary: dict[str, Any] = {}
    for block in ordered_blocks:
        index = np.asarray([number for number, name in enumerate(basis_blocks) if name == block], dtype=np.int64)
        local = 0.5 * (gram[np.ix_(index, index)] + gram[np.ix_(index, index)].T)
        values, vectors = v2.torch_eigh_numpy(local)
        maximum = max(float(values[-1]), 0.0)
        threshold = max(float(absolute_cutoff), float(relative_cutoff) * maximum)
        keep = values > threshold
        retained_values = values[keep]
        transform = np.zeros((gram.shape[0], int(keep.sum())), dtype=np.float64)
        if retained_values.size:
            transform[index] = vectors[:, keep] / np.sqrt(retained_values)[None, :]
        pieces.append(transform)
        transformed_blocks.extend([block] * int(keep.sum()))
        block_summary[block] = {
            "raw_basis_count": int(index.size),
            "retained_rank": int(keep.sum()),
            "discarded_rank": int(index.size - keep.sum()),
            "largest_eigenvalue": float(values[-1]),
            "smallest_retained_eigenvalue": float(retained_values[0]) if retained_values.size else None,
            "threshold": threshold,
        }
    combined = np.concatenate(pieces, axis=1)
    whitened = v2.torch_matmul_numpy(combined.T, v2.torch_matmul_numpy(gram, combined))
    global_values, _ = v2.torch_eigh_numpy(0.5 * (whitened + whitened.T))
    positive = global_values[global_values > max(1.0e-10, 1.0e-8 * max(float(global_values[-1]), 0.0))]
    summary = {
        "raw_basis_count": int(gram.shape[0]),
        "retained_blockwise_basis_count": int(combined.shape[1]),
        "whitening": "within declared blocks only",
        "blocks": block_summary,
        "global_whitened_smallest_eigenvalue": float(global_values[0]),
        "global_whitened_largest_eigenvalue": float(global_values[-1]),
        "global_whitened_condition_number": float(positive[-1] / positive[0]) if positive.size else None,
        "global_whitened_rank": int(positive.size),
    }
    return combined, transformed_blocks, summary


def time_features(sigma: float, reference: float) -> np.ndarray:
    return np.asarray([1.0, (float(sigma) / float(reference)) ** 2], dtype=np.float64)


def centered_noise(shape: Sequence[int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    noise = rng.standard_normal(shape).astype(np.float64)
    return noise - noise.mean(axis=1, keepdims=True)


def compute_dsm_moments(
    dictionary: nn.Module,
    transform: np.ndarray,
    coordinates: np.ndarray,
    sigmas: Sequence[float],
    sigma_reference: float,
    repeats: int,
    seed: int,
    batch_size: int,
) -> dict[str, Any]:
    rank = int(transform.shape[1])
    time_count = 2
    parameter_count = time_count * rank
    total_a = np.zeros((parameter_count, parameter_count), dtype=np.float64)
    total_b = np.zeros(parameter_count, dtype=np.float64)
    total_baseline = 0.0
    total_samples = 0
    per_sigma: dict[str, Any] = {}
    device = dictionary_device(dictionary)
    transform_tensor = torch.as_tensor(transform, dtype=torch.float64, device=device)

    def transformed_single(single: Tensor) -> Tensor:
        return dictionary(single.unsqueeze(0)).squeeze(0) @ transform_tensor

    jacobian_single = jacrev(transformed_single)
    first_noise_signature = None
    for sigma_number, sigma_value in enumerate(sigmas):
        sigma = float(sigma_value)
        features = time_features(sigma, sigma_reference)
        sigma_a = np.zeros_like(total_a)
        sigma_b = np.zeros_like(total_b)
        sigma_baseline = 0.0
        sigma_samples = 0
        for repeat in range(int(repeats)):
            noise_seed = int(seed + 1000003 * sigma_number + 10007 * repeat)
            noise_all = centered_noise(coordinates.shape, noise_seed)
            if first_noise_signature is None:
                first_noise_signature = sha256_array(noise_all)
            for sign in (-1.0, 1.0):
                signed_noise = sign * noise_all
                for start in range(0, coordinates.shape[0], int(batch_size)):
                    stop = min(start + int(batch_size), coordinates.shape[0])
                    clean = coordinates[start:stop]
                    epsilon_np = signed_noise[start:stop]
                    noisy = torch.as_tensor(clean + sigma * epsilon_np, dtype=torch.float64, device=device)
                    epsilon = torch.as_tensor(epsilon_np, dtype=torch.float64, device=device)
                    gradient = vmap(jacobian_single)(noisy).flatten(2)
                    gram_sum = torch.einsum("brd,bsd->rs", gradient, gradient).detach().cpu().numpy()
                    dot_sum = torch.einsum("brd,bd->r", gradient, epsilon.flatten(1)).detach().cpu().numpy()
                    batch_a = np.kron(np.outer(features, features), sigma * sigma * gram_sum)
                    batch_b = np.concatenate([sigma * value * dot_sum for value in features])
                    baseline = 0.5 * float(np.square(epsilon_np).sum())
                    sigma_a += batch_a
                    sigma_b += batch_b
                    sigma_baseline += baseline
                    sigma_samples += int(stop - start)
        sigma_a /= float(sigma_samples)
        sigma_b /= float(sigma_samples)
        sigma_baseline /= float(sigma_samples)
        key = f"{sigma:.8f}"
        per_sigma[key] = {"a": sigma_a, "b": sigma_b, "baseline": sigma_baseline, "samples": sigma_samples}
        total_a += sigma_a
        total_b += sigma_b
        total_baseline += sigma_baseline
        total_samples += sigma_samples
    sigma_count = float(len(sigmas))
    return {
        "a": total_a / sigma_count,
        "b": total_b / sigma_count,
        "baseline": total_baseline / sigma_count,
        "samples": total_samples,
        "per_sigma": per_sigma,
        "first_noise_sha256": first_noise_signature,
        "parameter_count": parameter_count,
        "retained_rank": rank,
    }


def objective(theta: np.ndarray, moments: Mapping[str, Any]) -> float:
    return float(
        moments["baseline"]
        - v2.dot_numpy(theta, moments["b"])
        + 0.5 * v2.quadratic_numpy(theta, moments["a"])
    )


def solve_ridges(train: Mapping[str, Any], validation: Mapping[str, Any], ridge_grid: Sequence[float]) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = 0.5 * (np.asarray(train["a"]) + np.asarray(train["a"]).T)
    values, vectors = v2.torch_eigh_numpy(matrix)
    projected = v2.torch_matmul_numpy(vectors.T, np.asarray(train["b"]))
    candidates = []
    coefficients = []
    for ridge_value in ridge_grid:
        ridge = float(ridge_value)
        theta = v2.torch_matmul_numpy(vectors, projected / (np.maximum(values, 0.0) + ridge))
        train_loss = objective(theta, train)
        validation_loss = objective(theta, validation)
        validation_gain = (float(validation["baseline"]) - validation_loss) / float(validation["baseline"])
        candidates.append(
            {
                "ridge": ridge,
                "train_objective_per_frame": train_loss,
                "validation_objective_per_frame": validation_loss,
                "validation_fractional_gain_over_zero_score": validation_gain,
                "coefficient_norm": array_norm(theta),
            }
        )
        coefficients.append(theta)
    selected_index = min(range(len(candidates)), key=lambda index: (candidates[index]["validation_objective_per_frame"], index))
    return coefficients[selected_index], {
        "candidates": candidates,
        "selected_index": int(selected_index),
        "selected_ridge": float(candidates[selected_index]["ridge"]),
        "selected_validation_fractional_gain_over_zero_score": float(candidates[selected_index]["validation_fractional_gain_over_zero_score"]),
        "train_normal_matrix_smallest_eigenvalue": float(values[0]),
        "train_normal_matrix_largest_eigenvalue": float(values[-1]),
    }


def moment_summary(theta: np.ndarray, moments: Mapping[str, Any]) -> dict[str, Any]:
    loss = objective(theta, moments)
    baseline = float(moments["baseline"])
    output = {
        "objective_per_frame": loss,
        "zero_score_objective_per_frame": baseline,
        "fractional_gain_over_zero_score": (baseline - loss) / baseline,
        "samples_including_antithetic_signs": int(moments["samples"]),
        "first_noise_sha256": moments["first_noise_sha256"],
        "per_sigma": {},
    }
    for key, values in moments["per_sigma"].items():
        local_loss = objective(theta, values)
        local_baseline = float(values["baseline"])
        output["per_sigma"][key] = {
            "objective_per_frame": local_loss,
            "zero_score_objective_per_frame": local_baseline,
            "fractional_gain_over_zero_score": (local_baseline - local_loss) / local_baseline,
            "samples_including_antithetic_signs": int(values["samples"]),
        }
    return output


def save_moments(path: Path, train: Mapping[str, Any], validation: Mapping[str, Any]) -> None:
    arrays: dict[str, np.ndarray] = {
        "train_a": np.asarray(train["a"]),
        "train_b": np.asarray(train["b"]),
        "train_baseline": np.asarray(train["baseline"]),
        "validation_a": np.asarray(validation["a"]),
        "validation_b": np.asarray(validation["b"]),
        "validation_baseline": np.asarray(validation["baseline"]),
    }
    for split, moments in (("train", train), ("validation", validation)):
        for sigma, values in moments["per_sigma"].items():
            label = sigma.replace(".", "p")
            arrays[f"{split}_{label}_a"] = np.asarray(values["a"])
            arrays[f"{split}_{label}_b"] = np.asarray(values["b"])
            arrays[f"{split}_{label}_baseline"] = np.asarray(values["baseline"])
    atomic_npz(path, **arrays)


def run_arm(
    arm: str,
    config: Mapping[str, Any],
    train: np.ndarray,
    validation: np.ndarray,
    output_dir: Path,
    smoke_frames: int | None,
) -> dict[str, Any]:
    arm_output = output_dir / arm
    arm_output.mkdir(parents=True, exist_ok=True)
    dictionary, dictionary_audit = build_dictionary(arm, config, train)
    expected = int(config["common_model"]["raw_basis_count"])
    if int(dictionary.basis_count) != expected:  # type: ignore[attr-defined]
        raise RuntimeError(f"{arm} raw basis count is not {expected}")
    device_name = str(config["dsm"]["device"])
    if smoke_frames is not None and not torch.cuda.is_available():
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in the AI4S environment")
    device = torch.device(device_name)
    dictionary = dictionary.to(device=device, dtype=torch.float64)
    batch_size = int(config["dsm"]["batch_size"])
    print(f"[unified-dsm] {arm}: clean score Gram on {train.shape[0]} frames", flush=True)
    clean_gram = compute_score_gram(dictionary, train, batch_size)
    preconditioner = config["preconditioner"]
    transform, transformed_blocks, rank_summary = blockwise_transform(
        dictionary,
        clean_gram,
        float(preconditioner["relative_eigenvalue_cutoff"]),
        float(preconditioner["absolute_eigenvalue_cutoff"]),
    )
    print(f"[unified-dsm] {arm}: retained {transform.shape[1]}/{expected}; train DSM moments", flush=True)
    dsm = config["dsm"]
    train_moments = compute_dsm_moments(
        dictionary,
        transform,
        train,
        dsm["noise_sigmas_nm"],
        float(config["common_model"]["sigma_reference_nm"]),
        int(dsm["fit_noise_repeats"]),
        int(dsm["fit_noise_seed"]),
        batch_size,
    )
    print(f"[unified-dsm] {arm}: validation DSM moments", flush=True)
    validation_moments = compute_dsm_moments(
        dictionary,
        transform,
        validation,
        dsm["noise_sigmas_nm"],
        float(config["common_model"]["sigma_reference_nm"]),
        int(dsm["selection_noise_repeats"]),
        int(dsm["selection_noise_seed"]),
        batch_size,
    )
    theta, fit = solve_ridges(train_moments, validation_moments, dsm["ridge_grid"])
    rank = int(transform.shape[1])
    alpha0 = theta[:rank]
    alpha2 = theta[rank:]
    eta0 = v2.torch_matmul_numpy(transform, alpha0)
    eta2 = v2.torch_matmul_numpy(transform, alpha2)
    fit["train"] = moment_summary(theta, train_moments)
    fit["validation"] = moment_summary(theta, validation_moments)
    fit["zero_noise_readout"] = {
        "definition": "eta0 in the raw scalar dictionary",
        "raw_eta0_norm": array_norm(eta0),
        "raw_eta2_norm": array_norm(eta2),
        "whitened_alpha0_norm": array_norm(alpha0),
        "whitened_alpha2_norm": array_norm(alpha2),
    }
    if hasattr(dictionary, "basis_block"):
        fit["zero_noise_readout"]["raw_eta0_norm_by_block"] = {
            block: array_norm(eta0[np.asarray(dictionary.basis_block) == block])  # type: ignore[attr-defined]
            for block in dict.fromkeys(dictionary.basis_block)  # type: ignore[attr-defined]
        }
    audit = {
        "created_at": utc_now(),
        "arm": arm,
        "smoke_frames": smoke_frames,
        "device": str(device),
        "dtype": "float64",
        "dictionary": dictionary_audit,
        "rank": rank_summary,
        "transformed_blocks": {name: transformed_blocks.count(name) for name in dict.fromkeys(transformed_blocks)},
        "coordinate_only_fit_and_selection": True,
        "legacy_test_opened": False,
        "force_energy_or_component_arrays_opened": False,
    }
    atomic_json(arm_output / "audit.json", audit)
    atomic_json(arm_output / "fit_summary.json", fit)
    atomic_npz(
        arm_output / "preconditioner.npz",
        clean_score_gram=clean_gram,
        blockwise_transform=transform,
        transformed_block=np.asarray(transformed_blocks),
    )
    save_moments(arm_output / "dsm_moments.npz", train_moments, validation_moments)
    atomic_npz(
        arm_output / "model.npz",
        theta=theta,
        alpha0=alpha0,
        alpha2=alpha2,
        raw_eta0=eta0,
        raw_eta2=eta2,
        blockwise_transform=transform,
    )
    return {"arm": arm, "audit": audit, "fit": fit}


def run(config_path: Path, output_dir: Path, arms: Sequence[str], smoke_frames: int | None) -> dict[str, Any]:
    config = load_protocol(config_path)
    train, train_family, validation, validation_family, data_audit = load_coordinate_splits(config)
    if smoke_frames is not None:
        count = int(smoke_frames)
        train, train_family = train[:count], train_family[:count]
        validation, validation_family = validation[:count], validation_family[:count]
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_audit = {
        "created_at": utc_now(),
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "source_path": str(Path(__file__).resolve()),
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "data": data_audit,
        "actual_fit_frames": int(train.shape[0]),
        "actual_selection_frames": int(validation.shape[0]),
        "actual_fit_families": [int(x) for x in np.unique(train_family)],
        "actual_selection_families": [int(x) for x in np.unique(validation_family)],
        "arms": list(arms),
        "smoke_frames": smoke_frames,
        "new_dsm_holdout_generated_or_opened": False,
    }
    atomic_json(output_dir / "protocol_audit.json", protocol_audit)
    results = []
    for arm in arms:
        if arm not in ARM_NAMES:
            raise ValueError(f"unknown arm: {arm}")
        results.append(run_arm(arm, config, train, validation, output_dir, smoke_frames))
    common_train_noise = {item["fit"]["train"]["first_noise_sha256"] for item in results}
    common_validation_noise = {item["fit"]["validation"]["first_noise_sha256"] for item in results}
    if len(common_train_noise) != 1 or len(common_validation_noise) != 1:
        raise RuntimeError("common-random-number control failed across arms")
    comparison = {
        "created_at": utc_now(),
        "protocol": protocol_audit,
        "common_train_noise_verified": True,
        "common_validation_noise_verified": True,
        "results": [
            {
                "arm": item["arm"],
                "raw_basis_count": item["audit"]["rank"]["raw_basis_count"],
                "retained_rank": item["audit"]["rank"]["retained_blockwise_basis_count"],
                "selected_ridge": item["fit"]["selected_ridge"],
                "validation_fractional_gain_over_zero_score": item["fit"]["validation"]["fractional_gain_over_zero_score"],
                "validation_per_sigma": item["fit"]["validation"]["per_sigma"],
            }
            for item in results
        ],
    }
    atomic_json(output_dir / "comparison.json", comparison)
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arms", default=",".join(ARM_NAMES), help="comma-separated subset of the three frozen arms")
    parser.add_argument("--smoke-frames", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arms = tuple(value.strip() for value in args.arms.split(",") if value.strip())
    result = run(args.config.resolve(), args.output_dir.resolve(), arms, args.smoke_frames)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
