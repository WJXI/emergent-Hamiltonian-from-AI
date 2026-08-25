"""Independent-low-noise DSM fits and clean-limit ensemble for Phenol."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import unified_dsm_experiment as base  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "clean_limit_protocol_v2.json"


def load_protocol(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = base.load_json(path)
    if tuple(config["arms"]) != base.ARM_NAMES:
        raise RuntimeError("Network 2 arm definition changed")
    if int(config["common_model"]["raw_basis_count"]) != 270:
        raise RuntimeError("raw basis capacity changed")
    if bool(config["common_model"]["coefficient_coupling_across_sigma"]):
        raise RuntimeError("sigma coupling was re-enabled")
    if config["dsm"]["noise_sigmas_nm"] != [0.000375, 0.00075, 0.0015]:
        raise RuntimeError("clean-limit sigma schedule changed")
    if len(config["dsm"]["fit_noise_seeds"]) != 3 or len(config["dsm"]["selection_noise_seeds"]) != 3:
        raise RuntimeError("three-seed replication changed")
    for key in ("legacy_test_used", "previous_final_holdout_used", "new_dsm_final_holdout_generated"):
        if bool(config["splits"][key]):
            raise RuntimeError(f"holdout firewall disabled: {key}")
    forbidden = (
        "energy_or_force_labels_for_fit_selection",
        "OpenMM_or_GAFF_parameters_for_basis",
        "OpenMM_force_term_lists_for_basis",
        "event_or_atom_name_embeddings",
        "shared_law_residual_decomposition",
    )
    if any(bool(config["guardrails"][key]) for key in forbidden):
        raise RuntimeError("oracle or residual guardrail disabled")
    return config


def array_metric(prediction: np.ndarray, reference: np.ndarray, gram: np.ndarray) -> dict[str, float | None]:
    def inner(left: np.ndarray, right: np.ndarray) -> float:
        return base.v2.dot_numpy(left, base.v2.torch_matmul_numpy(gram, right))

    pred2 = max(inner(prediction, prediction), 0.0)
    ref2 = max(inner(reference, reference), 0.0)
    cross = inner(prediction, reference)
    diff2 = max(inner(prediction - reference, prediction - reference), 0.0)
    return {
        "score_cosine": cross / math.sqrt(pred2 * ref2) if pred2 > 0.0 and ref2 > 0.0 else None,
        "score_norm_ratio_prediction_to_reference": math.sqrt(pred2 / ref2) if ref2 > 0.0 else None,
        "score_rms_difference_over_reference": math.sqrt(diff2 / ref2) if ref2 > 0.0 else None,
    }


def extrapolate_clean(eta_low: np.ndarray, sigma_low: float, eta_high: np.ndarray, sigma_high: float) -> np.ndarray:
    denominator = sigma_high**2 - sigma_low**2
    if denominator <= 0.0:
        raise ValueError("sigma_high must exceed sigma_low")
    return (sigma_high**2 * eta_low - sigma_low**2 * eta_high) / denominator


def reduced_moments(moments: Mapping[str, Any], rank: int) -> dict[str, Any]:
    return {
        "a": np.asarray(moments["a"], dtype=np.float64)[:rank, :rank],
        "b": np.asarray(moments["b"], dtype=np.float64)[:rank],
        "baseline": float(moments["baseline"]),
        "samples": int(moments["samples"]),
        "first_noise_sha256": str(moments["first_noise_sha256"]),
    }


def fit_one_sigma(
    dictionary: torch.nn.Module,
    transform: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    sigma: float,
    fit_seed: int,
    selection_seed: int,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    dsm = config["dsm"]
    reference = max(float(value) for value in dsm["noise_sigmas_nm"])
    train_full = base.compute_dsm_moments(
        dictionary,
        transform,
        train,
        [sigma],
        reference,
        int(dsm["fit_noise_repeats_per_sigma"]),
        fit_seed,
        int(dsm["batch_size"]),
    )
    validation_full = base.compute_dsm_moments(
        dictionary,
        transform,
        validation,
        [sigma],
        reference,
        int(dsm["selection_noise_repeats_per_sigma"]),
        selection_seed,
        int(dsm["batch_size"]),
    )
    rank = int(transform.shape[1])
    train_moments = reduced_moments(train_full, rank)
    validation_moments = reduced_moments(validation_full, rank)
    alpha, fit = base.solve_ridges(train_moments, validation_moments, dsm["ridge_grid"])
    eta = base.v2.torch_matmul_numpy(transform, alpha)
    summary = {
        "sigma_nm": sigma,
        "selected_ridge": fit["selected_ridge"],
        "train_fractional_gain_over_zero_score": (
            train_moments["baseline"] - base.objective(alpha, train_moments)
        )
        / train_moments["baseline"],
        "validation_fractional_gain_over_zero_score": (
            validation_moments["baseline"] - base.objective(alpha, validation_moments)
        )
        / validation_moments["baseline"],
        "fit_noise_sha256": train_moments["first_noise_sha256"],
        "selection_noise_sha256": validation_moments["first_noise_sha256"],
        "fit_samples_including_antithetic_signs": train_moments["samples"],
        "selection_samples_including_antithetic_signs": validation_moments["samples"],
        "ridge_candidates": fit["candidates"],
    }
    saved = {
        "train_a": np.asarray(train_moments["a"]),
        "train_b": np.asarray(train_moments["b"]),
        "train_baseline": np.asarray(train_moments["baseline"]),
        "selection_a": np.asarray(validation_moments["a"]),
        "selection_b": np.asarray(validation_moments["b"]),
        "selection_baseline": np.asarray(validation_moments["baseline"]),
        "alpha": alpha,
        "raw_eta": eta,
    }
    return eta, summary, saved


def run_arm(
    arm: str,
    config: Mapping[str, Any],
    train: np.ndarray,
    validation: np.ndarray,
    output_dir: Path,
    seed_limit: int | None,
) -> dict[str, Any]:
    dictionary, dictionary_audit = base.build_dictionary(arm, config, train)
    if int(dictionary.basis_count) != 270:  # type: ignore[attr-defined]
        raise RuntimeError(f"{arm} raw basis capacity changed")
    device_name = str(config["dsm"]["device"])
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in AI4S")
    dictionary = dictionary.to(device=torch.device(device_name), dtype=torch.float64)
    print(f"[clean-limit-v2] {arm}: clean score Gram", flush=True)
    gram = base.compute_score_gram(dictionary, train, int(config["dsm"]["batch_size"]))
    preflight = config["preconditioner"]
    transform, transformed_blocks, rank_summary = base.blockwise_transform(
        dictionary,
        gram,
        float(preflight["relative_eigenvalue_cutoff"]),
        float(preflight["absolute_eigenvalue_cutoff"]),
    )
    fit_seeds = [int(value) for value in config["dsm"]["fit_noise_seeds"]]
    selection_seeds = [int(value) for value in config["dsm"]["selection_noise_seeds"]]
    if seed_limit is not None:
        fit_seeds = fit_seeds[: int(seed_limit)]
        selection_seeds = selection_seeds[: int(seed_limit)]
    sigmas = [float(value) for value in config["dsm"]["noise_sigmas_nm"]]
    arm_output = output_dir / arm
    arm_output.mkdir(parents=True, exist_ok=True)
    seed_results = []
    extrapolated = []
    lowest = []
    for seed_number, (fit_seed, selection_seed) in enumerate(zip(fit_seeds, selection_seeds)):
        print(f"[clean-limit-v2] {arm}: noise seed {seed_number + 1}/{len(fit_seeds)}", flush=True)
        coefficients: dict[float, np.ndarray] = {}
        sigma_summary: dict[str, Any] = {}
        arrays: dict[str, np.ndarray] = {}
        for sigma in sigmas:
            print(f"[clean-limit-v2] {arm}:   sigma={sigma:.6g} nm", flush=True)
            eta, summary, saved = fit_one_sigma(
                dictionary, transform, train, validation, sigma, fit_seed, selection_seed, config
            )
            coefficients[sigma] = eta
            label = f"{sigma:.8f}".replace(".", "p")
            sigma_summary[f"{sigma:.8f}"] = summary
            arrays.update({f"{label}_{key}": value for key, value in saved.items()})
        low = float(config["clean_readout"]["low_sigma_nm"])
        high = float(config["clean_readout"]["high_sigma_nm"])
        clean = extrapolate_clean(coefficients[low], low, coefficients[high], high)
        lowest.append(coefficients[low])
        extrapolated.append(clean)
        arrays["raw_eta_lowest_sigma"] = coefficients[low]
        arrays["raw_eta_clean_extrapolated"] = clean
        base.atomic_npz(arm_output / f"seed_{seed_number:02d}_model_and_moments.npz", **arrays)
        seed_results.append(
            {
                "seed_index": seed_number,
                "fit_noise_seed": fit_seed,
                "selection_noise_seed": selection_seed,
                "per_sigma": sigma_summary,
                "low_vs_high_sigma": array_metric(coefficients[low], coefficients[high], gram),
            }
        )
    minimum_seed_cosine = 1.0
    pairwise = []
    for first in range(len(extrapolated)):
        for second in range(first + 1, len(extrapolated)):
            value = array_metric(extrapolated[first], extrapolated[second], gram)
            pairwise.append({"seeds": [first, second], **value})
            minimum_seed_cosine = min(minimum_seed_cosine, float(value["score_cosine"]))
    minimum_low_high = min(float(item["low_vs_high_sigma"]["score_cosine"]) for item in seed_results)
    gate = config["clean_readout"]["coordinate_only_stability_gate"]
    gate_passed = (
        minimum_seed_cosine >= float(gate["minimum_pairwise_seed_score_cosine"])
        and minimum_low_high >= float(gate["minimum_low_vs_high_sigma_score_cosine"])
    )
    selected_vectors = extrapolated if gate_passed else lowest
    ensemble = np.mean(np.stack(selected_vectors, axis=0), axis=0)
    base.atomic_npz(
        arm_output / "frozen_coordinate_only_ensemble.npz",
        raw_eta_clean=ensemble,
        raw_eta_extrapolated_seed=np.stack(extrapolated, axis=0),
        raw_eta_lowest_sigma_seed=np.stack(lowest, axis=0),
        clean_score_gram=gram,
        blockwise_transform=transform,
        transformed_block=np.asarray(transformed_blocks),
    )
    result = {
        "arm": arm,
        "dictionary": dictionary_audit,
        "rank": rank_summary,
        "seed_results": seed_results,
        "extrapolated_pairwise_seed_stability": pairwise,
        "minimum_pairwise_seed_score_cosine": minimum_seed_cosine,
        "minimum_low_vs_high_sigma_score_cosine": minimum_low_high,
        "clean_readout_stability_gate_passed": gate_passed,
        "frozen_readout": "three-seed mean extrapolated clean coefficient" if gate_passed else "three-seed mean at 0.000375 nm fallback",
        "legacy_test_or_labels_opened": False,
    }
    base.atomic_json(arm_output / "summary.json", result)
    return result


def run(config_path: Path, output_dir: Path, smoke_frames: int | None, seed_limit: int | None) -> dict[str, Any]:
    config = load_protocol(config_path)
    train, train_family, validation, validation_family, data_audit = base.load_coordinate_splits(config)
    if smoke_frames is not None:
        train = train[: int(smoke_frames)]
        train_family = train_family[: int(smoke_frames)]
        validation = validation[: int(smoke_frames)]
        validation_family = validation_family[: int(smoke_frames)]
    output_dir.mkdir(parents=True, exist_ok=True)
    results = [run_arm(arm, config, train, validation, output_dir, seed_limit) for arm in base.ARM_NAMES]
    signatures: dict[tuple[int, str, str], set[str]] = {}
    for result in results:
        for seed in result["seed_results"]:
            for sigma, summary in seed["per_sigma"].items():
                for split in ("fit", "selection"):
                    key = (int(seed["seed_index"]), sigma, split)
                    signatures.setdefault(key, set()).add(str(summary[f"{split}_noise_sha256"]))
    common_noise_verified = all(len(value) == 1 for value in signatures.values())
    if not common_noise_verified:
        raise RuntimeError("common random numbers changed across basis arms")
    comparison = {
        "created_at": base.utc_now(),
        "config_path": str(config_path.resolve()),
        "config_sha256": base.sha256_file(config_path),
        "source_path": str(Path(__file__).resolve()),
        "source_sha256": base.sha256_file(Path(__file__).resolve()),
        "data": data_audit,
        "actual_fit_frames": int(train.shape[0]),
        "actual_selection_frames": int(validation.shape[0]),
        "actual_fit_families": [int(x) for x in np.unique(train_family)],
        "actual_selection_families": [int(x) for x in np.unique(validation_family)],
        "smoke_frames": smoke_frames,
        "seed_limit": seed_limit,
        "common_noise_verified_across_arms": common_noise_verified,
        "legacy_test_or_force_energy_component_labels_opened": False,
        "new_dsm_holdout_generated_or_opened": False,
        "results": results,
    }
    base.atomic_json(output_dir / "comparison.json", comparison)
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke-frames", type=int, default=None)
    parser.add_argument("--seed-limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args.config.resolve(), args.output_dir.resolve(), args.smoke_frames, args.seed_limit)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
