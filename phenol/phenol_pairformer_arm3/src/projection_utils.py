"""Coordinate-only utilities for the sector-preserving Network 3 projection."""

from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor
from torch.func import jvp, vmap


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import pairformer_arm3 as arm3  # noqa: E402


def load_basis(result_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(result_dir / "basis_preconditioner.npz", allow_pickle=False) as values:
        return (
            np.asarray(values["transform"], dtype=np.float64),
            np.asarray(values["block"]).astype(str),
            np.asarray(values["center"], dtype=np.float64),
            np.asarray(values["scale"], dtype=np.float64),
        )


def all_checkpoint_paths(config: Mapping[str, Any], result_dir: Path) -> dict[tuple[int, float], Path]:
    return {
        (seed, float(sigma)): arm3.checkpoint_path(result_dir, seed, float(sigma))
        for seed in range(len(config["dsm"]["fit_noise_seeds"]))
        for sigma in config["dsm"]["noise_sigmas_nm"]
    }


def ensemble_score(
    config: Mapping[str, Any],
    result_dir: Path,
    train: np.ndarray,
    frames: np.ndarray,
    transform: np.ndarray,
    blocks: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    clean_gate: Mapping[str, Any],
) -> np.ndarray:
    device = torch.device(str(config["dsm"]["device"]) if torch.cuda.is_available() else "cpu")
    low = float(config["clean_readout"]["low_sigma_nm"])
    high = float(config["clean_readout"]["high_sigma_nm"])
    denominator = high * high - low * low
    predictions = []
    for seed in range(len(config["dsm"]["fit_noise_seeds"])):
        low_model, _ = arm3.load_model(
            arm3.checkpoint_path(result_dir, seed, low), train, transform, blocks, center, scale, device
        )
        low_score = arm3.score_frames(low_model, frames, low, int(config["dsm"]["batch_size"]), device)
        if bool(clean_gate["gate_passed"]):
            high_model, _ = arm3.load_model(
                arm3.checkpoint_path(result_dir, seed, high), train, transform, blocks, center, scale, device
            )
            high_score = arm3.score_frames(high_model, frames, high, int(config["dsm"]["batch_size"]), device)
            low_score = (high * high * low_score - low * low * high_score) / denominator
            del high_model
        predictions.append(low_score)
        del low_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.mean(np.stack(predictions), axis=0)


def standardized_gram(config: Mapping[str, Any], transform: np.ndarray, scale: np.ndarray) -> np.ndarray:
    source = arm3.EXPERIMENT_ROOT / str(config["basis_artifact"])
    with np.load(source, allow_pickle=False) as values:
        raw_gram = np.asarray(values["clean_score_gram"], dtype=np.float64)
    retained = arm3.controlled.v2.torch_matmul_numpy(
        transform.T, arm3.controlled.v2.torch_matmul_numpy(raw_gram, transform)
    )
    return retained / scale[:, None] / scale[None, :]


def basis_directional_derivative(
    model: arm3.TrueTopologyBasisPairformer,
    frames: np.ndarray,
    directions: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    output = []

    def single(point: Tensor) -> Tensor:
        return model.standardized_basis(point.unsqueeze(0))[0]

    def single_jvp(point: Tensor, direction: Tensor) -> Tensor:
        return jvp(single, (point,), (direction,))[1]

    mapped = vmap(single_jvp)
    for start in range(0, frames.shape[0], batch_size):
        point = torch.as_tensor(frames[start : start + batch_size], dtype=torch.float32, device=device)
        direction = torch.as_tensor(directions[start : start + batch_size], dtype=torch.float32, device=device)
        with torch.no_grad():
            value = mapped(point, direction)
        output.append(value.cpu().numpy().astype(np.float64))
    return np.concatenate(output)


def constant_score(
    model: arm3.TrueTopologyBasisPairformer,
    frames: np.ndarray,
    coefficients: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    alpha = torch.as_tensor(coefficients, dtype=torch.float32, device=device)
    output = []
    for start in range(0, frames.shape[0], batch_size):
        point = torch.as_tensor(frames[start : start + batch_size], dtype=torch.float32, device=device).requires_grad_(True)
        action = model.standardized_basis(point) @ alpha
        score = -torch.autograd.grad(action.sum(), point)[0]
        output.append(score.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(output)


def fit_constant_candidates(
    gram: np.ndarray,
    rhs: np.ndarray,
    ridges: list[float],
) -> list[tuple[float, np.ndarray]]:
    scale = float(np.trace(gram) / gram.shape[0])
    gram_tensor = torch.as_tensor(gram, dtype=torch.float64, device="cpu")
    rhs_tensor = torch.as_tensor(rhs, dtype=torch.float64, device="cpu")
    identity = torch.eye(gram.shape[0], dtype=torch.float64, device="cpu")
    output = []
    for ridge in ridges:
        coefficient = torch.linalg.solve(gram_tensor + ridge * scale * identity, rhs_tensor)
        output.append((ridge, coefficient.numpy()))
    return output


def coefficient_variation(
    config: Mapping[str, Any],
    result_dir: Path,
    train: np.ndarray,
    frames: np.ndarray,
    transform: np.ndarray,
    blocks: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    clean_gate: Mapping[str, Any],
    maximum_frames: int = 1000,
) -> dict[str, Any]:
    device = torch.device(str(config["dsm"]["device"]) if torch.cuda.is_available() else "cpu")
    frames = frames[:maximum_frames]
    low = float(config["clean_readout"]["low_sigma_nm"])
    high = float(config["clean_readout"]["high_sigma_nm"])
    denominator = high * high - low * low
    seed_values = []
    for seed in range(len(config["dsm"]["fit_noise_seeds"])):
        per_sigma = []
        sigmas = [low, high] if bool(clean_gate["gate_passed"]) else [low]
        for sigma_value in sigmas:
            model, _ = arm3.load_model(
                arm3.checkpoint_path(result_dir, seed, sigma_value), train, transform, blocks, center, scale, device
            )
            values = []
            for start in range(0, frames.shape[0], int(config["dsm"]["batch_size"])):
                point = torch.as_tensor(
                    frames[start : start + int(config["dsm"]["batch_size"])], dtype=torch.float32, device=device
                )
                sigma = torch.full((point.shape[0],), sigma_value, dtype=torch.float32, device=device)
                basis = model.standardized_basis(point).detach().requires_grad_(True)
                action = model.action_from_basis(basis, sigma)
                eta = torch.autograd.grad(action.sum(), basis)[0]
                values.append(eta.detach().cpu().numpy().astype(np.float64))
            per_sigma.append(np.concatenate(values))
            del model
        value = per_sigma[0]
        if len(per_sigma) == 2:
            value = (high * high * per_sigma[0] - low * low * per_sigma[1]) / denominator
        seed_values.append(value)
    eta = np.mean(np.stack(seed_values), axis=0)
    standard_deviation = eta.std(axis=0)
    centered_rms = float(np.sqrt(np.mean(np.square(eta - eta.mean(axis=0, keepdims=True)))))
    total_rms = float(np.sqrt(np.mean(np.square(eta))))
    return {
        "frames": int(eta.shape[0]),
        "coefficient_dimension": int(eta.shape[1]),
        "raw_effective_coefficient_centered_rms_over_total_rms": centered_rms / max(total_rms, 1.0e-300),
        "per_coordinate_std_quantiles": {
            "q00": float(np.quantile(standard_deviation, 0.0)),
            "q50": float(np.quantile(standard_deviation, 0.5)),
            "q90": float(np.quantile(standard_deviation, 0.9)),
            "q100": float(np.quantile(standard_deviation, 1.0)),
        },
        "decisive_for_emergence": False,
        "reason": "240 coefficients map to at most 33 internal Cartesian score directions per configuration and retain a large gauge freedom",
    }

