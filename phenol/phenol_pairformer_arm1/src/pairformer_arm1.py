"""Graph-free scalar Pairformer used as Phenol Network 1."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT.parent
CONTROLLED_ROOT = EXPERIMENT_ROOT / "phenol_dsm_controlled"
SHARED_SRC = EXPERIMENT_ROOT / "src"
for source in (CONTROLLED_ROOT / "src", SHARED_SRC):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

import unified_dsm_experiment as controlled  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "protocol_v1.json"
DEFAULT_OUTPUT = ROOT / "results" / "protocol_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".tmp", delete=False, dir=path.parent) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config["network"] != "Network 1":
        raise RuntimeError("unexpected Network 1 protocol")
    dsm = config["dsm"]
    if dsm["noise_sigmas_nm"] != [0.000375, 0.00075, 0.0015]:
        raise RuntimeError("low-noise sigma schedule differs from Arms 2/3")
    if dsm["fit_noise_seeds"] != [261301, 261302, 261303] or dsm["selection_noise_seeds"] != [261401, 261402, 261403]:
        raise RuntimeError("DSM noise seeds differ from Arms 2/3")
    if not dsm["antithetic_noise"] or not dsm["centered_cartesian_noise"]:
        raise RuntimeError("common DSM noise construction disabled")
    forbidden = (
        "water_coordinates_used",
        "energy_or_force_labels_used_for_fit_selection",
        "molecular_bond_graph_used",
        "OpenMM_or_GAFF_parameters_used",
        "atom_names_or_indices_used_as_embeddings",
        "holdout_force_labels_used_for_architecture_or_checkpoint_selection",
    )
    if any(bool(config["guardrails"][key]) for key in forbidden):
        raise RuntimeError("graph-free coordinate-only firewall disabled")
    return config


def load_coordinate_splits(config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    manifest_path = EXPERIMENT_ROOT / str(config["dataset_manifest"])
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not manifest["coordinate_only"] or manifest["q1_derived_coordinates_used"] or int(manifest["dummy_atoms"]) != 0:
        raise RuntimeError("coordinate dataset firewall failed")
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    audit: dict[str, Any] = {"manifest": str(manifest_path.resolve()), "manifest_sha256": sha256_file(manifest_path)}
    for role, split in (("fit", "train"), ("selection", "validation")):
        item = manifest["split_files"][split]
        path = Path(item["path"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"{role} coordinate hash mismatch")
        with np.load(path, allow_pickle=False) as values:
            if sorted(values.files) != ["family_id", "phenol_coordinates_nm"]:
                raise RuntimeError(f"forbidden array in {role} split")
            coordinate = np.asarray(values["phenol_coordinates_nm"], dtype=np.float32)
            family = np.asarray(values["family_id"], dtype=np.int16)
        expected_frames = int(config["splits"][f"{role}_frames"])
        expected_families = [int(x) for x in config["splits"][f"{role}_families"]]
        if coordinate.shape != (expected_frames, 13, 3) or sorted(map(int, np.unique(family))) != expected_families:
            raise RuntimeError(f"{role} split changed")
        if float(np.max(np.abs(coordinate.mean(axis=1)))) > 2.0e-6:
            raise RuntimeError(f"translation gauge not fixed in {role}")
        arrays[role] = (coordinate, family)
        audit[f"{role}_path"] = str(path.resolve())
        audit[f"{role}_sha256"] = item["sha256"]
    return *arrays["fit"], *arrays["selection"], audit


class SigmaEmbedding(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, sigma: Tensor) -> Tensor:
        value = torch.log(sigma.clamp_min(1.0e-8) / 0.00075)
        feature = torch.stack((value, value.square(), torch.sin(value), torch.cos(value), torch.sin(2 * value), torch.cos(2 * value)), dim=-1)
        return self.net(feature)


class MolecularPairformerBlock(nn.Module):
    """Symmetric pair update followed by pair-biased atom attention."""

    def __init__(self, single_width: int, pair_width: int, heads: int) -> None:
        super().__init__()
        if single_width % heads:
            raise ValueError("single width must divide evenly over attention heads")
        self.heads = int(heads)
        self.single_norm_1 = nn.LayerNorm(single_width)
        self.single_norm_2 = nn.LayerNorm(single_width)
        self.pair_norm_1 = nn.LayerNorm(pair_width)
        self.pair_norm_2 = nn.LayerNorm(pair_width)
        self.single_to_pair = nn.Linear(2 * single_width, pair_width)
        self.pair_transition = nn.Sequential(nn.Linear(pair_width, 2 * pair_width), nn.SiLU(), nn.Linear(2 * pair_width, pair_width))
        self.qkv = nn.Linear(single_width, 3 * single_width)
        self.pair_bias = nn.Linear(pair_width, heads)
        self.attention_output = nn.Linear(single_width, single_width)
        self.single_transition = nn.Sequential(nn.Linear(single_width, 4 * single_width), nn.SiLU(), nn.Linear(4 * single_width, single_width))

    def forward(self, single: Tensor, pair: Tensor) -> tuple[Tensor, Tensor]:
        batch, atoms, width = single.shape
        normalized = self.single_norm_1(single)
        left = normalized.unsqueeze(2).expand(-1, -1, atoms, -1)
        right = normalized.unsqueeze(1).expand(-1, atoms, -1, -1)
        symmetric = torch.cat((left + right, torch.abs(left - right)), dim=-1)
        pair = pair + self.single_to_pair(symmetric)
        pair = pair + self.pair_transition(self.pair_norm_2(pair))
        pair = 0.5 * (pair + pair.transpose(1, 2))

        q, k, v = self.qkv(normalized).chunk(3, dim=-1)
        head_width = width // self.heads
        q = q.view(batch, atoms, self.heads, head_width).transpose(1, 2)
        k = k.view(batch, atoms, self.heads, head_width).transpose(1, 2)
        v = v.view(batch, atoms, self.heads, head_width).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(head_width))
        logits = logits + self.pair_bias(self.pair_norm_1(pair)).permute(0, 3, 1, 2)
        attended = torch.matmul(F.softmax(logits, dim=-1), v)
        attended = attended.transpose(1, 2).contiguous().view(batch, atoms, width)
        single = single + self.attention_output(attended)
        single = single + self.single_transition(self.single_norm_2(single))
        return single, pair


class GraphFreeScalarPairformer(nn.Module):
    """E(3)- and permutation-invariant scalar action without a bond graph."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        architecture = config["architecture"]
        elements = list(config["molecule"]["elements"])
        element_names = sorted(set(elements))
        element_to_id = {name: index for index, name in enumerate(element_names)}
        element_id = torch.as_tensor([element_to_id[name] for name in elements], dtype=torch.long)
        self.register_buffer("element_id", element_id)
        atom_count = len(elements)
        pair_mask = torch.triu(torch.ones(atom_count, atom_count, dtype=torch.bool), diagonal=1)
        self.register_buffer("pair_mask", pair_mask)
        single_width = int(architecture["single_width"])
        pair_width = int(architecture["pair_width"])
        heads = int(architecture["attention_heads"])
        self.element_embedding = nn.Embedding(len(element_names), single_width)
        self.sigma_single = SigmaEmbedding(single_width)
        self.sigma_pair = SigmaEmbedding(pair_width)
        centers = torch.linspace(float(architecture["distance_rbf_min_nm"]), float(architecture["distance_rbf_max_nm"]), int(architecture["distance_rbf_count"]))
        self.register_buffer("distance_centers", centers)
        spacing = float(centers[1] - centers[0])
        self.distance_width = spacing * float(architecture["distance_rbf_width_multiplier"])
        pair_types = []
        type_names = sorted({"-".join(sorted((elements[i], elements[j]))) for i in range(atom_count) for j in range(atom_count)})
        type_to_id = {name: index for index, name in enumerate(type_names)}
        for i in range(atom_count):
            pair_types.append([type_to_id["-".join(sorted((elements[i], elements[j])))] for j in range(atom_count)])
        self.register_buffer("pair_type_id", torch.as_tensor(pair_types, dtype=torch.long))
        self.pair_type_embedding = nn.Embedding(len(type_names), pair_width)
        self.distance_projection = nn.Sequential(nn.Linear(int(architecture["distance_rbf_count"]) + 2, pair_width), nn.SiLU(), nn.Linear(pair_width, pair_width))
        self.blocks = nn.ModuleList([
            MolecularPairformerBlock(single_width, pair_width, heads) for _ in range(int(architecture["blocks"]))
        ])
        self.single_readout = nn.Sequential(nn.LayerNorm(single_width), nn.Linear(single_width, single_width), nn.SiLU(), nn.Linear(single_width, 1))
        self.pair_readout = nn.Sequential(nn.LayerNorm(pair_width), nn.Linear(pair_width, pair_width), nn.SiLU(), nn.Linear(pair_width, 1))
        self.global_readout = nn.Sequential(
            nn.Linear(single_width + pair_width, single_width), nn.SiLU(), nn.Linear(single_width, 1)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, positions: Tensor, sigma: Tensor) -> Tensor:
        batch, atoms, _ = positions.shape
        delta = positions.unsqueeze(2) - positions.unsqueeze(1)
        distance = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1.0e-6)
        rbf = torch.exp(-0.5 * ((distance.unsqueeze(-1) - self.distance_centers) / self.distance_width).square())
        distance_feature = torch.cat((rbf, distance.unsqueeze(-1), torch.log1p(distance / 0.1).unsqueeze(-1)), dim=-1)
        single = self.element_embedding(self.element_id).unsqueeze(0).expand(batch, -1, -1)
        single = single + self.sigma_single(sigma).unsqueeze(1)
        pair = self.pair_type_embedding(self.pair_type_id).unsqueeze(0).expand(batch, -1, -1, -1)
        pair = pair + self.distance_projection(distance_feature) + self.sigma_pair(sigma).view(batch, 1, 1, -1)
        pair = 0.5 * (pair + pair.transpose(1, 2))
        for block in self.blocks:
            single, pair = block(single, pair)
        single_pooled = single.mean(dim=1)
        pair_pooled = pair[:, self.pair_mask].mean(dim=1)
        local_single = self.single_readout(single).sum(dim=1).squeeze(-1) / math.sqrt(float(atoms))
        local_pair = self.pair_readout(pair[:, self.pair_mask]).sum(dim=1).squeeze(-1) / math.sqrt(float(self.pair_mask.sum()))
        global_value = self.global_readout(torch.cat((single_pooled, pair_pooled), dim=-1)).squeeze(-1)
        return local_single + local_pair + global_value

    def audit(self) -> dict[str, Any]:
        return {
            "family": "graph-free scalar Pairformer",
            "parameters": int(sum(value.numel() for value in self.parameters() if value.requires_grad)),
            "inputs": ["element identity", "all pair distances", "noise sigma"],
            "bond_graph_used": False,
            "atom_index_embedding_used": False,
            "translation_rotation_reflection_invariant": True,
            "permutation_equivariant_encoder_and_invariant_scalar_pooling": True,
            "scalar_action": True,
            "score_from_autograd": True,
            "component_heads": False,
        }


def centered_noise(shape: Sequence[int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    noise = rng.standard_normal(shape).astype(np.float64)
    return noise - noise.mean(axis=1, keepdims=True)


def noise_bank(coordinates: np.ndarray, seed: int, repeats: int) -> tuple[np.ndarray, np.ndarray, str]:
    frames = []
    noises = []
    signature = None
    for repeat in range(int(repeats)):
        noise = centered_noise(coordinates.shape, int(seed) + 10007 * repeat)
        if signature is None:
            signature = controlled.sha256_array(noise)
        for sign in (-1.0, 1.0):
            frames.append(np.arange(coordinates.shape[0], dtype=np.int64))
            noises.append((sign * noise).astype(np.float32))
    return np.concatenate(frames), np.concatenate(noises), str(signature)


def action_score(model: nn.Module, positions: Tensor, sigma: Tensor, create_graph: bool) -> tuple[Tensor, Tensor]:
    value = positions.detach().requires_grad_(True)
    action = model(value, sigma)
    gradient = torch.autograd.grad(action.sum(), value, create_graph=create_graph)[0]
    return action, -gradient


def dsm_loss(model: nn.Module, clean: Tensor, epsilon: Tensor, sigma_value: float, create_graph: bool) -> Tensor:
    sigma = torch.full((clean.shape[0],), float(sigma_value), dtype=clean.dtype, device=clean.device)
    noisy = clean + sigma.view(-1, 1, 1) * epsilon
    _, score = action_score(model, noisy, sigma, create_graph=create_graph)
    residual = epsilon + sigma.view(-1, 1, 1) * score
    return 0.5 * residual.square().sum(dim=(1, 2)).mean()


def evaluate_fixed_bank(
    model: nn.Module,
    coordinates: Tensor,
    frame_index: np.ndarray,
    noise: np.ndarray,
    sigma_value: float,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    baseline_sum = 0.0
    samples = 0
    for start in range(0, len(frame_index), int(batch_size)):
        stop = min(start + int(batch_size), len(frame_index))
        index = torch.as_tensor(frame_index[start:stop], dtype=torch.long, device=coordinates.device)
        epsilon = torch.as_tensor(noise[start:stop], dtype=coordinates.dtype, device=coordinates.device)
        with torch.enable_grad():
            loss = dsm_loss(model, coordinates[index], epsilon, sigma_value, create_graph=False)
        count = stop - start
        loss_sum += float(loss.detach()) * count
        baseline_sum += 0.5 * float(epsilon.square().sum().detach())
        samples += count
    loss_value = loss_sum / samples
    baseline = baseline_sum / samples
    return {
        "objective_per_frame": loss_value,
        "zero_score_objective_per_frame": baseline,
        "fractional_gain_over_zero_score": 1.0 - loss_value / baseline,
        "samples_including_antithetic_signs": int(samples),
    }


def save_checkpoint(path: Path, model: GraphFreeScalarPairformer, config: Mapping[str, Any], metadata: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "config": dict(config), "metadata": dict(metadata)}, temporary)
    os.replace(temporary, path)


def load_model(path: Path, device: torch.device) -> tuple[GraphFreeScalarPairformer, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = GraphFreeScalarPairformer(payload["config"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload["metadata"]


def train_one(
    config: Mapping[str, Any],
    train: np.ndarray,
    validation: np.ndarray,
    sigma_value: float,
    fit_seed: int,
    selection_seed: int,
    output_path: Path,
    smoke_epochs: int | None,
) -> dict[str, Any]:
    dsm = config["dsm"]
    epochs = int(smoke_epochs if smoke_epochs is not None else dsm["epochs"])
    device = torch.device(str(dsm["device"]) if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(fit_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(fit_seed))
    model = GraphFreeScalarPairformer(config).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(dsm["learning_rate"]), weight_decay=float(dsm["weight_decay"]))
    train_frames = torch.as_tensor(train, dtype=torch.float32, device=device)
    validation_frames = torch.as_tensor(validation, dtype=torch.float32, device=device)
    fit_index, fit_noise, fit_signature = noise_bank(train, fit_seed, int(dsm["fit_noise_repeats_per_sigma"]))
    selection_index, selection_noise, selection_signature = noise_bank(validation, selection_seed, int(dsm["selection_noise_repeats_per_sigma"]))
    rng = np.random.default_rng(int(fit_seed) + 81001)
    best_loss = math.inf
    best_epoch = -1
    best_state = None
    history = []
    batch_size = int(dsm["batch_size"])
    validation_every = int(dsm["validation_every_epochs"])
    running_path = output_path.with_suffix(".running.pt")
    start_epoch = 1
    if running_path.exists():
        running = torch.load(running_path, map_location="cpu", weights_only=False)
        model.load_state_dict(running["state_dict"])
        optimizer.load_state_dict(running["optimizer_state_dict"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
        best_loss = float(running["best_loss"])
        best_epoch = int(running["best_epoch"])
        best_state = running["best_state"]
        history = list(running["history"])
        rng.bit_generator.state = running["numpy_rng_state"]
        start_epoch = int(running["epoch"]) + 1
        print(json.dumps({"resume": str(running_path), "start_epoch": start_epoch, "sigma_nm": sigma_value, "fit_seed": fit_seed}), flush=True)
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        order = rng.permutation(len(fit_index))
        train_loss_sum = 0.0
        train_samples = 0
        for start in range(0, len(order), batch_size):
            chosen = order[start : start + batch_size]
            index = torch.as_tensor(fit_index[chosen], dtype=torch.long, device=device)
            epsilon = torch.as_tensor(fit_noise[chosen], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = dsm_loss(model, train_frames[index], epsilon, sigma_value, create_graph=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(dsm["gradient_clip"]))
            if not torch.isfinite(loss) or not torch.isfinite(gradient_norm):
                raise RuntimeError("non-finite Pairformer training state")
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(chosen)
            train_samples += len(chosen)
        if epoch == 1 or epoch % validation_every == 0 or epoch == epochs:
            validation_metric = evaluate_fixed_bank(model, validation_frames, selection_index, selection_noise, sigma_value, batch_size)
            row = {
                "epoch": epoch,
                "fit_objective_per_frame": train_loss_sum / train_samples,
                "selection_objective_per_frame": validation_metric["objective_per_frame"],
                "selection_fractional_gain": validation_metric["fractional_gain_over_zero_score"],
            }
            history.append(row)
            print(json.dumps({"sigma_nm": sigma_value, "fit_seed": fit_seed, **row}), flush=True)
            if float(validation_metric["objective_per_frame"]) < best_loss:
                best_loss = float(validation_metric["objective_per_frame"])
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            running_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_running = running_path.with_suffix(running_path.suffix + ".tmp")
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_loss": best_loss,
                    "best_epoch": best_epoch,
                    "best_state": best_state,
                    "history": history,
                    "numpy_rng_state": rng.bit_generator.state,
                },
                temporary_running,
            )
            os.replace(temporary_running, running_path)
    if best_state is None:
        raise RuntimeError("no Pairformer checkpoint selected")
    model.load_state_dict(best_state)
    fit_metric = evaluate_fixed_bank(model, train_frames, fit_index, fit_noise, sigma_value, batch_size)
    selection_metric = evaluate_fixed_bank(model, validation_frames, selection_index, selection_noise, sigma_value, batch_size)
    metadata = {
        "sigma_nm": float(sigma_value),
        "fit_seed": int(fit_seed),
        "selection_seed": int(selection_seed),
        "best_epoch": int(best_epoch),
        "fit_noise_sha256": fit_signature,
        "selection_noise_sha256": selection_signature,
        "fit": fit_metric,
        "selection": selection_metric,
        "history": history,
        "architecture": model.audit(),
    }
    save_checkpoint(output_path, model, config, metadata)
    metadata["checkpoint"] = str(output_path.resolve())
    metadata["checkpoint_sha256"] = sha256_file(output_path)
    return metadata


def score_frames(model: nn.Module, frames: np.ndarray, sigma_value: float, batch_size: int, device: torch.device) -> np.ndarray:
    output = []
    for start in range(0, frames.shape[0], int(batch_size)):
        batch = torch.as_tensor(frames[start : start + int(batch_size)], dtype=torch.float32, device=device)
        sigma = torch.full((batch.shape[0],), float(sigma_value), dtype=batch.dtype, device=device)
        with torch.enable_grad():
            _, score = action_score(model, batch, sigma, create_graph=False)
        output.append(score.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(output)


def score_metric(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    dot = float(np.sum(first * second))
    first_sq = float(np.sum(first * first))
    second_sq = float(np.sum(second * second))
    difference = float(np.sum((first - second) ** 2))
    return {
        "cosine": dot / math.sqrt(max(first_sq * second_sq, 1.0e-300)),
        "norm_ratio": math.sqrt(first_sq / max(second_sq, 1.0e-300)),
        "rms_difference_over_reference": math.sqrt(difference / max(second_sq, 1.0e-300)),
    }


def project_internal_force(positions: Tensor, forces: Tensor) -> Tensor:
    """Remove Euclidean translation and infinitesimal-rotation force modes."""
    centered_positions = positions - positions.mean(dim=1, keepdim=True)
    internal = forces - forces.mean(dim=1, keepdim=True)
    basis = torch.eye(3, dtype=positions.dtype, device=positions.device)
    modes = torch.cross(
        basis.reshape(1, 3, 1, 3).expand(positions.shape[0], -1, positions.shape[1], -1),
        centered_positions.unsqueeze(1).expand(-1, 3, -1, -1),
        dim=-1,
    )
    gram = torch.einsum("bkni,blni->bkl", modes, modes)
    rhs = torch.einsum("bkni,bni->bk", modes, internal)
    coefficients = torch.linalg.solve(gram, rhs.unsqueeze(-1)).squeeze(-1)
    rotation = torch.einsum("bk,bkni->bni", coefficients, modes)
    return internal - rotation


def point_metric(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    dot = float(np.sum(prediction * target))
    prediction_squared = float(np.sum(prediction * prediction))
    target_squared = float(np.sum(target * target))
    error_squared = float(np.sum((prediction - target) ** 2))
    return {
        "cosine": dot / math.sqrt(max(prediction_squared * target_squared, 1.0e-300)),
        "norm_ratio": math.sqrt(prediction_squared / max(target_squared, 1.0e-300)),
        "nrmse": math.sqrt(error_squared / max(target_squared, 1.0e-300)),
        "risk_reduction": 1.0 - error_squared / max(target_squared, 1.0e-300),
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
        "prediction_squared": np.sum(prediction * prediction, axis=(1, 2)),
        "target_squared": np.sum(target * target, axis=(1, 2)),
        "error_squared": np.sum((prediction - target) ** 2, axis=(1, 2)),
    }
    block_frames = int(bootstrap["block_frames"])
    blocks: dict[int, list[np.ndarray]] = {}
    for family in sorted(map(int, np.unique(family_id))):
        indices = np.flatnonzero(family_id == family)
        if len(indices) % block_frames:
            raise RuntimeError("family length is not divisible by bootstrap block")
        blocks[family] = [indices[start : start + block_frames] for start in range(0, len(indices), block_frames)]
    names = ("cosine", "norm_ratio", "nrmse", "risk_reduction")
    values = {name: np.empty(int(bootstrap["replicates"]), dtype=np.float64) for name in names}
    rng = np.random.default_rng(int(bootstrap["seed"]) + int(seed_offset))
    families = sorted(blocks)
    for replicate in range(int(bootstrap["replicates"])):
        selected = []
        for family in rng.choice(families, size=len(families), replace=True):
            family_blocks = blocks[int(family)]
            for chosen in rng.integers(0, len(family_blocks), size=len(family_blocks)):
                selected.append(family_blocks[int(chosen)])
        index = np.concatenate(selected)
        dot = float(per_frame["dot"][index].sum())
        prediction_squared = float(per_frame["prediction_squared"][index].sum())
        target_squared = float(per_frame["target_squared"][index].sum())
        error_squared = float(per_frame["error_squared"][index].sum())
        values["cosine"][replicate] = dot / math.sqrt(max(prediction_squared * target_squared, 1.0e-300))
        values["norm_ratio"][replicate] = math.sqrt(prediction_squared / max(target_squared, 1.0e-300))
        values["nrmse"][replicate] = math.sqrt(error_squared / max(target_squared, 1.0e-300))
        values["risk_reduction"][replicate] = 1.0 - error_squared / max(target_squared, 1.0e-300)
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


def assemble_clean_readout(config: Mapping[str, Any], checkpoints: Mapping[tuple[int, float], Path], validation: np.ndarray) -> dict[str, Any]:
    dsm = config["dsm"]
    clean = config["clean_readout"]
    device = torch.device(str(dsm["device"]) if torch.cuda.is_available() else "cpu")
    low = float(clean["low_sigma_nm"])
    high = float(clean["high_sigma_nm"])
    denominator = high * high - low * low
    extrapolated = []
    lowest = []
    low_high = []
    for seed_index in range(len(dsm["fit_noise_seeds"])):
        low_model, _ = load_model(checkpoints[(seed_index, low)], device)
        high_model, _ = load_model(checkpoints[(seed_index, high)], device)
        low_score = score_frames(low_model, validation, low, int(dsm["batch_size"]), device)
        high_score = score_frames(high_model, validation, high, int(dsm["batch_size"]), device)
        low_high.append({"seed_index": seed_index, **score_metric(low_score, high_score)})
        lowest.append(low_score)
        extrapolated.append((high * high * low_score - low * low * high_score) / denominator)
        del low_model, high_model
    pairwise = []
    minimum_seed = 1.0
    for first in range(len(extrapolated)):
        for second in range(first + 1, len(extrapolated)):
            metric = score_metric(extrapolated[first], extrapolated[second])
            pairwise.append({"seeds": [first, second], **metric})
            minimum_seed = min(minimum_seed, metric["cosine"])
    minimum_low_high = min(item["cosine"] for item in low_high)
    gate = clean["coordinate_only_stability_gate"]
    passed = minimum_seed >= float(gate["minimum_pairwise_seed_score_cosine"]) and minimum_low_high >= float(gate["minimum_low_vs_high_sigma_score_cosine"])
    return {
        "gate_passed": bool(passed),
        "minimum_pairwise_seed_score_cosine": float(minimum_seed),
        "minimum_low_vs_high_sigma_score_cosine": float(minimum_low_high),
        "pairwise_seed_metrics": pairwise,
        "low_vs_high_sigma_metrics": low_high,
        "frozen_readout": "three-seed mean extrapolated clean score" if passed else "three-seed mean at 0.000375 nm",
    }


def ensemble_score(config: Mapping[str, Any], checkpoints: Mapping[tuple[int, float], Path], frames: np.ndarray, gate_passed: bool) -> np.ndarray:
    dsm = config["dsm"]
    clean = config["clean_readout"]
    device = torch.device(str(dsm["device"]) if torch.cuda.is_available() else "cpu")
    low = float(clean["low_sigma_nm"])
    high = float(clean["high_sigma_nm"])
    denominator = high * high - low * low
    predictions = []
    for seed_index in range(len(dsm["fit_noise_seeds"])):
        low_model, _ = load_model(checkpoints[(seed_index, low)], device)
        low_score = score_frames(low_model, frames, low, int(dsm["batch_size"]), device)
        if gate_passed:
            high_model, _ = load_model(checkpoints[(seed_index, high)], device)
            high_score = score_frames(high_model, frames, high, int(dsm["batch_size"]), device)
            low_score = (high * high * low_score - low * low * high_score) / denominator
            del high_model
        predictions.append(low_score)
        del low_model
    return np.mean(np.stack(predictions), axis=0)


def run(
    config_path: Path,
    output_dir: Path,
    smoke_epochs: int | None,
    only_seed_index: int | None = None,
    only_sigma: float | None = None,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_config(config_path)
    train, train_family, validation, validation_family, data_audit = load_coordinate_splits(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = output_dir / "frozen_protocol.json"
    if not config_snapshot.exists():
        atomic_json(config_snapshot, {"config": config, "source_sha256": sha256_file(Path(__file__)), "config_sha256": sha256_file(config_path), "created_at": utc_now()})
    dsm = config["dsm"]
    checkpoints: dict[tuple[int, float], Path] = {}
    fits = []
    for seed_index, (fit_seed, selection_seed) in enumerate(zip(dsm["fit_noise_seeds"], dsm["selection_noise_seeds"])):
        if only_seed_index is not None and seed_index != int(only_seed_index):
            continue
        for sigma_value in dsm["noise_sigmas_nm"]:
            if only_sigma is not None and not math.isclose(float(sigma_value), float(only_sigma), rel_tol=0.0, abs_tol=1.0e-12):
                continue
            label = f"seed_{seed_index:02d}_sigma_{float(sigma_value):.8f}".replace(".", "p")
            checkpoint = output_dir / "checkpoints" / f"{label}.pt"
            metadata_path = checkpoint.with_suffix(".json")
            checkpoints[(seed_index, float(sigma_value))] = checkpoint
            if checkpoint.exists() and metadata_path.exists():
                with metadata_path.open("r", encoding="utf-8") as handle:
                    fits.append(json.load(handle))
                continue
            metadata = train_one(config, train, validation, float(sigma_value), int(fit_seed), int(selection_seed), checkpoint, smoke_epochs)
            atomic_json(metadata_path, metadata)
            fits.append(metadata)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    all_checkpoints: dict[tuple[int, float], Path] = {}
    for seed_index in range(len(dsm["fit_noise_seeds"])):
        for sigma_value in dsm["noise_sigmas_nm"]:
            label = f"seed_{seed_index:02d}_sigma_{float(sigma_value):.8f}".replace(".", "p")
            all_checkpoints[(seed_index, float(sigma_value))] = output_dir / "checkpoints" / f"{label}.pt"
    complete = all(path.exists() and path.with_suffix(".json").exists() for path in all_checkpoints.values())
    clean_audit = assemble_clean_readout(config, all_checkpoints, validation) if complete else {
        "status": "partial training; clean readout deferred",
        "complete_checkpoint_count": sum(path.exists() and path.with_suffix(".json").exists() for path in all_checkpoints.values()),
        "expected_checkpoint_count": len(all_checkpoints),
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": utc_now(),
        "protocol": config["protocol_name"],
        "smoke_epochs": smoke_epochs,
        "data": data_audit,
        "fit_families": sorted(map(int, np.unique(train_family))),
        "selection_families": sorted(map(int, np.unique(validation_family))),
        "fits": fits,
        "clean_readout": clean_audit,
        "architecture": GraphFreeScalarPairformer(config).audit(),
        "holdout_force_labels_used_for_training_or_selection": False,
    }
    atomic_json(output_dir / "summary.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke-epochs", type=int, default=None)
    parser.add_argument("--only-seed-index", type=int, default=None)
    parser.add_argument("--only-sigma", type=float, default=None)
    args = parser.parse_args()
    report = run(
        args.config,
        args.output.resolve(),
        args.smoke_epochs,
        only_seed_index=args.only_seed_index,
        only_sigma=args.only_sigma,
    )
    print(json.dumps({"summary": str((args.output.resolve() / 'summary.json')), "clean_readout": report["clean_readout"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
