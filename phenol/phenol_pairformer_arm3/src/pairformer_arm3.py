"""Structured-basis Pairformer with configuration-dependent coefficients."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT.parent
CONTROLLED_ROOT = EXPERIMENT_ROOT / "phenol_dsm_controlled"
ARM1_SRC = EXPERIMENT_ROOT / "phenol_pairformer_arm1" / "src"
SHARED_SRC = EXPERIMENT_ROOT / "src"
for source in (CONTROLLED_ROOT / "src", ARM1_SRC, SHARED_SRC):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

import unified_dsm_experiment as controlled  # noqa: E402
import pairformer_arm1 as arm1  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "protocol_v1.json"
DEFAULT_OUTPUT = ROOT / "results" / "protocol_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    if config["network"] != "Network 3":
        raise RuntimeError("unexpected Network 3 protocol")
    if int(config["basis"]["raw_count"]) != 270 or int(config["basis"]["retained_count"]) != 240:
        raise RuntimeError("frozen 270-to-240 basis definition changed")
    if sum(map(int, config["basis"]["retained_blocks"].values())) != 240:
        raise RuntimeError("retained block sizes do not sum to 240")
    dsm = config["dsm"]
    if dsm["noise_sigmas_nm"] != [0.000375, 0.00075, 0.0015]:
        raise RuntimeError("sigma schedule differs from Arms 1/2")
    if dsm["fit_noise_seeds"] != [261301, 261302, 261303] or dsm["selection_noise_seeds"] != [261401, 261402, 261403]:
        raise RuntimeError("common DSM seeds changed")
    if not dsm["antithetic_noise"] or not dsm["centered_cartesian_noise"]:
        raise RuntimeError("common DSM noise construction disabled")
    guardrails = config["guardrails"]
    forbidden = (
        "water_coordinates_used",
        "energy_or_force_labels_used_for_fit_selection",
        "OpenMM_or_GAFF_parameters_used",
        "atom_or_event_identity_embedding_used",
        "shared_law_residual_decomposition_used",
        "coefficient_variance_regularization_used",
    )
    if any(bool(guardrails[key]) for key in forbidden):
        raise RuntimeError("an Arm 3 no-oracle guardrail was disabled")
    architecture = config["architecture"]
    if architecture["constant_coefficient_penalty"] or architecture["linear_constant_skip"]:
        raise RuntimeError("architecture is explicitly biased toward constant coefficients")
    return config


def load_coordinate_splits(config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    return arm1.load_coordinate_splits(config)


def parent_protocol() -> dict[str, Any]:
    path = CONTROLLED_ROOT / "configs" / "clean_limit_protocol_v2.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_raw_dictionary(train: np.ndarray) -> nn.Module:
    dictionary, _ = controlled.build_dictionary("true_topology", parent_protocol(), train)
    if int(dictionary.basis_count) != 270:  # type: ignore[attr-defined]
        raise RuntimeError("the structured raw dictionary is no longer 270-dimensional")
    return dictionary


def load_frozen_transform(config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, str]:
    path = EXPERIMENT_ROOT / str(config["basis_artifact"])
    with np.load(path, allow_pickle=False) as values:
        transform = np.asarray(values["blockwise_transform"], dtype=np.float64)
        blocks = np.asarray(values["transformed_block"]).astype(str)
    expected = config["basis"]["retained_blocks"]
    observed = {name: int(np.sum(blocks == name)) for name in expected}
    if transform.shape != (270, 240) or observed != {name: int(value) for name, value in expected.items()}:
        raise RuntimeError("frozen Arm 2 transform or block allocation changed")
    return transform, blocks, controlled.sha256_file(path)


def compute_value_preconditioner(
    dictionary: nn.Module,
    transform: np.ndarray,
    train: np.ndarray,
    floor: float,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    dictionary = dictionary.to(device=device, dtype=torch.float64)
    transform_tensor = torch.as_tensor(transform, dtype=torch.float64, device=device)
    total = np.zeros(240, dtype=np.float64)
    total_square = np.zeros(240, dtype=np.float64)
    count = 0
    with torch.no_grad():
        for start in range(0, train.shape[0], batch_size):
            batch = torch.as_tensor(train[start : start + batch_size], dtype=torch.float64, device=device)
            retained = dictionary(batch) @ transform_tensor
            array = retained.cpu().numpy()
            total += array.sum(axis=0)
            total_square += np.square(array).sum(axis=0)
            count += array.shape[0]
    mean = total / float(count)
    variance = np.maximum(total_square / float(count) - np.square(mean), 0.0)
    scale = np.maximum(np.sqrt(variance), float(floor))
    return mean, scale


def prepare_basis_artifact(
    config: Mapping[str, Any], train: np.ndarray, output_dir: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    transform, blocks, source_hash = load_frozen_transform(config)
    path = output_dir / "basis_preconditioner.npz"
    device = torch.device(str(config["dsm"]["device"]) if torch.cuda.is_available() else "cpu")
    if path.exists():
        with np.load(path, allow_pickle=False) as values:
            saved_transform = np.asarray(values["transform"], dtype=np.float64)
            center = np.asarray(values["center"], dtype=np.float64)
            scale = np.asarray(values["scale"], dtype=np.float64)
            saved_blocks = np.asarray(values["block"]).astype(str)
        if not np.array_equal(saved_transform, transform) or not np.array_equal(saved_blocks, blocks):
            raise RuntimeError("saved Arm 3 basis artifact differs from frozen Arm 2 transform")
    else:
        dictionary = build_raw_dictionary(train)
        center, scale = compute_value_preconditioner(
            dictionary,
            transform,
            train,
            float(config["basis"]["value_scale_floor"]),
            device,
        )
        controlled.atomic_npz(path, transform=transform, center=center, scale=scale, block=blocks)
    audit = {
        "path": str(path.resolve()),
        "sha256": controlled.sha256_file(path),
        "source_transform_sha256": source_hash,
        "raw_basis_count": 270,
        "retained_basis_count": 240,
        "block_counts": {name: int(np.sum(blocks == name)) for name in dict.fromkeys(blocks)},
        "value_scale_minimum": float(scale.min()),
        "value_scale_median": float(np.median(scale)),
        "value_scale_maximum": float(scale.max()),
    }
    return transform, blocks, center, scale, audit


class TrueTopologyBasisPairformer(nn.Module):
    """A conservative action F(Phi(R)) with six physical-basis tokens."""

    def __init__(
        self,
        config: Mapping[str, Any],
        dictionary: nn.Module,
        transform: np.ndarray,
        blocks: np.ndarray,
        center: np.ndarray,
        scale: np.ndarray,
    ) -> None:
        super().__init__()
        architecture = config["architecture"]
        self.dictionary = dictionary
        for parameter in self.dictionary.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("basis_transform", torch.as_tensor(transform, dtype=torch.float32))
        self.register_buffer("basis_center", torch.as_tensor(center, dtype=torch.float32))
        self.register_buffer("basis_scale", torch.as_tensor(scale, dtype=torch.float32))
        ordered_blocks = list(config["basis"]["retained_blocks"])
        if list(dict.fromkeys(blocks.tolist())) != ordered_blocks:
            raise RuntimeError("retained basis block order changed")
        sizes = [int(np.sum(blocks == name)) for name in ordered_blocks]
        self.block_names = tuple(ordered_blocks)
        self.block_sizes = tuple(sizes)
        offsets = np.cumsum([0, *sizes])
        self.block_slices = tuple((int(offsets[i]), int(offsets[i + 1])) for i in range(len(sizes)))
        single_width = int(architecture["single_width"])
        pair_width = int(architecture["pair_width"])
        heads = int(architecture["attention_heads"])
        self.block_projection = nn.ModuleList([nn.Linear(size, single_width) for size in sizes])
        self.block_embedding = nn.Embedding(len(sizes), single_width)
        self.sigma_single = arm1.SigmaEmbedding(single_width)
        self.sigma_pair = arm1.SigmaEmbedding(pair_width)
        self.pair_seed = nn.Parameter(torch.zeros(len(sizes), len(sizes), pair_width))
        self.blocks = nn.ModuleList(
            [arm1.MolecularPairformerBlock(single_width, pair_width, heads) for _ in range(int(architecture["blocks"]))]
        )
        self.token_readout = nn.Sequential(
            nn.LayerNorm(single_width), nn.Linear(single_width, single_width), nn.SiLU(), nn.Linear(single_width, 1)
        )
        self.global_readout = nn.Sequential(
            nn.Linear(single_width + pair_width, single_width), nn.SiLU(), nn.Linear(single_width, 1)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.block_embedding.weight, std=0.02)
        nn.init.normal_(self.pair_seed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.25)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def standardized_basis(self, positions: Tensor) -> Tensor:
        raw = self.dictionary(positions)
        retained = raw @ self.basis_transform
        return (retained - self.basis_center) / self.basis_scale

    def action_from_basis(self, basis: Tensor, sigma: Tensor) -> Tensor:
        token = torch.stack(
            [projection(basis[:, start:stop]) for projection, (start, stop) in zip(self.block_projection, self.block_slices)],
            dim=1,
        )
        count = len(self.block_slices)
        block_id = torch.arange(count, dtype=torch.long, device=basis.device)
        token = token + self.block_embedding(block_id).unsqueeze(0) + self.sigma_single(sigma).unsqueeze(1)
        pair = 0.5 * (self.pair_seed + self.pair_seed.transpose(0, 1))
        pair = pair.unsqueeze(0).expand(basis.shape[0], -1, -1, -1)
        pair = pair + self.sigma_pair(sigma).view(basis.shape[0], 1, 1, -1)
        for block in self.blocks:
            token, pair = block(token, pair)
        token_pooled = token.mean(dim=1)
        pair_mask = torch.triu(torch.ones(count, count, dtype=torch.bool, device=basis.device), diagonal=1)
        pair_pooled = pair[:, pair_mask].mean(dim=1)
        local = self.token_readout(token).sum(dim=1).squeeze(-1) / math.sqrt(float(count))
        global_value = self.global_readout(torch.cat((token_pooled, pair_pooled), dim=-1)).squeeze(-1)
        return local + global_value

    def forward(self, positions: Tensor, sigma: Tensor) -> Tensor:
        return self.action_from_basis(self.standardized_basis(positions), sigma)

    def audit(self) -> dict[str, Any]:
        return {
            "family": "six-token true-topology basis Pairformer",
            "parameters": int(sum(value.numel() for value in self.parameters() if value.requires_grad)),
            "raw_basis_count": 270,
            "retained_basis_count": 240,
            "block_sizes": dict(zip(self.block_names, self.block_sizes)),
            "scalar_action": True,
            "score_from_autograd": True,
            "configuration_dependent_effective_coefficients_allowed": True,
            "constant_coefficient_penalty": False,
            "linear_constant_skip": False,
            "component_force_labels": False,
        }


def make_model(
    config: Mapping[str, Any],
    train: np.ndarray,
    transform: np.ndarray,
    blocks: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
) -> TrueTopologyBasisPairformer:
    return TrueTopologyBasisPairformer(config, build_raw_dictionary(train), transform, blocks, center, scale)


def save_checkpoint(
    path: Path, model: TrueTopologyBasisPairformer, config: Mapping[str, Any], metadata: Mapping[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "config": dict(config),
            "metadata": dict(metadata),
        },
        temporary,
    )
    os.replace(temporary, path)


def load_model(
    path: Path,
    train: np.ndarray,
    transform: np.ndarray,
    blocks: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
) -> tuple[TrueTopologyBasisPairformer, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = make_model(payload["config"], train, transform, blocks, center, scale).to(device=device, dtype=torch.float32)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload["metadata"]


def train_one(
    config: Mapping[str, Any],
    train: np.ndarray,
    validation: np.ndarray,
    transform: np.ndarray,
    blocks: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
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
    model = make_model(config, train, transform, blocks, center, scale).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(dsm["learning_rate"]), weight_decay=float(dsm["weight_decay"]))
    train_frames = torch.as_tensor(train, dtype=torch.float32, device=device)
    validation_frames = torch.as_tensor(validation, dtype=torch.float32, device=device)
    fit_index, fit_noise, fit_signature = arm1.noise_bank(train, fit_seed, int(dsm["fit_noise_repeats_per_sigma"]))
    selection_index, selection_noise, selection_signature = arm1.noise_bank(
        validation, selection_seed, int(dsm["selection_noise_repeats_per_sigma"])
    )
    rng = np.random.default_rng(int(fit_seed) + 83001)
    best_loss = math.inf
    best_epoch = -1
    best_state = None
    history: list[dict[str, Any]] = []
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
                if isinstance(value, Tensor):
                    state[key] = value.to(device)
        best_loss = float(running["best_loss"])
        best_epoch = int(running["best_epoch"])
        best_state = running["best_state"]
        history = list(running["history"])
        rng.bit_generator.state = running["numpy_rng_state"]
        start_epoch = int(running["epoch"]) + 1
        print(json.dumps({"resume": str(running_path), "start_epoch": start_epoch}), flush=True)
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
            loss = arm1.dsm_loss(model, train_frames[index], epsilon, sigma_value, create_graph=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(dsm["gradient_clip"]))
            if not torch.isfinite(loss) or not torch.isfinite(gradient_norm):
                raise RuntimeError("non-finite Arm 3 training state")
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(chosen)
            train_samples += len(chosen)
        if epoch == 1 or epoch % validation_every == 0 or epoch == epochs:
            metric = arm1.evaluate_fixed_bank(
                model, validation_frames, selection_index, selection_noise, sigma_value, batch_size
            )
            row = {
                "epoch": epoch,
                "fit_objective_per_frame": train_loss_sum / train_samples,
                "selection_objective_per_frame": metric["objective_per_frame"],
                "selection_fractional_gain": metric["fractional_gain_over_zero_score"],
            }
            history.append(row)
            print(json.dumps({"sigma_nm": sigma_value, "fit_seed": fit_seed, **row}), flush=True)
            if float(metric["objective_per_frame"]) < best_loss:
                best_loss = float(metric["objective_per_frame"])
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            running_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = running_path.with_suffix(running_path.suffix + ".tmp")
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
                temporary,
            )
            os.replace(temporary, running_path)
    if best_state is None:
        raise RuntimeError("no Arm 3 checkpoint selected")
    model.load_state_dict(best_state)
    fit_metric = arm1.evaluate_fixed_bank(model, train_frames, fit_index, fit_noise, sigma_value, batch_size)
    selection_metric = arm1.evaluate_fixed_bank(
        model, validation_frames, selection_index, selection_noise, sigma_value, batch_size
    )
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
    metadata["checkpoint_sha256"] = controlled.sha256_file(output_path)
    return metadata


def score_frames(model: nn.Module, frames: np.ndarray, sigma_value: float, batch_size: int, device: torch.device) -> np.ndarray:
    return arm1.score_frames(model, frames, sigma_value, batch_size, device)


def checkpoint_path(output_dir: Path, seed_index: int, sigma: float) -> Path:
    label = f"seed_{seed_index:02d}_sigma_{float(sigma):.8f}".replace(".", "p")
    return output_dir / "checkpoints" / f"{label}.pt"


def clean_readout_audit(
    config: Mapping[str, Any],
    output_dir: Path,
    train: np.ndarray,
    validation: np.ndarray,
    transform: np.ndarray,
    blocks: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
) -> dict[str, Any]:
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
        low_model, _ = load_model(checkpoint_path(output_dir, seed_index, low), train, transform, blocks, center, scale, device)
        high_model, _ = load_model(checkpoint_path(output_dir, seed_index, high), train, transform, blocks, center, scale, device)
        low_score = score_frames(low_model, validation, low, int(dsm["batch_size"]), device)
        high_score = score_frames(high_model, validation, high, int(dsm["batch_size"]), device)
        low_high.append({"seed_index": seed_index, **arm1.score_metric(low_score, high_score)})
        lowest.append(low_score)
        extrapolated.append((high * high * low_score - low * low * high_score) / denominator)
        del low_model, high_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    pairwise = []
    minimum_seed = 1.0
    for first in range(len(extrapolated)):
        for second in range(first + 1, len(extrapolated)):
            metric = arm1.score_metric(extrapolated[first], extrapolated[second])
            pairwise.append({"seeds": [first, second], **metric})
            minimum_seed = min(minimum_seed, metric["cosine"])
    minimum_low_high = min(item["cosine"] for item in low_high)
    gate = clean["coordinate_only_stability_gate"]
    passed = minimum_seed >= float(gate["minimum_pairwise_seed_score_cosine"]) and minimum_low_high >= float(
        gate["minimum_low_vs_high_sigma_score_cosine"]
    )
    return {
        "gate_passed": bool(passed),
        "minimum_pairwise_seed_score_cosine": float(minimum_seed),
        "minimum_low_vs_high_sigma_score_cosine": float(minimum_low_high),
        "pairwise_seed_metrics": pairwise,
        "low_vs_high_sigma_metrics": low_high,
        "frozen_readout": "three-seed mean extrapolated clean score" if passed else "three-seed mean at 0.000375 nm",
    }


def run(
    config_path: Path,
    output_dir: Path,
    smoke_epochs: int | None,
    smoke_frames: int | None,
    only_seed_index: int | None,
    only_sigma: float | None,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_config(config_path)
    train_full, train_family_full, validation_full, validation_family_full, data_audit = load_coordinate_splits(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    transform, blocks, center, scale, basis_audit = prepare_basis_artifact(config, train_full, output_dir)
    train = train_full if smoke_frames is None else train_full[: int(smoke_frames)]
    validation = validation_full if smoke_frames is None else validation_full[: int(smoke_frames)]
    train_family = train_family_full[: train.shape[0]]
    validation_family = validation_family_full[: validation.shape[0]]
    snapshot = output_dir / "frozen_protocol.json"
    if not snapshot.exists():
        atomic_json(
            snapshot,
            {
                "config": config,
                "config_sha256": controlled.sha256_file(config_path),
                "source_sha256": controlled.sha256_file(Path(__file__)),
                "basis": basis_audit,
                "created_at": utc_now(),
            },
        )
    dsm = config["dsm"]
    fits = []
    for seed_index, (fit_seed, selection_seed) in enumerate(zip(dsm["fit_noise_seeds"], dsm["selection_noise_seeds"])):
        if only_seed_index is not None and seed_index != int(only_seed_index):
            continue
        for sigma in dsm["noise_sigmas_nm"]:
            if only_sigma is not None and not math.isclose(float(sigma), float(only_sigma), abs_tol=1.0e-12):
                continue
            checkpoint = checkpoint_path(output_dir, seed_index, float(sigma))
            metadata_path = checkpoint.with_suffix(".json")
            if checkpoint.exists() and metadata_path.exists():
                with metadata_path.open("r", encoding="utf-8") as handle:
                    fits.append(json.load(handle))
                continue
            metadata = train_one(
                config,
                train,
                validation,
                transform,
                blocks,
                center,
                scale,
                float(sigma),
                int(fit_seed),
                int(selection_seed),
                checkpoint,
                smoke_epochs,
            )
            atomic_json(metadata_path, metadata)
            fits.append(metadata)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    expected = [checkpoint_path(output_dir, i, float(s)) for i in range(3) for s in dsm["noise_sigmas_nm"]]
    complete = smoke_epochs is None and smoke_frames is None and all(
        path.exists() and path.with_suffix(".json").exists() for path in expected
    )
    clean_audit = (
        clean_readout_audit(config, output_dir, train_full, validation_full, transform, blocks, center, scale)
        if complete
        else {
            "status": "partial or smoke training; clean readout deferred",
            "complete_checkpoint_count": sum(path.exists() and path.with_suffix(".json").exists() for path in expected),
            "expected_checkpoint_count": len(expected),
        }
    )
    audit_model = make_model(config, train_full, transform, blocks, center, scale)
    report = {
        "schema_version": 1,
        "created_at": utc_now(),
        "protocol": config["protocol_name"],
        "smoke_epochs": smoke_epochs,
        "smoke_frames": smoke_frames,
        "data": data_audit,
        "basis": basis_audit,
        "fit_families": sorted(map(int, np.unique(train_family))),
        "selection_families": sorted(map(int, np.unique(validation_family))),
        "fits": fits,
        "clean_readout": clean_audit,
        "architecture": audit_model.audit(),
        "force_or_component_labels_opened": False,
        "constant_law_or_variance_penalty_used": False,
    }
    atomic_json(output_dir / "summary.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke-epochs", type=int, default=None)
    parser.add_argument("--smoke-frames", type=int, default=None)
    parser.add_argument("--only-seed-index", type=int, default=None)
    parser.add_argument("--only-sigma", type=float, default=None)
    args = parser.parse_args()
    report = run(
        args.config,
        args.output.resolve(),
        args.smoke_epochs,
        args.smoke_frames,
        args.only_seed_index,
        args.only_sigma,
    )
    print(json.dumps({"summary": str((args.output.resolve() / "summary.json")), "clean_readout": report["clean_readout"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
