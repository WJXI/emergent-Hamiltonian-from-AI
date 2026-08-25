"""Lightweight diagnostics for numerical targets and the internal W representation.

This file is intentionally self-contained and does not modify the training code.
Run with the ai4s interpreter.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


SPIN_ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = SPIN_ROOT / "data" / "identifiability_500"
CHECKPOINT = SPIN_ROOT / "checkpoints" / "epoch_26.pt"
OUT = SPIN_ROOT / "results" / "identifiability.json"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import MiniAF3ScoreModel, sample_spherical_noise_and_target


def target_diagnostics(device: torch.device) -> dict:
    """Compare the implemented target to a stable exact log-map calculation."""
    torch.manual_seed(1729)
    batch, sites = 4096, 8
    clean = torch.randn(batch, sites, 3, device=device)
    clean = clean / torch.linalg.vector_norm(clean, dim=-1, keepdim=True)
    rows = []
    for time_value in [1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2]:
        time = torch.full((batch, sites, 1), time_value, device=device)
        noisy, target = sample_spherical_noise_and_target(clean, time)
        dot = torch.sum(clean * noisy, dim=-1, keepdim=True).clamp(-1.0, 1.0)
        tangent = clean - dot * noisy
        tangent_norm = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True)
        angle = torch.atan2(tangent_norm, dot)
        exact = (angle / tangent_norm.clamp_min(1.0e-12)) * tangent / time
        target_tangent = target - torch.sum(target * noisy, dim=-1, keepdim=True) * noisy
        target_radial = torch.sum(target * noisy, dim=-1)
        rel_tangent_error = torch.linalg.vector_norm(target_tangent - exact, dim=-1) / (
            torch.linalg.vector_norm(exact, dim=-1).clamp_min(1.0e-12)
        )
        weighted_radial_mse = time_value * torch.mean(target_radial**2)
        weighted_total_target_mse = time_value * torch.mean(torch.sum(target**2, dim=-1))
        rows.append(
            {
                "t": time_value,
                "mean_relative_tangent_error": float(rel_tangent_error.mean().item()),
                "p99_relative_tangent_error": float(
                    torch.quantile(rel_tangent_error, 0.99).item()
                ),
                "weighted_radial_mse": float(weighted_radial_mse.item()),
                "radial_fraction_of_target_energy": float(
                    (weighted_radial_mse / weighted_total_target_mse).item()
                ),
                "fraction_dot_above_0p9999": float((dot > 0.9999).float().mean().item()),
            }
        )
    return {"rows": rows}


def first_records(limit: int = 32):
    records = []
    for path in sorted(TEST_DIR.glob("chunk_*.npz")):
        with np.load(path, allow_pickle=True) as data:
            for sequence, snapshots in zip(data["sequences"], data["spins"]):
                for spins in snapshots:
                    records.append((str(sequence), np.asarray(spins, dtype=np.float32)))
                    if len(records) >= limit:
                        return records
    return records


def load_model(device: torch.device):
    payload = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    state = payload["model_state_dict"] if "model_state_dict" in payload else payload
    model = MiniAF3ScoreModel(c_s=64, c_z=32, num_blocks=4).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def symmetric_w(model, sequence, spins, mask, time):
    captured = {}

    def hook(_module, _inputs, output):
        captured["raw"] = output.detach()

    handle = model.weight_head.register_forward_hook(hook)
    with torch.inference_mode():
        score = model(sequence, spins, time, mask)
    handle.remove()
    raw = captured["raw"].squeeze(-1)
    weights = 0.5 * (raw + raw.transpose(1, 2))
    eye = torch.eye(weights.shape[1], device=weights.device).unsqueeze(0)
    weights = weights * (1.0 - eye) * mask.unsqueeze(1) * mask.unsqueeze(2)
    return weights, score


def representation_diagnostics(device: torch.device) -> dict:
    records = first_records(32)
    if not records:
        raise FileNotFoundError(TEST_DIR)
    size = len(records)
    sequence = torch.zeros(size, 50, dtype=torch.long, device=device)
    spins = torch.zeros(size, 50, 3, device=device)
    mask = torch.zeros(size, 50, dtype=torch.bool, device=device)
    lengths = []
    for row, (text, values) in enumerate(records):
        length = len(text)
        lengths.append(length)
        sequence[row, :length] = torch.tensor(
            [0 if token == "A" else 1 for token in text], device=device
        )
        spins[row, :length] = torch.as_tensor(values, device=device)
        mask[row, :length] = True
    time = torch.full((size, 1), 1.0e-4, device=device)
    model = load_model(device)
    weights, total_score = symmetric_w(model, sequence, spins, mask, time)

    distance_sums = {}
    distance_counts = {}
    support_w_sq = nonsupport_w_sq = 0.0
    support_score_sq = nonsupport_score_sq = total_score_sq = 0.0
    cross_score = 0.0
    oracle_score_sq = full_oracle_error_sq = support_oracle_error_sq = 0.0
    full_oracle_dot = support_oracle_dot = 0.0
    full_null_fractions = []
    support_null_fractions = []
    full_rank_fractions = []
    support_rank_fractions = []
    support_condition_numbers = []
    shared_six_gram = torch.zeros((6, 6), dtype=torch.float64, device=device)

    for row, length in enumerate(lengths):
        w = weights[row, :length, :length].double()
        s = spins[row, :length].double()
        positions = torch.arange(length, device=device)
        distance = torch.abs(positions[:, None] - positions[None, :])
        support = ((distance == 1) | (distance == 2)).double()
        nonsupport = ((distance > 2)).double()
        w_support = w * support
        w_nonsupport = w * nonsupport
        support_w_sq += float(torch.sum(w_support**2).item())
        nonsupport_w_sq += float(torch.sum(w_nonsupport**2).item())

        def score_from(matrix):
            raw = matrix @ s
            return raw - torch.sum(raw * s, dim=-1, keepdim=True) * s

        score_support = score_from(w_support)
        score_nonsupport = score_from(w_nonsupport)
        text = records[row][0]
        oracle_matrix = torch.zeros_like(w)
        for i in range(length - 1):
            pair = text[i] + text[i + 1]
            value = 1.0 if pair == "AA" else 0.5 if pair == "BB" else 0.8
            oracle_matrix[i, i + 1] = oracle_matrix[i + 1, i] = value
        for i in range(length - 2):
            pair = text[i] + text[i + 2]
            value = -0.45 if pair == "AA" else -0.25 if pair == "BB" else -0.35
            oracle_matrix[i, i + 2] = oracle_matrix[i + 2, i] = value
        oracle_score = 20.0 * score_from(oracle_matrix)
        full_score = score_support + score_nonsupport
        support_score_sq += float(torch.sum(score_support**2).item())
        nonsupport_score_sq += float(torch.sum(score_nonsupport**2).item())
        total_score_sq += float(torch.sum(full_score**2).item())
        cross_score += float(torch.sum(score_support * score_nonsupport).item())
        oracle_score_sq += float(torch.sum(oracle_score**2).item())
        full_oracle_error_sq += float(torch.sum((full_score - oracle_score) ** 2).item())
        support_oracle_error_sq += float(torch.sum((score_support - oracle_score) ** 2).item())
        full_oracle_dot += float(torch.sum(full_score * oracle_score).item())
        support_oracle_dot += float(torch.sum(score_support * oracle_score).item())

        for d in range(1, length):
            values = w[distance == d]
            distance_sums[d] = distance_sums.get(d, 0.0) + float(torch.sum(values**2).item())
            distance_counts[d] = distance_counts.get(d, 0) + int(values.numel())

        # Row-wise null energy: only two tangent components per row can affect score.
        for index in range(length):
            projector = torch.eye(3, dtype=torch.float64, device=device) - torch.outer(
                s[index], s[index]
            )
            design = projector @ s.T  # 3 x N, rank at most 2
            coeff = w[index]
            identifiable = torch.linalg.pinv(design) @ (design @ coeff)
            full_null_fractions.append(
                float(torch.sum((coeff - identifiable) ** 2).item() / torch.sum(coeff**2).clamp_min(1e-30).item())
            )
            coeff_s = w_support[index]
            identifiable_s = torch.linalg.pinv(design * support[index].unsqueeze(0)) @ (
                (design * support[index].unsqueeze(0)) @ coeff_s
            )
            denom_s = torch.sum(coeff_s**2).clamp_min(1e-30)
            support_null_fractions.append(
                float(torch.sum((coeff_s - identifiable_s) ** 2).item() / denom_s.item())
            )

        # Exact dimension count/rank for symmetric edge-to-tangent-score maps.
        def map_rank(edge_pairs):
            matrix = torch.zeros((3 * length, len(edge_pairs)), dtype=torch.float64, device=device)
            for col, (i, j) in enumerate(edge_pairs):
                pi = torch.eye(3, dtype=torch.float64, device=device) - torch.outer(s[i], s[i])
                pj = torch.eye(3, dtype=torch.float64, device=device) - torch.outer(s[j], s[j])
                matrix[3 * i : 3 * i + 3, col] = pi @ s[j]
                matrix[3 * j : 3 * j + 3, col] = pj @ s[i]
            singular = torch.linalg.svdvals(matrix)
            tolerance = singular[0] * max(matrix.shape) * torch.finfo(singular.dtype).eps
            nonzero = singular[singular > tolerance]
            rank = int(nonzero.numel())
            condition = float((nonzero[0] / nonzero[-1]).item())
            return rank, len(edge_pairs), condition

        full_edges = [(i, j) for i in range(length) for j in range(i + 1, length)]
        support_edges = [(i, j) for i, j in full_edges if j - i <= 2]
        rank, edges, _condition = map_rank(full_edges)
        rank_s, edges_s, condition_s = map_rank(support_edges)
        full_rank_fractions.append(rank / edges)
        support_rank_fractions.append(rank_s / edges_s)
        support_condition_numbers.append(condition_s)

        # If W is tied to the six (distance, species-pair) classes, the global
        # score design has only six columns.  Accumulate its Gram matrix without J.
        class_columns = []
        for class_index in range(6):
            class_matrix = torch.zeros_like(w)
            for i, j in support_edges:
                if text[i] == text[j] == "A":
                    pair_index = 0
                elif text[i] == text[j] == "B":
                    pair_index = 1
                else:
                    pair_index = 2
                edge_class = pair_index if j - i == 1 else pair_index + 3
                if edge_class == class_index:
                    class_matrix[i, j] = class_matrix[j, i] = 1.0
            class_columns.append(score_from(class_matrix).reshape(-1))
        class_design = torch.stack(class_columns, dim=1)
        shared_six_gram += class_design.T @ class_design

    by_distance = [
        {
            "distance": int(d),
            "rms_W": math.sqrt(distance_sums[d] / distance_counts[d]),
            "count": distance_counts[d],
        }
        for d in sorted(distance_sums)
    ]
    six_eigenvalues, six_eigenvectors = torch.linalg.eigh(shared_six_gram)
    six_condition = float(torch.sqrt(six_eigenvalues[-1] / six_eigenvalues[0]).item())
    weakest_mode = six_eigenvectors[:, 0]
    weakest_mode = weakest_mode / torch.max(torch.abs(weakest_mode))
    return {
        "checkpoint": str(CHECKPOINT),
        "configurations": len(records),
        "support_W_energy_fraction": support_w_sq / (support_w_sq + nonsupport_w_sq),
        "nonsupport_W_energy_fraction": nonsupport_w_sq / (support_w_sq + nonsupport_w_sq),
        "support_score_energy_over_total": support_score_sq / total_score_sq,
        "nonsupport_score_energy_over_total": nonsupport_score_sq / total_score_sq,
        "support_nonsupport_cross_over_total": 2.0 * cross_score / total_score_sq,
        "full_score_relative_rmse_to_beta20_oracle": math.sqrt(
            full_oracle_error_sq / oracle_score_sq
        ),
        "support_only_score_relative_rmse_to_beta20_oracle": math.sqrt(
            support_oracle_error_sq / oracle_score_sq
        ),
        "full_score_scale_against_beta20_oracle": full_oracle_dot / oracle_score_sq,
        "support_only_score_scale_against_beta20_oracle": support_oracle_dot
        / oracle_score_sq,
        "mean_rowwise_full_W_null_energy_fraction": float(np.mean(full_null_fractions)),
        "mean_rowwise_support_W_null_energy_fraction": float(np.mean(support_null_fractions)),
        "mean_full_symmetric_map_rank_fraction": float(np.mean(full_rank_fractions)),
        "mean_support_symmetric_map_rank_fraction": float(np.mean(support_rank_fractions)),
        "median_support_map_condition_number_per_configuration": float(
            np.median(support_condition_numbers)
        ),
        "max_support_map_condition_number_per_configuration": float(
            np.max(support_condition_numbers)
        ),
        "shared_six_class_design_condition_number": six_condition,
        "shared_six_class_gram_eigenvalues": six_eigenvalues.detach().cpu().tolist(),
        "shared_six_class_gram": shared_six_gram.detach().cpu().tolist(),
        "shared_six_class_weakest_mode_AA_BB_AB_d1_d2": weakest_mode.detach()
        .cpu()
        .tolist(),
        "by_distance": by_distance,
    }


def model_diagnostics() -> dict:
    model = MiniAF3ScoreModel(c_s=64, c_z=32, num_blocks=4)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    return {
        "parameters": parameters,
        "raw_time_embedding": True,
        "absolute_position_embedding": True,
        "weight_head_has_spin_inner_product_input": True,
        "hard_locality_mask_before_score_aggregation": False,
        "direct_W_supervision": False,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = {
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__, "device": str(device)},
        "model": model_diagnostics(),
        "target": target_diagnostics(device),
        "representation": representation_diagnostics(device),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
