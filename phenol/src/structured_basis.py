"""Fixed, force-free geometric basis used by Phenol Networks 2 and 3.

Only solute coordinates, element identities, and the molecular bond graph are
used to construct the basis.  No force-field energies, forces, parameters, or
OpenMM component labels enter this module.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn


PHENOL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PHENOL_ROOT / "configs" / "structured_basis.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_protocol(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_json(path)
    required_blocks = ["bond", "angle", "proper", "out_of_plane", "graph_distance_3", "graph_distance_gt_3"]
    if config["dictionary"]["blocks"] != required_blocks:
        raise RuntimeError("the preregistered semantic block list changed")
    guardrails = config["guardrails"]
    if not bool(guardrails["coordinate_arrays_only"]):
        raise RuntimeError("coordinate-only firewall disabled")
    forbidden = (
        "energy_labels_used_for_basis",
        "force_labels_used_for_basis",
        "component_labels_used_for_basis",
        "force_field_parameters_used_for_basis",
        "event_or_atom_name_embeddings",
    )
    if any(bool(guardrails[key]) for key in forbidden):
        raise RuntimeError("a forbidden oracle input was enabled")
    return config


def graph_data(num_atoms: int, bonds: Iterable[Iterable[int]]) -> dict[str, Any]:
    canonical_bonds = sorted({tuple(sorted((int(i), int(j)))) for i, j in bonds})
    neighbors: list[list[int]] = [[] for _ in range(num_atoms)]
    graph_distance = np.full((num_atoms, num_atoms), 999, dtype=np.int64)
    np.fill_diagonal(graph_distance, 0)
    for i, j in canonical_bonds:
        neighbors[i].append(j)
        neighbors[j].append(i)
        graph_distance[i, j] = graph_distance[j, i] = 1
    for middle in range(num_atoms):
        graph_distance = np.minimum(graph_distance, graph_distance[:, [middle]] + graph_distance[[middle], :])
    angles: list[tuple[int, int, int]] = []
    for center in range(num_atoms):
        for first, third in combinations(sorted(neighbors[center]), 2):
            angles.append((first, center, third))
    proper: set[tuple[int, int, int, int]] = set()
    for middle_first, middle_second in canonical_bonds:
        for first in neighbors[middle_first]:
            if first == middle_second:
                continue
            for fourth in neighbors[middle_second]:
                if fourth in (middle_first, first):
                    continue
                candidate = (first, middle_first, middle_second, fourth)
                proper.add(min(candidate, tuple(reversed(candidate))))
    out_of_plane: list[tuple[int, int, int, int]] = []
    for center, adjacent in enumerate(neighbors):
        for triple in combinations(sorted(adjacent), 3):
            out_of_plane.append((center, *triple))
    pairs_d3 = []
    pairs_gt3 = []
    for first in range(num_atoms):
        for second in range(first + 1, num_atoms):
            distance = int(graph_distance[first, second])
            if distance == 3:
                pairs_d3.append((first, second))
            elif distance > 3:
                pairs_gt3.append((first, second))
    return {
        "bonds": canonical_bonds,
        "neighbors": [sorted(row) for row in neighbors],
        "graph_distance": graph_distance,
        "angle": sorted(angles),
        "proper": sorted(proper),
        "out_of_plane": sorted(out_of_plane),
        "graph_distance_3": pairs_d3,
        "graph_distance_gt_3": pairs_gt3,
    }


def node_types(elements: Sequence[str], neighbors: Sequence[Sequence[int]]) -> list[str]:
    output = []
    for atom, element in enumerate(elements):
        neighbor_elements = ",".join(sorted(str(elements[index]) for index in neighbors[atom]))
        output.append(f"{element}|deg{len(neighbors[atom])}|nbr[{neighbor_elements}]")
    return output


def canonical_sequence(values: Sequence[str]) -> tuple[str, ...]:
    forward = tuple(str(x) for x in values)
    reverse = tuple(reversed(forward))
    return min(forward, reverse)


def group_events(config: Mapping[str, Any]) -> tuple[dict[str, dict[str, list[tuple[int, ...]]]], dict[str, Any]]:
    elements = [str(x) for x in config["molecule"]["elements"]]
    topology = graph_data(int(config["molecule"]["atoms"]), config["molecule"]["bonds"])
    types = node_types(elements, topology["neighbors"])
    grouped: dict[str, dict[str, list[tuple[int, ...]]]] = {name: defaultdict(list) for name in config["dictionary"]["blocks"]}
    for event in topology["bonds"]:
        event_type = "--".join(sorted((types[event[0]], types[event[1]])))
        grouped["bond"][event_type].append(event)
    for event in topology["angle"]:
        endpoint = sorted((types[event[0]], types[event[2]]))
        event_type = f"center({types[event[1]]})|ends({endpoint[0]};{endpoint[1]})"
        grouped["angle"][event_type].append(event)
    for event in topology["proper"]:
        event_type = "--".join(canonical_sequence([types[index] for index in event]))
        grouped["proper"][event_type].append(event)
    for event in topology["out_of_plane"]:
        center, *adjacent = event
        event_type = f"center({types[center]})|nbrs({';'.join(sorted(types[index] for index in adjacent))})"
        grouped["out_of_plane"][event_type].append(event)
    for block in ("graph_distance_3", "graph_distance_gt_3"):
        for event in topology[block]:
            event_type = "--".join(sorted((types[event[0]], types[event[1]])))
            grouped[block][event_type].append(event)
    grouped_plain = {block: {key: sorted(value) for key, value in sorted(groups.items())} for block, groups in grouped.items()}
    audit = {
        "node_type_rule": config["dictionary"]["node_type_rule"],
        "node_type_counts": dict(sorted((key, types.count(key)) for key in set(types))),
        "event_counts": {
            "bond": len(topology["bonds"]),
            "angle": len(topology["angle"]),
            "proper": len(topology["proper"]),
            "out_of_plane": len(topology["out_of_plane"]),
            "graph_distance_3": len(topology["graph_distance_3"]),
            "graph_distance_gt_3": len(topology["graph_distance_gt_3"]),
        },
        "event_type_counts": {block: len(groups) for block, groups in grouped_plain.items()},
        "atom_names_used": False,
        "atom_indices_used_as_learned_identity": False,
        "openmm_term_lists_used": False,
    }
    return grouped_plain, audit


def distance_coordinate(x: np.ndarray, events: Sequence[Sequence[int]]) -> np.ndarray:
    index = np.asarray(events, dtype=np.int64)
    return np.linalg.norm(x[:, index[:, 1]] - x[:, index[:, 0]], axis=-1)


def angle_coordinate(x: np.ndarray, events: Sequence[Sequence[int]]) -> np.ndarray:
    index = np.asarray(events, dtype=np.int64)
    left = x[:, index[:, 0]] - x[:, index[:, 1]]
    right = x[:, index[:, 2]] - x[:, index[:, 1]]
    cosine = np.sum(left * right, axis=-1) / np.maximum(np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1), 1.0e-12)
    return np.arccos(np.clip(cosine, -1.0 + 1.0e-10, 1.0 - 1.0e-10))


def dihedral_coordinate_np(x: np.ndarray, events: Sequence[Sequence[int]]) -> np.ndarray:
    index = np.asarray(events, dtype=np.int64)
    p0, p1, p2, p3 = (x[:, index[:, column]] for column in range(4))
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1 = b1 / np.maximum(np.linalg.norm(b1, axis=-1, keepdims=True), 1.0e-12)
    v = b0 - np.sum(b0 * b1, axis=-1, keepdims=True) * b1
    w = b2 - np.sum(b2 * b1, axis=-1, keepdims=True) * b1
    return np.arctan2(np.sum(np.cross(b1, v, axis=-1) * w, axis=-1), np.sum(v * w, axis=-1))


def out_of_plane_coordinate_np(x: np.ndarray, events: Sequence[Sequence[int]]) -> np.ndarray:
    index = np.asarray(events, dtype=np.int64)
    center = x[:, index[:, 0]]
    vectors = [x[:, index[:, column]] - center for column in (1, 2, 3)]
    unit = [value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1.0e-12) for value in vectors]
    triple = np.sum(unit[0] * np.cross(unit[1], unit[2], axis=-1), axis=-1)
    return np.square(triple)


def robust_location_scale(values: np.ndarray, minimum_scale: float) -> tuple[float, float, dict[str, float]]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    median = float(np.median(flat))
    q25, q75 = np.quantile(flat, [0.25, 0.75])
    robust = float((q75 - q25) / 1.349)
    standard = float(np.std(flat))
    scale = max(robust, standard * 0.25, float(minimum_scale))
    return median, scale, {
        "median": median,
        "scale": scale,
        "q01": float(np.quantile(flat, 0.01)),
        "q25": float(q25),
        "q75": float(q75),
        "q99": float(np.quantile(flat, 0.99)),
        "standard_deviation": standard,
        "observations": int(flat.size),
    }


@dataclass(frozen=True)
class DictionaryGroup:
    block: str
    event_type: str
    kind: str
    events: tuple[tuple[int, ...], ...]
    center: float
    scale: float
    basis_labels: tuple[str, ...]


def build_dictionary_groups(config: Mapping[str, Any], train: np.ndarray) -> tuple[list[DictionaryGroup], dict[str, Any]]:
    grouped, topology_audit = group_events(config)
    control = config.get("control")
    if control is not None:
        permutation = [int(x) for x in control["event_node_permutation"]]
        if sorted(permutation) != list(range(int(config["molecule"]["atoms"]))):
            raise RuntimeError("control node permutation is invalid")
        grouped = {
            block: {
                event_type: [tuple(permutation[index] for index in event) for event in events]
                for event_type, events in groups.items()
            }
            for block, groups in grouped.items()
        }
        topology_audit["control"] = {
            "kind": control["kind"],
            "event_node_permutation": permutation,
            "same_event_groups_and_basis_dimension_as_true_topology": True,
            "training_coordinates_used_to_recompute_control_center_and_scale": True,
        }
    centers = [float(x) for x in config["dictionary"]["radial_and_angle_basis"]["gaussian_centers_standardized"]]
    generic_labels = ("linear", "quadratic", *(f"rbf_z{value:+.1f}" for value in centers))
    fourier_orders = [int(x) for x in config["dictionary"]["proper_fourier_orders"]]
    groups: list[DictionaryGroup] = []
    stats: list[dict[str, Any]] = []
    coordinate_functions = {
        "bond": (distance_coordinate, 1.0e-4, "radial"),
        "angle": (angle_coordinate, 1.0e-3, "angle"),
        "out_of_plane": (out_of_plane_coordinate_np, 1.0e-5, "out_of_plane_squared"),
        "graph_distance_3": (distance_coordinate, 1.0e-4, "radial"),
        "graph_distance_gt_3": (distance_coordinate, 1.0e-4, "radial"),
    }
    pair_basis = config["dictionary"].get("pair_basis", {})
    pair_physics_enabled = pair_basis.get("mode") == "dimensionless_inverse_power"
    for block in config["dictionary"]["blocks"]:
        for event_type, events_list in grouped[block].items():
            events = tuple(tuple(int(value) for value in event) for event in events_list)
            if block == "proper":
                coordinate = dihedral_coordinate_np(train, events)
                labels = tuple(label for order in fourier_orders for label in (f"sin{order}", f"cos{order}"))
                group = DictionaryGroup(block, event_type, "proper", events, 0.0, 1.0, labels)
                group_stat = {
                    "block": block,
                    "event_type": event_type,
                    "events": len(events),
                    "coordinate": "dihedral_radians",
                    "q01": float(np.quantile(coordinate, 0.01)),
                    "q99": float(np.quantile(coordinate, 0.99)),
                    "circular_resultant": float(abs(np.mean(np.exp(1j * coordinate)))),
                    "observations": int(coordinate.size),
                }
            else:
                function, minimum, kind = coordinate_functions[block]
                coordinate = function(train, events)
                center, scale, summary = robust_location_scale(coordinate, minimum)
                if pair_physics_enabled and block in ("graph_distance_3", "graph_distance_gt_3"):
                    powers = tuple(int(value) for value in pair_basis["inverse_powers"])
                    labels = tuple(f"inverse_r{power}" for power in powers)
                    kind = "pair_inverse_power"
                else:
                    labels = tuple(generic_labels)
                group = DictionaryGroup(block, event_type, kind, events, center, scale, labels)
                group_stat = {"block": block, "event_type": event_type, "events": len(events), "coordinate": kind, **summary}
            groups.append(group)
            stats.append(group_stat)
    audit = {
        **topology_audit,
        "groups": stats,
        "basis_functions_before_rank_truncation": int(sum(len(group.basis_labels) for group in groups)),
        "train_frames_used_for_shared_type_statistics": int(train.shape[0]),
        "per_event_statistics_used": False,
        "force_or_energy_information_used": False,
        "pair_basis": pair_basis if pair_physics_enabled else {"mode": "generic_train_standardized_rbf"},
    }
    return groups, audit


class GeometryDictionary(nn.Module):
    """Generic scalar geometry dictionary with no trainable parameters."""

    def __init__(self, groups: Sequence[DictionaryGroup], gaussian_centers: Sequence[float], gaussian_width: float, fourier_orders: Sequence[int]) -> None:
        super().__init__()
        self.groups = list(groups)
        self.gaussian_centers = tuple(float(x) for x in gaussian_centers)
        self.gaussian_width = float(gaussian_width)
        self.fourier_orders = tuple(int(x) for x in fourier_orders)
        self._event_names: list[str] = []
        for number, group in enumerate(self.groups):
            name = f"events_{number:03d}"
            self.register_buffer(name, torch.as_tensor(group.events, dtype=torch.long))
            self._event_names.append(name)
        basis_block: list[str] = []
        basis_type: list[str] = []
        basis_label: list[str] = []
        for group in self.groups:
            basis_block.extend([group.block] * len(group.basis_labels))
            basis_type.extend([group.event_type] * len(group.basis_labels))
            basis_label.extend(group.basis_labels)
        self.basis_block = tuple(basis_block)
        self.basis_type = tuple(basis_type)
        self.basis_label = tuple(basis_label)

    @property
    def basis_count(self) -> int:
        return len(self.basis_block)

    def _distance(self, x: Tensor, index: Tensor) -> Tensor:
        return torch.linalg.vector_norm(x[:, index[:, 1]] - x[:, index[:, 0]], dim=-1).clamp_min(1.0e-10)

    def _angle(self, x: Tensor, index: Tensor) -> Tensor:
        left = x[:, index[:, 0]] - x[:, index[:, 1]]
        right = x[:, index[:, 2]] - x[:, index[:, 1]]
        cosine = (left * right).sum(dim=-1) / (torch.linalg.vector_norm(left, dim=-1) * torch.linalg.vector_norm(right, dim=-1)).clamp_min(1.0e-12)
        return torch.acos(cosine.clamp(-1.0 + 1.0e-10, 1.0 - 1.0e-10))

    def _proper(self, x: Tensor, index: Tensor) -> Tensor:
        p0, p1, p2, p3 = (x[:, index[:, column]] for column in range(4))
        b0 = p0 - p1
        b1 = p2 - p1
        b2 = p3 - p2
        b1 = b1 / torch.linalg.vector_norm(b1, dim=-1, keepdim=True).clamp_min(1.0e-12)
        v = b0 - (b0 * b1).sum(dim=-1, keepdim=True) * b1
        w = b2 - (b2 * b1).sum(dim=-1, keepdim=True) * b1
        return torch.atan2((torch.cross(b1, v, dim=-1) * w).sum(dim=-1), (v * w).sum(dim=-1))

    def _out_of_plane_squared(self, x: Tensor, index: Tensor) -> Tensor:
        center = x[:, index[:, 0]]
        vectors = [x[:, index[:, column]] - center for column in (1, 2, 3)]
        unit = [value / torch.linalg.vector_norm(value, dim=-1, keepdim=True).clamp_min(1.0e-12) for value in vectors]
        triple = (unit[0] * torch.cross(unit[1], unit[2], dim=-1)).sum(dim=-1)
        return triple.square()

    def _generic_features(self, coordinate: Tensor, center: float, scale: float) -> Tensor:
        z = (coordinate - float(center)) / float(scale)
        features = [z, z.square()]
        for rbf_center in self.gaussian_centers:
            features.append(torch.exp(-0.5 * ((z - rbf_center) / self.gaussian_width).square()))
        return torch.stack(features, dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        pieces = []
        for group, event_name in zip(self.groups, self._event_names):
            index = getattr(self, event_name)
            if group.kind == "radial":
                coordinate = self._distance(x, index)
                features = self._generic_features(coordinate, group.center, group.scale)
            elif group.kind == "angle":
                coordinate = self._angle(x, index)
                features = self._generic_features(coordinate, group.center, group.scale)
            elif group.kind == "out_of_plane_squared":
                coordinate = self._out_of_plane_squared(x, index)
                features = self._generic_features(coordinate, group.center, group.scale)
            elif group.kind == "proper":
                phi = self._proper(x, index)
                feature_list = []
                for order in self.fourier_orders:
                    feature_list.extend((torch.sin(order * phi), torch.cos(order * phi)))
                features = torch.stack(feature_list, dim=-1)
            elif group.kind == "pair_inverse_power":
                coordinate = self._distance(x, index)
                ratio = float(group.center) / coordinate
                powers = [int(label.removeprefix("inverse_r")) for label in group.basis_labels]
                features = torch.stack([ratio.pow(power) for power in powers], dim=-1)
            else:
                raise RuntimeError(f"unknown dictionary kind {group.kind}")
            pieces.append(features.sum(dim=1))
        return torch.cat(pieces, dim=-1)

    def metadata(self) -> list[dict[str, Any]]:
        return [
            {"index": index, "block": block, "event_type": event_type, "basis": label}
            for index, (block, event_type, label) in enumerate(zip(self.basis_block, self.basis_type, self.basis_label))
        ]


def linalg_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def torch_eigh_numpy(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tensor = torch.as_tensor(np.asarray(matrix), dtype=torch.float64, device=linalg_device())
    values, vectors = torch.linalg.eigh(tensor)
    return values.detach().cpu().numpy(), vectors.detach().cpu().numpy()


def torch_matmul_numpy(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_tensor = torch.as_tensor(np.asarray(left), dtype=torch.float64, device=linalg_device())
    right_tensor = torch.as_tensor(np.asarray(right), dtype=torch.float64, device=linalg_device())
    return torch.matmul(left_tensor, right_tensor).detach().cpu().numpy()


def dot_numpy(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sum(np.asarray(left) * np.asarray(right)))


def quadratic_numpy(left: np.ndarray, matrix: np.ndarray, right: np.ndarray | None = None) -> float:
    if right is None:
        right = left
    return dot_numpy(left, torch_matmul_numpy(matrix, right))
