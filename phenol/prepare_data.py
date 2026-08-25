"""Prepare the released Phenol coordinate splits and physical holdout targets.

Raw production trajectories are generated with ``src/fresh_physical_families.py``.
This script then fixes the solute translation/periodic-image gauge.  In
``holdout`` mode it additionally evaluates the six frozen OpenMM force sectors;
those labels are never used by any training or projection script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from openmm import Context, Platform, Vec3, VerletIntegrator, unit
from openmm import HarmonicAngleForce, HarmonicBondForce, NonbondedForce, PeriodicTorsionForce


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import phenol_system  # noqa: E402


TRAINING_CONFIG = ROOT / "configs" / "fresh_physical_families.json"
HOLDOUT_CONFIG = ROOT / "phenol_dsm_controlled" / "configs" / "final_holdout_v1.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def spanning_tree(atom_count: int, bonds: Iterable[Iterable[int]]) -> list[tuple[int, int]]:
    neighbors: list[list[int]] = [[] for _ in range(atom_count)]
    for first, second in bonds:
        i, j = int(first), int(second)
        neighbors[i].append(j)
        neighbors[j].append(i)
    queue: deque[int] = deque([0])
    visited = {0}
    edges: list[tuple[int, int]] = []
    while queue:
        parent = queue.popleft()
        for child in sorted(neighbors[parent]):
            if child not in visited:
                visited.add(child)
                queue.append(child)
                edges.append((parent, child))
    if len(visited) != atom_count:
        raise RuntimeError("the molecular bond graph is disconnected")
    return edges


def unwrap_and_center(
    positions: np.ndarray, boxes: np.ndarray, bonds: list[list[int]]
) -> np.ndarray:
    wrapped = np.asarray(positions, dtype=np.float64)
    boxes = np.asarray(boxes, dtype=np.float64)
    if wrapped.ndim != 3 or wrapped.shape[1:] != (13, 3):
        raise ValueError(f"expected (frames,13,3), got {wrapped.shape}")
    if boxes.shape != (wrapped.shape[0], 3, 3):
        raise ValueError("box array shape mismatch")
    diagonal = np.diagonal(boxes, axis1=1, axis2=2)
    off_diagonal = boxes.copy()
    off_diagonal[:, np.arange(3), np.arange(3)] = 0.0
    if np.max(np.abs(off_diagonal)) > 1.0e-7 or np.min(diagonal) <= 0.0:
        raise RuntimeError("the released protocol requires positive orthorhombic boxes")
    output = np.empty_like(wrapped)
    output[:, 0] = wrapped[:, 0]
    for parent, child in spanning_tree(13, bonds):
        delta = wrapped[:, child] - wrapped[:, parent]
        delta -= np.rint(delta / diagonal) * diagonal
        output[:, child] = output[:, parent] + delta
    output -= output.mean(axis=1, keepdims=True)
    for first, second in bonds:
        length = np.linalg.norm(output[:, int(second)] - output[:, int(first)], axis=-1)
        if np.max(length) > 0.20:
            raise RuntimeError("periodic reconstruction produced a bond longer than 0.20 nm")
    return output.astype(np.float32)


def load_production(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        if sorted(values.files) != ["boxes_nm", "positions_nm"]:
            raise RuntimeError(f"unexpected arrays in {path}")
        return (
            np.asarray(values["positions_nm"], dtype=np.float32),
            np.asarray(values["boxes_nm"], dtype=np.float32),
        )


def prepare_training() -> Path:
    config = load_json(TRAINING_CONFIG)
    bonds = load_json(ROOT / "configs" / "structured_basis.json")["molecule"]["bonds"]
    source_root = ROOT / "data" / str(config["path_suffix"]) / "production"
    output_root = ROOT / "data" / str(config["path_suffix"]) / "model_pilot_v1"
    output_root.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {}
    for split in ("train", "validation"):
        coordinate_parts, family_parts = [], []
        for family in config["splits"][split]:
            positions, boxes = load_production(source_root / f"family_{int(family):02d}.npz")
            coordinate_parts.append(unwrap_and_center(positions[:, :13], boxes, bonds))
            family_parts.append(np.full(positions.shape[0], int(family), dtype=np.int16))
        coordinates = np.concatenate(coordinate_parts)
        families = np.concatenate(family_parts)
        path = output_root / f"{split}.npz"
        np.savez_compressed(path, phenol_coordinates_nm=coordinates, family_id=families)
        files[split] = {
            "path": path.name,
            "sha256": sha256_file(path),
            "frames": int(coordinates.shape[0]),
            "families": list(map(int, config["splits"][split])),
            "production_ns": float(len(config["splits"][split]))
            * float(config["sampling"]["production_picoseconds"])
            / 1000.0,
            "arrays": ["phenol_coordinates_nm", "family_id"],
        }
    manifest = {
        "schema_version": 1,
        "protocol_name": "phenol-coordinate-only-training-and-selection-v1",
        "coordinate_only": True,
        "q1_derived_coordinates_used": False,
        "dummy_atoms": 0,
        "periodic_unwrap": "bond-graph minimum-image spanning tree",
        "translation_gauge": "arithmetic center removed",
        "physical_system": "13-atom phenol in 512 rigid TIP3P waters; no dummy atom",
        "split_files": files,
    }
    path = output_root / "dataset_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def torsion_classification(
    torsion: PeriodicTorsionForce, bonds: list[list[int]]
) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]:
    edges = {tuple(sorted(map(int, edge))) for edge in bonds}
    proper, improper = [], []
    for index in range(torsion.getNumTorsions()):
        row = torsion.getTorsionParameters(index)
        atoms = tuple(map(int, row[:4]))
        path = all(
            tuple(sorted(edge)) in edges
            for edge in ((atoms[0], atoms[1]), (atoms[1], atoms[2]), (atoms[2], atoms[3]))
        )
        star = [
            center
            for center in atoms
            if all(tuple(sorted((center, other))) in edges for other in atoms if other != center)
        ]
        if path:
            proper.append(row)
        elif len(star) == 1:
            improper.append(row)
        else:
            raise RuntimeError(f"cannot classify torsion term {atoms}")
    if len(proper) != 26 or len(improper) != 6:
        raise RuntimeError("the phenol proper/improper term counts changed")
    return proper, improper


def force_group_system(config: Mapping[str, Any]) -> tuple[Any, dict[str, int]]:
    _, system, _, _ = phenol_system.build_physical_system(config)
    bonds = config["molecule"]["bonds"]
    torsion_index = next(
        index for index, force in enumerate(system.getForces()) if isinstance(force, PeriodicTorsionForce)
    )
    proper_rows, improper_rows = torsion_classification(system.getForce(torsion_index), bonds)
    system.removeForce(torsion_index)
    proper_force, improper_force = PeriodicTorsionForce(), PeriodicTorsionForce()
    for destination, rows in ((proper_force, proper_rows), (improper_force, improper_rows)):
        for row in rows:
            destination.addTorsion(*row)
    groups = {"bond": 1, "angle": 2, "proper": 3, "improper": 4, "nonbonded": 5}
    for force in system.getForces():
        if isinstance(force, HarmonicBondForce):
            force.setForceGroup(groups["bond"])
        elif isinstance(force, HarmonicAngleForce):
            force.setForceGroup(groups["angle"])
        elif isinstance(force, NonbondedForce):
            force.setForceGroup(groups["nonbonded"])
    proper_force.setForceGroup(groups["proper"])
    improper_force.setForceGroup(groups["improper"])
    system.addForce(proper_force)
    system.addForce(improper_force)
    return system, groups


def openmm_force_components(
    config: Mapping[str, Any], full_positions: np.ndarray, boxes: np.ndarray
) -> dict[str, np.ndarray]:
    system, groups = force_group_system(config)
    md_config = load_json(ROOT / str(config["md_config"]))
    runtime = md_config["runtime"]
    platform = Platform.getPlatformByName(str(runtime["platform"]))
    context = Context(
        system,
        VerletIntegrator(0.001),
        platform,
        {str(key): str(value) for key, value in runtime["platform_properties"].items()},
    )
    names = tuple(groups)
    collected: dict[str, list[np.ndarray]] = {name: [] for name in (*names, "total")}
    for positions, box in zip(full_positions, boxes):
        context.setPeriodicBoxVectors(*[Vec3(*map(float, row)) * unit.nanometer for row in box])
        context.setPositions(positions * unit.nanometer)
        for name in names:
            state = context.getState(getForces=True, groups={groups[name]})
            value = state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule_per_mole / unit.nanometer
            )
            collected[name].append(np.asarray(value[:13], dtype=np.float64))
        state = context.getState(getForces=True, groups=set(groups.values()))
        value = state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer
        )
        collected["total"].append(np.asarray(value[:13], dtype=np.float64))
    result = {name: np.asarray(values) for name, values in collected.items()}
    reconstruction = sum(result[name] for name in names)
    relative = np.sqrt(np.mean((reconstruction - result["total"]) ** 2)) / np.sqrt(
        np.mean(result["total"] ** 2)
    )
    if relative > 2.0e-5:
        raise RuntimeError(f"force-group reconstruction failed: {relative}")
    return result


def prepare_holdout() -> tuple[Path, Path]:
    config = load_json(HOLDOUT_CONFIG)
    md_config = load_json(ROOT / str(config["md_config"]))
    source_root = ROOT / "data" / str(md_config["path_suffix"]) / "production"
    coordinates, full_parts, box_parts, family_parts = [], [], [], []
    stride = int(config["selection"]["production_frame_stride"])
    for family in config["selection"]["families"]:
        positions, boxes = load_production(source_root / f"family_{int(family):02d}.npz")
        index = np.arange(0, positions.shape[0], stride)
        if index.size != int(config["selection"]["expected_frames_per_family"]):
            raise RuntimeError(f"unexpected selected-frame count in family {family}")
        selected_positions, selected_boxes = positions[index], boxes[index]
        coordinates.append(unwrap_and_center(selected_positions[:, :13], selected_boxes, config["molecule"]["bonds"]))
        full_parts.append(selected_positions)
        box_parts.append(selected_boxes)
        family_parts.append(np.full(index.size, int(family), dtype=np.int16))
    centered = np.concatenate(coordinates)
    full = np.concatenate(full_parts)
    boxes = np.concatenate(box_parts)
    families = np.concatenate(family_parts)
    output = ROOT / "phenol_dsm_controlled" / "results" / "final_holdout_v1"
    output.mkdir(parents=True, exist_ok=True)
    coordinate_path = output / "selected_coordinates.npz"
    np.savez_compressed(coordinate_path, phenol_coordinates_nm=centered, family_id=families)
    force_path = output / "raw_openmm_force_components.npz"
    np.savez_compressed(force_path, **openmm_force_components(config, full, boxes))
    return coordinate_path, force_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("training", "holdout"))
    args = parser.parse_args()
    if args.mode == "training":
        print(prepare_training())
    else:
        print("\n".join(map(str, prepare_holdout())))


if __name__ == "__main__":
    main()
