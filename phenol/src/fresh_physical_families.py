"""Generate independent explicit-water Phenol families used in the paper."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from openmm import Vec3, unit

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "fresh_physical_families.json"
ACTIVE_CONFIG_PATH = CONFIG_PATH
SOURCE_PATH = Path(__file__).resolve()
BASE_SOURCE = ROOT / "src" / "md_utils.py"
import md_utils as base


def repository_root() -> Path:
    return ROOT


def ascii_repository_root() -> Path:
    return ROOT


def _pdb_atom(
    serial: int,
    name: str,
    residue: str,
    residue_index: int,
    xyz_angstrom: np.ndarray,
    symbol: str,
) -> str:
    x, y, z = map(float, xyz_angstrom)
    return (
        f"HETATM{serial:5d} {name:<4s} {residue:>3s} A"
        f"{residue_index:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
        f"  1.00  0.00          {symbol:>2s}\n"
    )


def _water_template() -> tuple[str, np.ndarray]:
    oh = 0.09572
    hh = 0.15139006545247014
    cosine = (2.0 * oh * oh - hh * hh) / (2.0 * oh * oh)
    coordinates = np.asarray(
        [[0.0, 0.0, 0.0], [oh, 0.0, 0.0], [oh * cosine, oh * np.sqrt(1.0 - cosine**2), 0.0]]
    )
    text = "".join(
        _pdb_atom(
            index + 1,
            ("O", "H1", "H2")[index],
            "HOH",
            2,
            10.0 * coordinates[index],
            ("O", "H", "H")[index],
        )
        for index in range(3)
    ) + "END\n"
    return text, coordinates


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def load_config(config_path: Path = CONFIG_PATH, *, smoke: bool = False) -> dict[str, Any]:
    global ACTIVE_CONFIG_PATH
    ACTIVE_CONFIG_PATH = Path(config_path).resolve()
    config = json.loads(ACTIVE_CONFIG_PATH.read_text(encoding="utf-8"))
    if smoke:
        config = deepcopy(config)
        config["protocol_name"] += "-smoke"
        config["path_suffix"] = "fresh_physical_smoke"
        config["sampling"]["families"] = 1
        config["sampling"]["velocity_seeds"] = config["sampling"]["velocity_seeds"][:1]
        config["sampling"]["equilibration_picoseconds"] = 2.0
        config["sampling"]["equilibration_diagnostic_interval_picoseconds"] = 1.0
        config["sampling"]["production_picoseconds"] = 2.0
        config["sampling"]["production_sample_interval_picoseconds"] = 1.0
        config["packing"]["packmol_seeds"] = config["packing"]["packmol_seeds"][:1]
        config["packing"]["oh_template_by_family"] = config["packing"][
            "oh_template_by_family"
        ][:1]
    return config


def result_root(config: Mapping[str, Any]) -> Path:
    return ROOT / "results" / str(config["path_suffix"])


def data_root(config: Mapping[str, Any]) -> Path:
    return ROOT / "data" / str(config["path_suffix"])


def raw_packing_path(config: Mapping[str, Any], family: int) -> Path:
    return data_root(config) / "packings" / "raw" / f"family_{family:02d}.npz"


def minimized_packing_path(config: Mapping[str, Any], family: int) -> Path:
    return data_root(config) / "packings" / "minimized" / f"family_{family:02d}.npz"


def production_path(config: Mapping[str, Any], family: int) -> Path:
    return data_root(config) / "production" / f"family_{family:02d}.npz"


def diagnostic_path(config: Mapping[str, Any], family: int) -> Path:
    return data_root(config) / "diagnostics" / f"family_{family:02d}.npz"


def packmol_paths() -> tuple[Path, Path]:
    explicit = os.environ.get("PACKMOL_EXECUTABLE")
    executable = Path(explicit).resolve() if explicit else None
    if executable is None:
        discovered = shutil.which("packmol") or shutil.which("packmol.exe")
        executable = Path(discovered).resolve() if discovered else None
    if executable is None or not executable.is_file():
        raise FileNotFoundError(
            "Packmol was not found. Install Packmol 21.2.3 or set PACKMOL_EXECUTABLE."
        )
    runtime = Path(os.environ.get("PACKMOL_RUNTIME_BIN", str(executable.parent))).resolve()
    return executable, runtime


def validate(config: Mapping[str, Any]) -> None:
    families = int(config["sampling"]["families"])
    packing = config["packing"]
    if int(config["system"]["phenol_atoms"]) != 13:
        raise RuntimeError("fresh physical ligand atom count changed")
    if int(config["system"]["water_count"]) != 512:
        raise RuntimeError("fresh physical water count changed")
    if len(packing["packmol_seeds"]) != families:
        raise RuntimeError("Packmol seed count mismatch")
    if len(config["sampling"]["velocity_seeds"]) != families:
        raise RuntimeError("velocity seed count mismatch")
    if len(set(map(int, packing["packmol_seeds"]))) != families:
        raise RuntimeError("Packmol seeds are not unique")
    if len(set(map(int, config["sampling"]["velocity_seeds"]))) != families:
        raise RuntimeError("velocity seeds are not unique")
    expected_templates = ["plus" if family % 2 == 0 else "minus" for family in range(families)]
    if packing["oh_template_by_family"] != expected_templates:
        raise RuntimeError("OH template balance changed")
    expected_firewall = {
        "q1_derived_coordinates_forbidden": True,
        "all_families_independent_packmol": True,
        "production_arrays_coordinate_only": True,
        "diagnostic_energies_for_training_forbidden": True,
        "training_requires_fresh_family_qualification_pass": True,
    }
    if config["data_firewall"] != expected_firewall:
        raise RuntimeError("fresh-family firewall changed")
    if str(config.get("purpose", "model_pilot")) == "final_holdout":
        expected_holdout = {
            "model_protocol_frozen_before_packmol": True,
            "all_model_checkpoints_frozen_before_packmol": True,
            "holdout_used_for_training": False,
            "holdout_used_for_checkpoint_selection": False,
            "holdout_used_for_hyperparameter_selection": False,
            "one_shot_evaluation_only": True,
        }
        if config.get("holdout_firewall") != expected_holdout:
            raise RuntimeError("final-holdout firewall changed")
        if sorted(config["splits"]) != ["final_holdout"]:
            raise RuntimeError("final holdout may not be relabeled as a training split")


def physical_templates(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    _, _, positions, _ = base.build_physical_system(config)
    centered = np.asarray(positions, dtype=np.float64) - np.mean(positions[:6], axis=0)
    c4, oxygen, hydrogen = centered[3], centered[6], centered[12]
    axis = oxygen - c4
    axis /= np.sqrt(np.sum(axis * axis))
    vector = hydrogen - oxygen
    mirrored = centered.copy()
    mirrored[12] = oxygen + 2.0 * float(np.dot(axis, vector)) * axis - vector
    result = {"plus": centered, "minus": mirrored}
    q_values = {}
    box = np.eye(3) * float(config["system"]["box_length_nm"])
    for name, value in result.items():
        pair = base.ligand_pair_distances(value, box)
        pairs = [tuple(map(int, item)) for item in base.pair_indices(13)]
        lookup = {item: index for index, item in enumerate(pairs)}
        q_values[name] = float(pair[lookup[(2, 12)]] - pair[lookup[(4, 12)]])
    if q_values["plus"] * q_values["minus"] >= 0.0:
        raise RuntimeError("mirrored OH templates do not have opposite orientation")
    if not np.isclose(abs(q_values["plus"]), abs(q_values["minus"]), rtol=0.02, atol=1.0e-4):
        raise RuntimeError("OH template symmetry audit failed")
    return result


def ligand_pdb_text(config: Mapping[str, Any], template: str) -> str:
    topology, _, _, metadata = base.build_physical_system(config)
    atoms = list(topology.atoms())[:13]
    positions = physical_templates(config)[template]
    return "".join(
        _pdb_atom(
            index + 1,
            atom.name,
            "PHN",
            1,
            10.0 * positions[index],
            atom.element.symbol,
        )
        for index, atom in enumerate(atoms)
    ) + "END\n"


def packmol_input(config: Mapping[str, Any], seed: int) -> str:
    half = 5.0 * float(config["system"]["box_length_nm"])
    bounds = f"{-half:.6f} {-half:.6f} {-half:.6f} {half:.6f} {half:.6f} {half:.6f}"
    packing = config["packing"]
    return (
        f"tolerance {float(packing['tolerance_angstrom']):.6f}\n"
        "filetype pdb\n"
        "output packed.pdb\n"
        f"seed {int(seed)}\n"
        f"precision {float(packing['precision']):.6f}\n"
        f"nloop {int(packing['nloop'])}\n"
        "avoid_overlap yes\n"
        "randominitialpoint\n"
        f"pbc {bounds}\n\n"
        "structure ligand.pdb\n"
        "  number 1\n"
        "  fixed 0. 0. 0. 0. 0. 0.\n"
        "end structure\n\n"
        "structure water.pdb\n"
        f"  number {int(config['system']['water_count'])}\n"
        "  resnumbers 3\n"
        "end structure\n"
    )


def parse_packed(path: Path, water_count: int) -> np.ndarray:
    positions, names, residues = [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            names.append(line[12:16].strip())
            residues.append(line[17:20].strip())
            positions.append(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            )
    expected = 13 + 3 * int(water_count)
    if len(positions) != expected:
        raise RuntimeError("fresh Packmol atom-count contract failed")
    if residues[:13] != ["PHN"] * 13:
        raise RuntimeError("fresh Packmol ligand order changed")
    for start in range(13, expected, 3):
        if names[start : start + 3] != ["O", "H1", "H2"]:
            raise RuntimeError("fresh Packmol water order changed")
        if residues[start : start + 3] != ["HOH"] * 3:
            raise RuntimeError("fresh Packmol water residue changed")
    return np.asarray(positions, dtype=np.float64) / 10.0


def packing_diagnostics(positions: np.ndarray, length: float) -> dict[str, float]:
    ligand = positions[:13]
    waters = positions[13:].reshape(-1, 3, 3)
    oxygen = waters[:, 0]
    oo = oxygen[:, None, :] - oxygen[None, :, :]
    oo -= length * np.round(oo / length)
    oo_distance = np.sqrt(np.sum(oo * oo, axis=-1))
    oo_distance[np.eye(len(oxygen), dtype=bool)] = np.inf
    water_ligand = waters.reshape(-1, 3)[:, None, :] - ligand[None, :, :]
    water_ligand -= length * np.round(water_ligand / length)
    return {
        "oxygen_oxygen_minimum_nm": float(np.min(oo_distance)),
        "water_ligand_atom_minimum_nm": float(
            np.min(np.sqrt(np.sum(water_ligand * water_ligand, axis=-1)))
        ),
    }


def freeze(config: Mapping[str, Any]) -> Path:
    validate(config)
    path = result_root(config) / "protocol.json"
    executable, runtime = packmol_paths()
    payload = {
        "schema_version": 1,
        "created_at": utc_now(),
        "protocol_name": config["protocol_name"],
        "claim": config["claim"],
        "config": config,
        "config_file_sha256": sha256_file(ACTIVE_CONFIG_PATH),
        "source_sha256": sha256_file(SOURCE_PATH),
        "physical_system_source_sha256": sha256_file(BASE_SOURCE),
        "packmol_executable_sha256": sha256_file(executable),
        "packmol_runtime_dll_sha256": {
            name: sha256_file(runtime / name)
            for name in ("libgcc_s_seh-1.dll", "libgfortran-5.dll", "libquadmath-0.dll", "libwinpthread-1.dll")
            if (runtime / name).is_file()
        },
        "q1_derived_input_files": [],
        "physical_hamiltonian": True,
        "dummy_atoms": 0,
        "model_training_authorized": False,
        "final_holdout_evaluation_authorized": False,
    }
    if str(config.get("purpose", "model_pilot")) == "final_holdout":
        checkpoint_manifest = ROOT / str(config["frozen_checkpoint_manifest"])
        if not checkpoint_manifest.is_file():
            raise RuntimeError(
                "all model checkpoints must be frozen before final-holdout Packmol"
            )
        payload["frozen_checkpoint_manifest"] = str(checkpoint_manifest.resolve())
        payload["frozen_checkpoint_manifest_sha256"] = sha256_file(
            checkpoint_manifest
        )
    payload["protocol_sha256"] = canonical_sha256(payload)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        existing_without_hash = dict(existing)
        claimed = existing_without_hash.pop("protocol_sha256")
        if canonical_sha256(existing_without_hash) != claimed:
            raise RuntimeError("fresh-family protocol corrupt")
        payload["created_at"] = existing["created_at"]
        payload["protocol_sha256"] = canonical_sha256(
            {key: value for key, value in payload.items() if key != "protocol_sha256"}
        )
        if payload != existing:
            raise RuntimeError("fresh-family frozen inputs changed")
        return path
    atomic_json(path, payload)
    return path


def load_protocol(config: Mapping[str, Any]) -> dict[str, Any]:
    path = freeze(config)
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("protocol_sha256")
    if canonical_sha256(payload) != claimed:
        raise RuntimeError("fresh-family protocol authentication failed")
    payload["protocol_sha256"] = claimed
    return payload


def build_packings(config: Mapping[str, Any]) -> Path:
    protocol = load_protocol(config)
    manifest_path = result_root(config) / "packing_manifest.json"
    if manifest_path.is_file():
        return manifest_path
    executable, runtime = packmol_paths()
    records = []
    families = int(config["sampling"]["families"])
    for family in range(families):
        seed = int(config["packing"]["packmol_seeds"][family])
        template = str(config["packing"]["oh_template_by_family"][family])
        job = data_root(config) / "packmol_jobs" / f"family_{family:02d}"
        job.mkdir(parents=True, exist_ok=False)
        (job / "ligand.pdb").write_text(
            ligand_pdb_text(config, template), encoding="ascii", newline="\n"
        )
        water_text, _ = _water_template()
        (job / "water.pdb").write_text(water_text, encoding="ascii", newline="\n")
        (job / "packmol.inp").write_text(
            packmol_input(config, seed), encoding="ascii", newline="\n"
        )
        environment = dict(os.environ)
        environment["PATH"] = str(runtime) + os.pathsep + environment.get("PATH", "")
        completed = subprocess.run(
            [str(executable), "-i", "packmol.inp", "-o", "packed.pdb"],
            cwd=str(job),
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=float(config["packing"]["timeout_seconds"]),
        )
        (job / "packmol.log").write_text(
            completed.stdout + "\n--- STDERR ---\n" + completed.stderr,
            encoding="utf-8",
            newline="\n",
        )
        packed = job / "packed.pdb"
        if completed.returncode != 0 or not packed.is_file():
            raise RuntimeError(f"fresh Packmol failed for family {family}")
        positions = parse_packed(packed, int(config["system"]["water_count"]))
        diagnostics = packing_diagnostics(
            positions, float(config["system"]["box_length_nm"])
        )
        if diagnostics["oxygen_oxygen_minimum_nm"] < float(
            config["packing"]["oxygen_oxygen_minimum_nm"]
        ):
            raise RuntimeError("fresh Packmol O-O audit failed")
        if diagnostics["water_ligand_atom_minimum_nm"] < float(
            config["packing"]["water_ligand_atom_minimum_nm"]
        ):
            raise RuntimeError("fresh Packmol water-ligand audit failed")
        destination = raw_packing_path(config, family)
        box = np.eye(3, dtype=np.float32) * float(config["system"]["box_length_nm"])
        atomic_npz(
            destination,
            positions_nm=positions.astype(np.float32),
            box_nm=box,
        )
        records.append(
            {
                "family": family,
                "packmol_seed": seed,
                "oh_template": template,
                "path": str(destination.resolve()),
                "sha256": sha256_file(destination),
                **diagnostics,
            }
        )
    atomic_json(
        manifest_path,
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "protocol_sha256": protocol["protocol_sha256"],
            "all_families_independent_packmol": True,
            "q1_derived_coordinates_used": False,
            "files": records,
        },
    )
    return manifest_path


def minimize_packings(config: Mapping[str, Any]) -> Path:
    protocol = load_protocol(config)
    packing_manifest = build_packings(config)
    manifest_path = result_root(config) / "minimized_packing_manifest.json"
    if manifest_path.is_file():
        return manifest_path
    topology, system, _, _ = base.build_physical_system(config)
    records = []
    for family in range(int(config["sampling"]["families"])):
        with np.load(raw_packing_path(config, family), allow_pickle=False) as loaded:
            positions = np.asarray(loaded["positions_nm"], dtype=np.float64)
            box = np.asarray(loaded["box_nm"], dtype=np.float64)
        simulation, integrator = base.create_simulation(
            config, topology, system, 73001 + 2 * family
        )
        simulation.context.setPeriodicBoxVectors(
            *[Vec3(*map(float, row)) * unit.nanometer for row in box]
        )
        simulation.context.setPositions(positions * unit.nanometer)
        simulation.context.applyConstraints(1.0e-8)
        simulation.minimizeEnergy(
            maxIterations=int(config["sampling"]["minimization_max_iterations"])
        )
        state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
        minimized = np.asarray(
            state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
            dtype=np.float32,
        )
        destination = minimized_packing_path(config, family)
        atomic_npz(destination, positions_nm=minimized, box_nm=box.astype(np.float32))
        records.append(
            {
                "family": family,
                "raw_path": str(raw_packing_path(config, family).resolve()),
                "raw_sha256": sha256_file(raw_packing_path(config, family)),
                "path": str(destination.resolve()),
                "sha256": sha256_file(destination),
            }
        )
        del simulation, integrator
    atomic_json(
        manifest_path,
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "protocol_sha256": protocol["protocol_sha256"],
            "packing_manifest_sha256": sha256_file(packing_manifest),
            "q1_derived_coordinates_used": False,
            "files": records,
        },
    )
    return manifest_path


def sampling_protocol(config: Mapping[str, Any]) -> Path:
    parent = load_protocol(config)
    minimized_manifest = minimize_packings(config)
    path = result_root(config) / "sampling_protocol.json"
    payload = {
        "schema_version": 1,
        "created_at": utc_now(),
        "parent_protocol_sha256": parent["protocol_sha256"],
        "minimized_packing_manifest_file_sha256": sha256_file(minimized_manifest),
        "minimized_inputs": {
            str(minimized_packing_path(config, family).resolve()): sha256_file(
                minimized_packing_path(config, family)
            )
            for family in range(int(config["sampling"]["families"]))
        },
        "q1_derived_coordinates_used": False,
        "model_training_authorized": False,
    }
    payload["protocol_sha256"] = canonical_sha256(payload)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        existing_without_hash = dict(existing)
        claimed = existing_without_hash.pop("protocol_sha256")
        if canonical_sha256(existing_without_hash) != claimed:
            raise RuntimeError("fresh sampling protocol corrupt")
        payload["created_at"] = existing["created_at"]
        payload["protocol_sha256"] = canonical_sha256(
            {key: value for key, value in payload.items() if key != "protocol_sha256"}
        )
        if payload != existing:
            raise RuntimeError("fresh minimized inputs changed after freeze")
        return path
    atomic_json(path, payload)
    return path


def run_sampling(config: Mapping[str, Any]) -> Path:
    sampling_protocol_path = sampling_protocol(config)
    protocol = json.loads(sampling_protocol_path.read_text(encoding="utf-8"))
    topology, system, _, metadata = base.build_physical_system(config)
    sampling = config["sampling"]
    timestep = float(sampling["timestep_femtoseconds"])
    equil_steps = base._steps(float(sampling["equilibration_picoseconds"]), timestep)
    equil_diag_steps = base._steps(
        float(sampling["equilibration_diagnostic_interval_picoseconds"]), timestep
    )
    prod_steps = base._steps(float(sampling["production_picoseconds"]), timestep)
    sample_steps = base._steps(
        float(sampling["production_sample_interval_picoseconds"]), timestep
    )
    if equil_steps % equil_diag_steps or prod_steps % sample_steps:
        raise RuntimeError("fresh sampling intervals do not divide durations")
    families = int(sampling["families"])
    manifest_path = result_root(config) / "sampling_manifest.json"
    if manifest_path.is_file() and all(
        production_path(config, family).is_file() for family in range(families)
    ):
        return manifest_path
    started = time.perf_counter()
    durations: list[float] = []
    records = []
    dof = 3 * int(metadata["particles"]) - int(metadata["constraints"]) - 3
    cutoff = float(config["qualification"]["water_shell_cutoff_nm"])
    for family, seed in enumerate(sampling["velocity_seeds"]):
        destination = production_path(config, family)
        diag_destination = diagnostic_path(config, family)
        if destination.is_file() and diag_destination.is_file():
            records.append(
                {
                    "family": family,
                    "production_path": str(destination.resolve()),
                    "production_sha256": sha256_file(destination),
                    "diagnostic_path": str(diag_destination.resolve()),
                    "diagnostic_sha256": sha256_file(diag_destination),
                    "resumed": True,
                }
            )
            continue
        forecast = max(durations[-3:] or [0.0])
        elapsed = time.perf_counter() - started
        if elapsed + forecast >= float(sampling["wall_limit_seconds_per_invocation"]) - float(
            sampling["wall_safety_reserve_seconds"]
        ):
            incomplete = result_root(config) / "SAMPLING_INCOMPLETE_RERUN.json"
            atomic_json(
                incomplete,
                {
                    "created_at": utc_now(),
                    "protocol_sha256": protocol["protocol_sha256"],
                    "completed_families": len(records),
                    "expected_families": families,
                    "elapsed_seconds_this_invocation": elapsed,
                },
            )
            return incomplete
        family_started = time.perf_counter()
        with np.load(minimized_packing_path(config, family), allow_pickle=False) as loaded:
            positions = np.asarray(loaded["positions_nm"], dtype=np.float64)
            box = np.asarray(loaded["box_nm"], dtype=np.float64)
        initial_shell = base.hydration_shell(positions, box, cutoff)
        simulation, integrator = base.create_simulation(
            config, topology, system, int(seed)
        )
        simulation.context.setPeriodicBoxVectors(
            *[Vec3(*map(float, row)) * unit.nanometer for row in box]
        )
        simulation.context.setPositions(positions * unit.nanometer)
        simulation.context.applyConstraints(1.0e-8)
        simulation.context.setVelocitiesToTemperature(
            float(sampling["temperature_kelvin"]) * unit.kelvin, int(seed)
        )
        print(f"fresh family {family:02d}: 1-ns equilibration started", flush=True)
        diag_time, diag_stage = [], []
        potentials, kinetics, temperatures = [], [], []
        pairs, shell_counts, shell_jaccards = [], [], []

        def record(time_ps: float, stage: int) -> dict[str, Any]:
            observed = base.state_observables(simulation, initial_shell, cutoff, dof)
            diag_time.append(float(time_ps))
            diag_stage.append(int(stage))
            potentials.append(observed["potential"])
            kinetics.append(observed["kinetic"])
            temperatures.append(observed["temperature"])
            pairs.append(observed["pair_distances"])
            shell_counts.append(observed["shell_count"])
            shell_jaccards.append(observed["shell_jaccard"])
            return observed

        record(0.0, 0)
        equil_chunks = equil_steps // equil_diag_steps
        equil_interval = float(sampling["equilibration_diagnostic_interval_picoseconds"])
        for chunk in range(equil_chunks):
            simulation.step(equil_diag_steps)
            record((chunk + 1) * equil_interval, 0)
            if (chunk + 1) % max(1, equil_chunks // 5) == 0:
                print(
                    f"fresh family {family:02d}: equilibrated {(chunk + 1) * equil_interval:.0f} ps",
                    flush=True,
                )
        production_positions, production_boxes = [], []
        production_chunks = prod_steps // sample_steps
        prod_interval = float(sampling["production_sample_interval_picoseconds"])
        equil_time = float(sampling["equilibration_picoseconds"])
        for chunk in range(production_chunks):
            simulation.step(sample_steps)
            observed = record(equil_time + (chunk + 1) * prod_interval, 1)
            production_positions.append(observed["positions"].astype(np.float32))
            production_boxes.append(observed["box"].astype(np.float32))
            if (chunk + 1) % max(1, production_chunks // 5) == 0:
                print(
                    f"fresh family {family:02d}: produced {(chunk + 1) * prod_interval:.0f} ps",
                    flush=True,
                )
        coordinate_arrays = {
            "positions_nm": np.asarray(production_positions, dtype=np.float32),
            "boxes_nm": np.asarray(production_boxes, dtype=np.float32),
        }
        if not all(np.all(np.isfinite(value)) for value in coordinate_arrays.values()):
            raise RuntimeError("fresh family produced non-finite coordinate")
        atomic_npz(destination, **coordinate_arrays)
        atomic_npz(
            diag_destination,
            time_ps=np.asarray(diag_time, dtype=np.float32),
            stage=np.asarray(diag_stage, dtype=np.int8),
            potential_kj_mol=np.asarray(potentials, dtype=np.float64),
            kinetic_kj_mol=np.asarray(kinetics, dtype=np.float64),
            temperature_kelvin=np.asarray(temperatures, dtype=np.float64),
            ligand_pair_distances_nm=np.asarray(pairs, dtype=np.float32),
            shell_count=np.asarray(shell_counts, dtype=np.int16),
            shell_jaccard_from_packed_start=np.asarray(shell_jaccards, dtype=np.float32),
        )
        duration = time.perf_counter() - family_started
        durations.append(duration)
        records.append(
            {
                "family": family,
                "packmol_seed": int(config["packing"]["packmol_seeds"][family]),
                "oh_initial_template": config["packing"]["oh_template_by_family"][family],
                "velocity_seed": int(seed),
                "minimized_input_sha256": sha256_file(minimized_packing_path(config, family)),
                "production_path": str(destination.resolve()),
                "production_sha256": sha256_file(destination),
                "diagnostic_path": str(diag_destination.resolve()),
                "diagnostic_sha256": sha256_file(diag_destination),
                "frames": len(production_positions),
                "elapsed_seconds": duration,
                "resumed": False,
            }
        )
        del simulation, integrator
        print(f"fresh family {family:02d}: completed in {duration:.1f} s", flush=True)
    atomic_json(
        manifest_path,
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "protocol_sha256": protocol["protocol_sha256"],
            "q1_derived_coordinates_used": False,
            "all_families_independent_packmol": True,
            "physical_hamiltonian": True,
            "dummy_atoms": 0,
            "coordinate_only_production": True,
            "diagnostic_energies_for_training_forbidden": True,
            "family_split": config["splits"],
            "files": records,
            "elapsed_seconds_this_invocation": time.perf_counter() - started,
        },
    )
    return manifest_path


def window(values: np.ndarray, which: str) -> np.ndarray:
    third = max(1, len(values) // 3)
    if which == "early":
        return values[:third]
    if which == "late":
        return values[-third:]
    raise ValueError(which)


def orientation_coordinate(pair_values: np.ndarray) -> np.ndarray:
    pairs = [tuple(map(int, item)) for item in base.pair_indices(13)]
    lookup = {pair: index for index, pair in enumerate(pairs)}
    return pair_values[:, lookup[(2, 12)]] - pair_values[:, lookup[(4, 12)]]


def hysteretic_states(values: np.ndarray, threshold: float) -> np.ndarray:
    states = np.empty(len(values), dtype=np.int8)
    state = 1 if values[0] >= 0.0 else 0
    for index, value in enumerate(values):
        if value > threshold:
            state = 1
        elif value < -threshold:
            state = 0
        states[index] = state
    return states


def integrated_autocorrelation_time(values: np.ndarray) -> float:
    centered = np.asarray(values, dtype=np.float64) - float(np.mean(values))
    variance = float(np.mean(centered**2))
    if variance <= 0.0:
        return float("inf")
    correlation = np.correlate(centered, centered, mode="full")[len(centered) - 1 :]
    correlation /= variance * np.arange(len(centered), 0, -1)
    total = 1.0
    for lag in range(1, len(correlation)):
        if correlation[lag] <= 0.0:
            break
        total += 2.0 * float(correlation[lag])
    return max(1.0, total)


def qualify(config: Mapping[str, Any]) -> Path:
    protocol_path = sampling_protocol(config)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    thresholds = config["qualification"]
    family_records, pair_by_family, states_by_family = [], [], []
    early_states, late_states = [], []
    all_finite = True
    pair_index = base.pair_indices(13)
    heavy = np.asarray([(a <= 6 and b <= 6) for a, b in pair_index])
    substitution = np.asarray(
        [tuple(map(int, pair)) in {(3, 6), (6, 12)} for pair in pair_index]
    )
    total_01 = total_10 = total_0 = total_1 = 0
    window_effective = 0.0
    for family in range(int(config["sampling"]["families"])):
        with np.load(production_path(config, family), allow_pickle=False) as coordinates:
            all_finite &= all(np.all(np.isfinite(coordinates[key])) for key in coordinates.files)
        with np.load(diagnostic_path(config, family), allow_pickle=False) as diagnostic:
            stage = np.asarray(diagnostic["stage"])
            prod = stage == 1
            temperature = np.asarray(diagnostic["temperature_kelvin"])[prod]
            potential = np.asarray(diagnostic["potential_kj_mol"])[prod]
            pairs = np.asarray(diagnostic["ligand_pair_distances_nm"])
            pair = pairs[prod]
            jaccard = np.asarray(diagnostic["shell_jaccard_from_packed_start"])
            all_finite &= all(np.all(np.isfinite(diagnostic[key])) for key in diagnostic.files)
        early_pair, late_pair = window(pair, "early"), window(pair, "late")
        pair_sd = np.maximum(np.std(pair, axis=0, ddof=1), 1.0e-4)
        delta_z = (np.mean(late_pair, axis=0) - np.mean(early_pair, axis=0)) / pair_sd
        potential_shift = abs(
            float(np.mean(window(potential, "late")) - np.mean(window(potential, "early")))
        ) / max(float(np.std(potential, ddof=1)), 1.0e-12)
        states = hysteretic_states(
            orientation_coordinate(pair), float(thresholds["oh_orientation_hysteresis_nm"])
        )
        changes = states[1:] - states[:-1]
        count_01 = int(np.count_nonzero(changes == 1))
        count_10 = int(np.count_nonzero(changes == -1))
        total_01 += count_01
        total_10 += count_10
        total_0 += int(np.count_nonzero(states[:-1] == 0))
        total_1 += int(np.count_nonzero(states[:-1] == 1))
        early_state, late_state = window(states, "early"), window(states, "late")
        tau = integrated_autocorrelation_time(states)
        window_effective += len(early_state) / tau
        early_states.append(early_state)
        late_states.append(late_state)
        states_by_family.append(states)
        pair_by_family.append(pair)
        production_start = int(np.flatnonzero(prod)[0])
        family_records.append(
            {
                "family": family,
                "temperature_mean_kelvin": float(np.mean(temperature)),
                "temperature_relative_error": abs(
                    float(np.mean(temperature)) - float(config["sampling"]["temperature_kelvin"])
                ) / float(config["sampling"]["temperature_kelvin"]),
                "potential_early_late_shift_sigma": potential_shift,
                "heavy_atom_pair_early_late_rms_z": float(
                    np.sqrt(np.mean(delta_z[heavy] ** 2))
                ),
                "substitution_bond_mean_shift_nm": float(
                    np.max(
                        np.abs(np.mean(late_pair, axis=0) - np.mean(early_pair, axis=0))[
                            substitution
                        ]
                    )
                ),
                "initial_shell_production_start_jaccard": float(jaccard[production_start]),
                "oh_plus_fraction": float(np.mean(states)),
                "oh_transitions_0_to_1": count_01,
                "oh_transitions_1_to_0": count_10,
                "oh_transitions_total": count_01 + count_10,
                "oh_integrated_autocorrelation_time_ps": float(2.0 * tau),
            }
        )
    stacked_pairs = np.asarray(pair_by_family)
    family_means = np.mean(stacked_pairs, axis=1)
    pooled_sd = np.maximum(
        np.sqrt(np.mean(np.var(stacked_pairs, axis=1, ddof=1), axis=0)), 1.0e-4
    )
    grand = np.mean(family_means, axis=0)
    family_rms = np.sqrt(np.mean(((family_means - grand) / pooled_sd) ** 2, axis=1))
    for record, value in zip(family_records, family_rms):
        record["family_pair_mean_rms_z"] = float(value)
    pooled_states = np.concatenate(states_by_family)
    pooled_early = np.concatenate(early_states)
    pooled_late = np.concatenate(late_states)
    early_fraction, late_fraction = float(np.mean(pooled_early)), float(np.mean(pooled_late))
    difference = abs(late_fraction - early_fraction)
    difference_se = float(
        np.sqrt(
            early_fraction * (1.0 - early_fraction) / window_effective
            + late_fraction * (1.0 - late_fraction) / window_effective
        )
    )
    rate_01 = total_01 / max(total_0, 1)
    rate_10 = total_10 / max(total_1, 1)
    stationary = (
        rate_01 / (rate_01 + rate_10)
        if rate_01 + rate_10 > 0.0
        else 0.5
    )
    split_fractions = {}
    for split, members in config["splits"].items():
        values = np.concatenate([states_by_family[int(member)] for member in members])
        split_fractions[split] = float(np.mean(values))
    gates = {
        "all_arrays_finite": bool(all_finite),
        "temperature": all(
            item["temperature_relative_error"]
            <= float(thresholds["temperature_relative_tolerance"])
            for item in family_records
        ),
        "potential_stationarity": all(
            item["potential_early_late_shift_sigma"]
            <= float(thresholds["potential_early_late_shift_sigma_max"])
            for item in family_records
        ),
        "heavy_atom_stationarity": all(
            item["heavy_atom_pair_early_late_rms_z"]
            <= float(thresholds["heavy_atom_pair_early_late_rms_z_max"])
            and item["substitution_bond_mean_shift_nm"]
            <= float(thresholds["substitution_bond_mean_shift_nm_max"])
            for item in family_records
        ),
        "packed_water_shell_memory_lost": float(
            np.median(
                [item["initial_shell_production_start_jaccard"] for item in family_records]
            )
        )
        <= float(thresholds["initial_shell_production_start_jaccard_median_max"]),
        "oh_transitions_each_family": all(
            item["oh_transitions_total"]
            >= int(thresholds["oh_orientation_transitions_per_family_min"])
            for item in family_records
        ),
        "oh_pooled_occupancy": float(thresholds["pooled_oh_plus_fraction_min"])
        <= float(np.mean(pooled_states))
        <= float(thresholds["pooled_oh_plus_fraction_max"]),
        "oh_rate_implied_occupancy": float(thresholds["rate_implied_oh_plus_fraction_min"])
        <= stationary
        <= float(thresholds["rate_implied_oh_plus_fraction_max"]),
        "oh_autocorrelation_aware_window_stability": difference / max(difference_se, 1.0e-12)
        <= float(thresholds["early_late_difference_z_max"]),
        "oh_both_states_each_split": all(
            min(value, 1.0 - value)
            >= float(thresholds["split_oh_minor_state_fraction_min"])
            for value in split_fractions.values()
        ),
        "family_consistency": bool(
            np.max(family_rms) <= float(thresholds["family_pair_mean_rms_z_max"])
        ),
    }
    passed = all(gates.values())
    purpose = str(config.get("purpose", "model_pilot"))
    if purpose not in {"model_pilot", "final_holdout"}:
        raise RuntimeError(f"unknown fresh-family purpose: {purpose}")
    pass_decision = (
        "PASS_TO_FINAL_HOLDOUT_EVALUATION"
        if purpose == "final_holdout"
        else "PASS_TO_MODEL_PILOT"
    )
    output = result_root(config) / "qualification.json"
    atomic_json(
        output,
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "protocol_sha256": protocol["protocol_sha256"],
            "decision": pass_decision if passed else "STOP_OR_EXTEND_FRESH_MD_BEFORE_EVALUATION",
            "purpose": purpose,
            "model_training_authorized": bool(passed and purpose == "model_pilot"),
            "final_holdout_evaluation_authorized": bool(
                passed and purpose == "final_holdout"
            ),
            "q1_derived_coordinates_used": False,
            "thresholds": thresholds,
            "gates": gates,
            "families": family_records,
            "pooled": {
                "temperature_mean_kelvin": float(
                    np.mean([item["temperature_mean_kelvin"] for item in family_records])
                ),
                "oh_plus_fraction": float(np.mean(pooled_states)),
                "oh_rate_implied_stationary_plus_fraction": float(stationary),
                "oh_transitions_0_to_1": total_01,
                "oh_transitions_1_to_0": total_10,
                "oh_early_plus_fraction": early_fraction,
                "oh_late_plus_fraction": late_fraction,
                "oh_early_late_difference": difference,
                "oh_early_late_difference_standard_error": difference_se,
                "oh_early_late_difference_z": difference / max(difference_se, 1.0e-12),
                "split_oh_plus_fractions": split_fractions,
                "family_pair_mean_rms_z_max": float(np.max(family_rms)),
            },
        },
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("freeze", "pack", "minimize", "sample", "qualify", "all")
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, smoke=args.smoke)
    if args.stage == "freeze":
        result: Any = freeze(config)
    elif args.stage == "pack":
        result = build_packings(config)
    elif args.stage == "minimize":
        result = minimize_packings(config)
    elif args.stage == "sample":
        result = run_sampling(config)
    elif args.stage == "qualify":
        result = qualify(config)
    else:
        protocol = freeze(config)
        packings = build_packings(config)
        minimized = minimize_packings(config)
        sampling = run_sampling(config)
        qualification = (
            qualify(config) if sampling.name == "sampling_manifest.json" else "sampling incomplete"
        )
        result = {
            "protocol": str(protocol),
            "packings": str(packings),
            "minimized": str(minimized),
            "sampling": str(sampling),
            "qualification": str(qualification),
        }
    print(json.dumps(result if isinstance(result, dict) else {"result": str(result)}, indent=2))


if __name__ == "__main__":
    main()
