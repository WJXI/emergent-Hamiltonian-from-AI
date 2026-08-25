"""OpenMM and geometric helpers shared by the released Phenol MD workflows."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from openmm import LangevinMiddleIntegrator, Platform, unit
from openmm.app import Simulation

import phenol_system


build_physical_system = phenol_system.build_physical_system


def _steps(picoseconds: float, timestep_femtoseconds: float) -> int:
    value = 1000.0 * float(picoseconds) / float(timestep_femtoseconds)
    rounded = int(round(value))
    if not np.isclose(value, rounded, rtol=0.0, atol=1.0e-9):
        raise ValueError("the requested interval is not an integer number of MD steps")
    return rounded


def pair_indices(atom_count: int = 13) -> np.ndarray:
    return np.asarray(
        [(first, second) for first in range(atom_count) for second in range(first + 1, atom_count)],
        dtype=np.int16,
    )


def minimum_image_displacements(displacements: np.ndarray, box: np.ndarray) -> np.ndarray:
    box = np.asarray(box, dtype=np.float64)
    diagonal = np.diag(box)
    if box.shape != (3, 3) or not np.allclose(box, np.diag(diagonal), atol=1.0e-7):
        raise RuntimeError("the released protocol requires an orthorhombic box")
    if np.any(diagonal <= 0.0):
        raise RuntimeError("periodic box lengths must be positive")
    fractional = np.asarray(displacements, dtype=np.float64) / diagonal
    fractional -= np.round(fractional)
    return fractional * diagonal


def ligand_pair_distances(positions: np.ndarray, box: np.ndarray) -> np.ndarray:
    pairs = pair_indices(13)
    displacement = positions[pairs[:, 1]] - positions[pairs[:, 0]]
    return np.linalg.norm(minimum_image_displacements(displacement, box), axis=1)


def hydration_shell(positions: np.ndarray, box: np.ndarray, cutoff_nm: float) -> np.ndarray:
    heavy = positions[:7]
    water_oxygen = positions[13::3]
    displacement = water_oxygen[:, None, :] - heavy[None, :, :]
    distance = np.linalg.norm(minimum_image_displacements(displacement, box), axis=2)
    return np.flatnonzero(distance.min(axis=1) < float(cutoff_nm)).astype(np.int16)


def shell_jaccard(first: np.ndarray, second: np.ndarray) -> float:
    left, right = set(map(int, first)), set(map(int, second))
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def create_simulation(
    config: Mapping[str, Any], topology: Any, system: Any, seed: int
) -> tuple[Simulation, LangevinMiddleIntegrator]:
    sampling = config["sampling"]
    integrator = LangevinMiddleIntegrator(
        float(sampling["temperature_kelvin"]) * unit.kelvin,
        float(sampling["friction_per_picosecond"]) / unit.picosecond,
        float(sampling["timestep_femtoseconds"]) * unit.femtosecond,
    )
    integrator.setRandomNumberSeed(int(seed))
    runtime = config["runtime"]
    platform = Platform.getPlatformByName(str(runtime["platform"]))
    properties = {str(key): str(value) for key, value in runtime["platform_properties"].items()}
    return Simulation(topology, system, integrator, platform, properties), integrator


def state_observables(
    simulation: Simulation,
    initial_shell: np.ndarray,
    cutoff_nm: float,
    degrees_of_freedom: int,
) -> dict[str, Any]:
    state = simulation.context.getState(
        getPositions=True, getEnergy=True, enforcePeriodicBox=True
    )
    positions = np.asarray(
        state.getPositions(asNumpy=True).value_in_unit(unit.nanometer), dtype=np.float64
    )
    box = np.asarray(
        state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer), dtype=np.float64
    )
    potential = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    kinetic = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
    gas_constant = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(
        unit.kilojoule_per_mole / unit.kelvin
    )
    temperature = 2.0 * kinetic / (degrees_of_freedom * gas_constant)
    current_shell = hydration_shell(positions, box, cutoff_nm)
    return {
        "positions": positions,
        "box": box,
        "potential": float(potential),
        "kinetic": float(kinetic),
        "temperature": float(temperature),
        "pair_distances": ligand_pair_distances(positions, box),
        "shell_count": int(len(current_shell)),
        "shell_jaccard": shell_jaccard(initial_shell, current_shell),
    }
