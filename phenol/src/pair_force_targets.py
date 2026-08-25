"""Construct the two intramolecular nonbonded targets used in Table I."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import torch
from openmm import Context, NonbondedForce, Platform, VerletIntegrator, unit

import phenol_system


def graph_distances(atom_count: int, bonds: list[list[int]]) -> np.ndarray:
    distance = np.full((atom_count, atom_count), atom_count + 1, dtype=np.int16)
    np.fill_diagonal(distance, 0)
    for first, second in bonds:
        distance[first, second] = distance[second, first] = 1
    for middle in range(atom_count):
        distance = np.minimum(distance, distance[:, middle, None] + distance[None, middle, :])
    return distance


def extract_nonbonded_parameters(config: Mapping[str, Any]) -> dict[str, Any]:
    _, system, _, metadata = phenol_system.build_physical_system(config)
    nonbonded = phenol_system.force(system, NonbondedForce)
    particles = []
    for index in range(13):
        charge, sigma, epsilon = nonbonded.getParticleParameters(index)
        particles.append(
            (
                float(charge.value_in_unit(unit.elementary_charge)),
                float(sigma.value_in_unit(unit.nanometer)),
                float(epsilon.value_in_unit(unit.kilojoule_per_mole)),
            )
        )
    exceptions = []
    for index in range(nonbonded.getNumExceptions()):
        first, second, charge, sigma, epsilon = nonbonded.getExceptionParameters(index)
        if int(first) < 13 and int(second) < 13:
            exceptions.append(
                (
                    min(int(first), int(second)),
                    max(int(first), int(second)),
                    float(charge.value_in_unit(unit.elementary_charge**2)),
                    float(sigma.value_in_unit(unit.nanometer)),
                    float(epsilon.value_in_unit(unit.kilojoule_per_mole)),
                )
            )
    integrator = VerletIntegrator(0.001)
    context = Context(system, integrator, Platform.getPlatformByName("CPU"))
    alpha, grid_x, grid_y, grid_z = nonbonded.getPMEParametersInContext(context)
    del context, integrator

    distance = graph_distances(13, config["molecule"]["bonds"])
    graph_d3 = {(i, j) for i in range(13) for j in range(i + 1, 13) if distance[i, j] == 3}
    graph_gt3 = {(i, j) for i in range(13) for j in range(i + 1, 13) if distance[i, j] > 3}
    nonzero = {(i, j) for i, j, charge, _, epsilon in exceptions if abs(charge) + abs(epsilon) > 1e-14}
    zero = {(i, j) for i, j, charge, _, epsilon in exceptions if abs(charge) + abs(epsilon) <= 1e-14}
    expected_zero = {(i, j) for i in range(13) for j in range(i + 1, 13) if distance[i, j] in (1, 2)}
    if nonzero != graph_d3 or zero != expected_zero:
        raise RuntimeError("OpenMM solute exceptions no longer match the declared molecular graph")
    lookup = {(row[0], row[1]): row[2:] for row in exceptions}
    direct = []
    for first, second in sorted(graph_gt3):
        if (first, second) in lookup:
            raise RuntimeError("longer-range pair unexpectedly has an OpenMM exception")
        q1, s1, e1 = particles[first]
        q2, s2, e2 = particles[second]
        direct.append((first, second, q1 * q2, 0.5 * (s1 + s2), math.sqrt(e1 * e2)))
    return {
        "exception_rows": [row for row in exceptions if (row[0], row[1]) in nonzero],
        "direct_rows": direct,
        "pme_alpha_per_nm": float(alpha),
        "pme_grid": [int(grid_x), int(grid_y), int(grid_z)],
        "cutoff_nm": float(nonbonded.getCutoffDistance().value_in_unit(unit.nanometer)),
        "system_metadata": metadata,
        "sets_match_graph_exactly": True,
    }


def pair_force_targets(
    frames: np.ndarray,
    rows: list[tuple[int, int, float, float, float]],
    coulomb_constant: float,
    mode: str,
    pme_alpha: float | None = None,
    device: torch.device | None = None,
) -> np.ndarray:
    if mode not in ("bare", "pme_real") or (mode == "pme_real" and pme_alpha is None):
        raise ValueError("invalid pair-force mode")
    device = device or torch.device("cpu")
    coordinates = torch.as_tensor(frames, dtype=torch.float64, device=device).requires_grad_(True)
    index = torch.as_tensor([[row[0], row[1]] for row in rows], dtype=torch.long, device=device)
    charge = torch.as_tensor([row[2] for row in rows], dtype=torch.float64, device=device)
    sigma = torch.as_tensor([row[3] for row in rows], dtype=torch.float64, device=device)
    epsilon = torch.as_tensor([row[4] for row in rows], dtype=torch.float64, device=device)
    distance = torch.linalg.vector_norm(
        coordinates[:, index[:, 1]] - coordinates[:, index[:, 0]], dim=-1
    ).clamp_min(1e-10)
    ratio = sigma.unsqueeze(0) / distance
    lennard_jones = 4.0 * epsilon.unsqueeze(0) * (ratio.pow(12) - ratio.pow(6))
    if mode == "bare":
        coulomb = float(coulomb_constant) * charge.unsqueeze(0) / distance
    else:
        coulomb = (
            float(coulomb_constant)
            * charge.unsqueeze(0)
            * torch.erfc(float(pme_alpha) * distance)
            / distance
        )
    energy = (lennard_jones + coulomb).sum(dim=1)
    return (-torch.autograd.grad(energy.sum(), coordinates)[0]).detach().cpu().numpy()
