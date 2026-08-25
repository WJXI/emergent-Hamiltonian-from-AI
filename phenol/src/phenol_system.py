"""Build the physical 13-atom phenol + 512-water system used in the paper."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from openmm import (
    CMMotionRemover,
    HarmonicAngleForce,
    HarmonicBondForce,
    NonbondedForce,
    PeriodicTorsionForce,
    System,
    Vec3,
    unit,
)
from openmm.app import AmberInpcrdFile, AmberPrmtopFile, ForceField, NoCutoff, Topology


PHENOL_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT = PHENOL_ROOT / "data" / "public" / "freesolv_v0_52_parameter_source" / "selected"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def authenticate_phenol_parameters() -> dict[str, Any]:
    metadata_path = PUBLIC_ROOT / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    record = metadata["molecules"]["phenol"]
    for relative, expected in record["files"].items():
        path = PUBLIC_ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"public phenol parameter authentication failed: {relative}")
    return metadata


def load_phenol_endpoint() -> tuple[Topology, System, np.ndarray]:
    metadata = authenticate_phenol_parameters()
    prefix = metadata["molecules"]["phenol"]["freesolv_id"]
    prmtop = AmberPrmtopFile(str(PUBLIC_ROOT / "amber" / f"{prefix}.prmtop"))
    inpcrd = AmberInpcrdFile(str(PUBLIC_ROOT / "amber" / f"{prefix}.inpcrd"))
    system = prmtop.createSystem(
        nonbondedMethod=NoCutoff,
        constraints=None,
        rigidWater=False,
        removeCMMotion=False,
    )
    positions = np.asarray(inpcrd.positions.value_in_unit(unit.nanometer), dtype=np.float64)
    if system.getNumParticles() != 13 or positions.shape != (13, 3):
        raise RuntimeError("the authenticated phenol endpoint is no longer 13 atoms")
    return prmtop.topology, system, positions


def force(system: System, force_type):
    matches = [item for item in system.getForces() if isinstance(item, force_type)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {force_type.__name__}, found {len(matches)}")
    return matches[0]


def optional_force(system: System, force_type):
    matches = [item for item in system.getForces() if isinstance(item, force_type)]
    if len(matches) > 1:
        raise RuntimeError(f"expected at most one {force_type.__name__}")
    return matches[0] if matches else None


def water_system(water_count: int, box_length_nm: float) -> tuple[Topology, System]:
    topology = Topology()
    chain = topology.addChain("W")
    from openmm.app import element

    for index in range(int(water_count)):
        residue = topology.addResidue("HOH", chain, str(index + 2))
        oxygen = topology.addAtom("O", element.oxygen, residue)
        hydrogen_1 = topology.addAtom("H1", element.hydrogen, residue)
        hydrogen_2 = topology.addAtom("H2", element.hydrogen, residue)
        topology.addBond(oxygen, hydrogen_1)
        topology.addBond(oxygen, hydrogen_2)
    length = float(box_length_nm)
    topology.setPeriodicBoxVectors(
        (Vec3(length, 0, 0), Vec3(0, length, 0), Vec3(0, 0, length)) * unit.nanometer
    )
    system = ForceField("amber14/tip3p.xml").createSystem(
        topology,
        nonbondedMethod=NoCutoff,
        constraints=None,
        rigidWater=True,
        removeCMMotion=False,
    )
    if system.getNumParticles() != 3 * int(water_count):
        raise RuntimeError("TIP3P particle count changed")
    return topology, system


def copy_bonds(destination: HarmonicBondForce, source: HarmonicBondForce, mapping: Mapping[int, int]) -> None:
    for index in range(source.getNumBonds()):
        first, second, length, spring = source.getBondParameters(index)
        destination.addBond(mapping[int(first)], mapping[int(second)], length, spring)


def copy_angles(destination: HarmonicAngleForce, source: HarmonicAngleForce, mapping: Mapping[int, int]) -> None:
    for index in range(source.getNumAngles()):
        first, second, third, angle, spring = source.getAngleParameters(index)
        destination.addAngle(
            mapping[int(first)], mapping[int(second)], mapping[int(third)], angle, spring
        )


def copy_torsions(
    destination: PeriodicTorsionForce,
    source: PeriodicTorsionForce,
    mapping: Mapping[int, int],
) -> None:
    for index in range(source.getNumTorsions()):
        first, second, third, fourth, periodicity, phase, spring = source.getTorsionParameters(index)
        destination.addTorsion(
            mapping[int(first)],
            mapping[int(second)],
            mapping[int(third)],
            mapping[int(fourth)],
            periodicity,
            phase,
            spring,
        )


def copy_nonbonded_settings(destination: NonbondedForce, config: Mapping[str, Any]) -> None:
    system = config["system"]
    destination.setNonbondedMethod(NonbondedForce.PME)
    destination.setCutoffDistance(float(system["nonbonded_cutoff_nm"]) * unit.nanometer)
    destination.setEwaldErrorTolerance(float(system.get("ewald_error_tolerance", 1.0e-5)))
    destination.setUseDispersionCorrection(bool(system.get("use_dispersion_correction", True)))


def combined_topology(ligand: Topology, water: Topology, box_length_nm: float) -> Topology:
    combined = Topology()
    old_to_new = {}
    for source in (ligand, water):
        for source_chain in source.chains():
            chain = combined.addChain(source_chain.id)
            for source_residue in source_chain.residues():
                residue = combined.addResidue(source_residue.name, chain, source_residue.id)
                for source_atom in source_residue.atoms():
                    old_to_new[source_atom] = combined.addAtom(
                        source_atom.name, source_atom.element, residue
                    )
        for first, second in source.bonds():
            combined.addBond(old_to_new[first], old_to_new[second])
    length = float(box_length_nm)
    combined.setPeriodicBoxVectors(
        (Vec3(length, 0, 0), Vec3(0, length, 0), Vec3(0, 0, length)) * unit.nanometer
    )
    return combined


def build_physical_system(config: Mapping[str, Any]) -> tuple[Topology, System, np.ndarray, dict[str, Any]]:
    """Return the no-dummy OpenMM system used for all reported phenol MD."""
    if "system" in config:
        effective_config = dict(config)
    elif "md_config" in config:
        md_path = PHENOL_ROOT / str(config["md_config"])
        effective_config = json.loads(md_path.read_text(encoding="utf-8"))
    else:
        raise KeyError("the protocol must contain either system or md_config")
    settings = effective_config["system"]
    phenol_topology, phenol, positions = load_phenol_endpoint()
    water_topology, water = water_system(
        int(settings["water_count"]), float(settings["box_length_nm"])
    )
    physical = System()
    for index in range(phenol.getNumParticles()):
        physical.addParticle(phenol.getParticleMass(index))
    for index in range(water.getNumParticles()):
        physical.addParticle(water.getParticleMass(index))
    for index in range(water.getNumConstraints()):
        first, second, distance = water.getConstraintParameters(index)
        physical.addConstraint(13 + int(first), 13 + int(second), distance)

    length = float(settings["box_length_nm"])
    physical.setDefaultPeriodicBoxVectors(
        Vec3(length, 0, 0) * unit.nanometer,
        Vec3(0, length, 0) * unit.nanometer,
        Vec3(0, 0, length) * unit.nanometer,
    )
    ligand_map = {index: index for index in range(13)}
    water_map = {index: 13 + index for index in range(water.getNumParticles())}
    bonds, angles, torsions = HarmonicBondForce(), HarmonicAngleForce(), PeriodicTorsionForce()
    copy_bonds(bonds, force(phenol, HarmonicBondForce), ligand_map)
    copy_angles(angles, force(phenol, HarmonicAngleForce), ligand_map)
    copy_torsions(torsions, force(phenol, PeriodicTorsionForce), ligand_map)
    for force_type, destination, copier in (
        (HarmonicBondForce, bonds, copy_bonds),
        (HarmonicAngleForce, angles, copy_angles),
        (PeriodicTorsionForce, torsions, copy_torsions),
    ):
        source = optional_force(water, force_type)
        if source is not None:
            copier(destination, source, water_map)
    physical.addForce(bonds)
    physical.addForce(angles)
    physical.addForce(torsions)

    nonbonded = NonbondedForce()
    copy_nonbonded_settings(nonbonded, effective_config)
    phenol_nb, water_nb = force(phenol, NonbondedForce), force(water, NonbondedForce)
    for index in range(phenol_nb.getNumParticles()):
        nonbonded.addParticle(*phenol_nb.getParticleParameters(index))
    for index in range(water_nb.getNumParticles()):
        nonbonded.addParticle(*water_nb.getParticleParameters(index))
    for index in range(phenol_nb.getNumExceptions()):
        nonbonded.addException(*phenol_nb.getExceptionParameters(index))
    for index in range(water_nb.getNumExceptions()):
        first, second, charge, sigma, epsilon = water_nb.getExceptionParameters(index)
        nonbonded.addException(13 + int(first), 13 + int(second), charge, sigma, epsilon)
    physical.addForce(nonbonded)
    physical.addForce(CMMotionRemover())

    topology = combined_topology(phenol_topology, water_topology, length)
    metadata = {
        "physical_endpoint": "phenol",
        "dummy_atoms": 0,
        "phenol_atoms": 13,
        "water_count": int(settings["water_count"]),
        "particles": physical.getNumParticles(),
        "constraints": physical.getNumConstraints(),
        "forcefield": "FreeSolv v0.52 GAFF 1.7 / AM1-BCC + amber14 TIP3P",
        "nonbonded_method": "PME",
        "box_length_nm": length,
        "atom_names": [atom.name for atom in phenol_topology.atoms()],
        "atom_elements": [atom.element.symbol for atom in phenol_topology.atoms()],
    }
    return topology, physical, positions, metadata
