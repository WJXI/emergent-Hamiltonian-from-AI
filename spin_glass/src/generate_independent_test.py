"""Generate reproducible validation/test REMC snapshots on unseen sequences."""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from numba import njit


SPIN_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = SPIN_ROOT / "data" / "train_40000"
EXISTING_TEST_DIR = SPIN_ROOT / "data" / "test_2000"
DATA_DIR = SPIN_ROOT / "data" / "generated"

J1_AA, J1_BB, J1_AB = 1.0, 0.5, 0.8
J2_AA, J2_BB, J2_AB = -0.45, -0.25, -0.35


def get_couplings(sequence: str) -> tuple[np.ndarray, np.ndarray]:
    n = len(sequence)
    j1 = np.zeros(n, dtype=np.float64)
    j2 = np.zeros(n, dtype=np.float64)
    for i in range(n - 1):
        a, b = sequence[i], sequence[i + 1]
        j1[i] = J1_AA if a == b == "A" else J1_BB if a == b == "B" else J1_AB
    for i in range(n - 2):
        a, b = sequence[i], sequence[i + 2]
        j2[i] = J2_AA if a == b == "A" else J2_BB if a == b == "B" else J2_AB
    return j1, j2


@njit(nogil=True)
def local_field(index, spins, j1, j2):
    n = spins.shape[0]
    field = np.zeros(3)
    if index > 0:
        field += j1[index - 1] * spins[index - 1]
    if index < n - 1:
        field += j1[index] * spins[index + 1]
    if index > 1:
        field += j2[index - 2] * spins[index - 2]
    if index < n - 2:
        field += j2[index] * spins[index + 2]
    return field


@njit(nogil=True)
def total_energy(spins, j1, j2):
    n = spins.shape[0]
    energy = 0.0
    for i in range(n - 1):
        energy -= j1[i] * np.dot(spins[i], spins[i + 1])
    for i in range(n - 2):
        energy -= j2[i] * np.dot(spins[i], spins[i + 2])
    return energy


@njit(nogil=True)
def sample_vmf(mu, kappa):
    if kappa < 1.0e-8:
        vector = np.random.randn(3)
        return vector / np.linalg.norm(vector)
    uniform = np.random.rand()
    z = 1.0 + np.log(uniform + (1.0 - uniform) * np.exp(-2.0 * kappa)) / kappa
    z = min(1.0, max(-1.0, z))
    phi = np.random.rand() * 2.0 * np.pi
    radius = np.sqrt(1.0 - z * z)
    local = np.array([radius * np.cos(phi), radius * np.sin(phi), z])
    direction = mu / np.linalg.norm(mu)
    if direction[2] > 0.9999:
        return local
    if direction[2] < -0.9999:
        return np.array([local[0], -local[1], -local[2]])
    basis1 = np.array([-direction[1], direction[0], 0.0])
    basis1 /= np.linalg.norm(basis1)
    basis2 = np.cross(direction, basis1)
    result = local[0] * basis1 + local[1] * basis2 + local[2] * direction
    return result / np.linalg.norm(result)


@njit(nogil=True)
def sweep(spins, j1, j2, beta):
    for index in range(spins.shape[0]):
        field = local_field(index, spins, j1, j2)
        magnitude = np.linalg.norm(field)
        if magnitude > 1.0e-8:
            if np.random.rand() < 0.5:
                spins[index] = sample_vmf(field, beta * magnitude)
            else:
                direction = field / magnitude
                projection = np.dot(spins[index], direction)
                spins[index] = 2.0 * projection * direction - spins[index]
                spins[index] /= np.linalg.norm(spins[index])
    return spins


@njit(nogil=True)
def generate_snapshots(j1, j2, count, burn_in, thinning, seed):
    np.random.seed(seed)
    n = j1.shape[0]
    replicas_count = 16
    temperatures = np.zeros(replicas_count)
    for index in range(replicas_count):
        temperatures[index] = 10.0 ** (
            np.log10(5.0)
            + index * (np.log10(0.05) - np.log10(5.0)) / (replicas_count - 1)
        )
    betas = 1.0 / temperatures
    replicas = np.zeros((replicas_count, n, 3))
    for replica in range(replicas_count):
        for site in range(n):
            vector = np.random.randn(3)
            replicas[replica, site] = vector / np.linalg.norm(vector)
    energies = np.zeros(replicas_count)
    for replica in range(replicas_count):
        energies[replica] = total_energy(replicas[replica], j1, j2)

    for step in range(burn_in):
        for replica in range(replicas_count):
            replicas[replica] = sweep(replicas[replica], j1, j2, betas[replica])
            energies[replica] = total_energy(replicas[replica], j1, j2)
        if step % 10 == 0:
            for replica in range(replicas_count - 1):
                delta_beta = betas[replica + 1] - betas[replica]
                delta_energy = energies[replica + 1] - energies[replica]
                if delta_energy * delta_beta > 0.0 or np.random.rand() < np.exp(delta_beta * delta_energy):
                    temporary_spins = replicas[replica].copy()
                    replicas[replica] = replicas[replica + 1]
                    replicas[replica + 1] = temporary_spins
                    temporary_energy = energies[replica]
                    energies[replica] = energies[replica + 1]
                    energies[replica + 1] = temporary_energy

    snapshots = np.zeros((count, n, 3))
    for snapshot in range(count):
        for step in range(thinning):
            for replica in range(replicas_count):
                replicas[replica] = sweep(replicas[replica], j1, j2, betas[replica])
                energies[replica] = total_energy(replicas[replica], j1, j2)
            if step % 10 == 0:
                for replica in range(replicas_count - 1):
                    delta_beta = betas[replica + 1] - betas[replica]
                    delta_energy = energies[replica + 1] - energies[replica]
                    if delta_energy * delta_beta > 0.0 or np.random.rand() < np.exp(delta_beta * delta_energy):
                        temporary_spins = replicas[replica].copy()
                        replicas[replica] = replicas[replica + 1]
                        replicas[replica + 1] = temporary_spins
                        temporary_energy = energies[replica]
                        energies[replica] = energies[replica + 1]
                        energies[replica + 1] = temporary_energy
        snapshots[snapshot] = replicas[-1].copy()
    return snapshots


@njit(nogil=True)
def tangent_forces_for_snapshots(snapshots, j1, j2):
    forces = np.zeros_like(snapshots)
    for snapshot in range(snapshots.shape[0]):
        for site in range(snapshots.shape[1]):
            field = local_field(site, snapshots[snapshot], j1, j2)
            radial = np.dot(field, snapshots[snapshot, site])
            forces[snapshot, site] = field - radial * snapshots[snapshot, site]
    return forces


def load_existing_sequences() -> set[str]:
    sequences: set[str] = set()
    for folder in (TRAIN_DIR, EXISTING_TEST_DIR):
        for path in sorted(folder.glob("chunk_*.npz")):
            with np.load(path, allow_pickle=True) as data:
                sequences.update(str(value) for value in data["sequences"])
    # Exclude every validation/test sequence already inspected in phase 2 so
    # that a newly generated formal test remains genuinely untouched.
    for path in sorted(DATA_DIR.glob("**/chunk_*.npz")):
        with np.load(path, allow_pickle=True) as data:
            sequences.update(str(value) for value in data["sequences"])
    return sequences


def make_unique_sequences(count: int, forbidden: set[str], seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    sequences: list[str] = []
    selected: set[str] = set()
    while len(sequences) < count:
        length = int(rng.integers(20, 51))
        sequence = "".join(rng.choice(np.array(["A", "B"]), size=length).tolist())
        if sequence not in forbidden and sequence not in selected:
            selected.add(sequence)
            sequences.append(sequence)
    return sequences


def worker(
    index: int,
    sequence: str,
    snapshots: int,
    burn_in: int,
    thinning: int,
    base_seed: int,
    include_forces: bool,
):
    j1, j2 = get_couplings(sequence)
    values = generate_snapshots(j1, j2, snapshots, burn_in, thinning, base_seed + index)
    forces = tangent_forces_for_snapshots(values, j1, j2) if include_forces else None
    return index, sequence, values.astype(np.float32), None if forces is None else forces.astype(np.float32)


def save_dataset(
    name: str,
    sequences: list[str],
    snapshots: int,
    burn_in: int,
    thinning: int,
    base_seed: int,
    workers: int,
    include_forces: bool,
) -> None:
    if not sequences:
        print(f"skipping empty dataset {name}", flush=True)
        return
    output_dir = DATA_DIR / name
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = output_dir / "chunk_0000.npz"
    if chunk_path.exists():
        print(f"using existing {chunk_path}", flush=True)
        return
    started = time.time()
    results: list[tuple[int, str, np.ndarray]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                worker,
                index,
                sequence,
                snapshots,
                burn_in,
                thinning,
                base_seed,
                include_forces,
            )
            for index, sequence in enumerate(sequences)
        ]
        for completed, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if completed == 1 or completed % 25 == 0:
                print(f"{name}: {completed}/{len(sequences)} sequences", flush=True)
    results.sort(key=lambda item: item[0])
    sequence_array = np.asarray([item[1] for item in results])
    spin_array = np.empty(len(results), dtype=object)
    force_array = np.empty(len(results), dtype=object) if include_forces else None
    for row, (_index, _sequence, values, forces) in enumerate(results):
        spin_array[row] = values
        if force_array is not None:
            force_array[row] = forces
    payload = {"sequences": sequence_array, "spins": spin_array}
    if force_array is not None:
        payload["tangent_forces"] = force_array
    np.savez_compressed(chunk_path, **payload)
    metadata = {
        "name": name,
        "num_sequences": len(sequences),
        "snapshots_per_sequence": snapshots,
        "burn_in": burn_in,
        "thinning": thinning,
        "base_seed": base_seed,
        "workers": workers,
        "include_forces": include_forces,
        "seconds": time.time() - started,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-sequences", type=int, default=200)
    parser.add_argument("--test-sequences", type=int, default=500)
    parser.add_argument("--snapshots", type=int, default=5)
    parser.add_argument("--burn-in", type=int, default=2000)
    parser.add_argument("--thinning", type=int, default=100)
    parser.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 1))
    parser.add_argument("--tag", default="")
    parser.add_argument("--sequence-seed", type=int, default=20260802)
    parser.add_argument("--test-forces", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    return parser.parse_args()


def main() -> None:
    global DATA_DIR
    args = parse_args()
    DATA_DIR = args.output_dir.resolve()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    forbidden = load_existing_sequences()
    all_new = make_unique_sequences(
        args.validation_sequences + args.test_sequences,
        forbidden,
        seed=args.sequence_seed,
    )
    validation = all_new[: args.validation_sequences]
    test = all_new[args.validation_sequences :]
    manifest = {
        "original_unique_sequences_excluded": len(forbidden),
        "validation_sequences": len(validation),
        "test_sequences": len(test),
        "validation_test_overlap": len(set(validation) & set(test)),
        "new_original_overlap": len((set(validation) | set(test)) & forbidden),
        "sequence_seed": args.sequence_seed,
        "tag": args.tag,
    }
    manifest_name = f"sequence_manifest_{args.tag}.json" if args.tag else "sequence_manifest.json"
    (DATA_DIR / manifest_name).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest), flush=True)

    # Compile Numba kernels before starting the worker pool.
    warm_j1, warm_j2 = get_couplings("AB" * 10)
    generate_snapshots(warm_j1, warm_j2, 1, 2, 2, 1)
    save_dataset(
        f"validation_{args.validation_sequences}{'_' + args.tag if args.tag else ''}",
        validation,
        args.snapshots,
        args.burn_in,
        args.thinning,
        base_seed=310000,
        workers=args.workers,
        include_forces=False,
    )
    save_dataset(
        f"test_{args.test_sequences}{'_' + args.tag if args.tag else ''}",
        test,
        args.snapshots,
        args.burn_in,
        args.thinning,
        base_seed=410000,
        workers=args.workers,
        include_forces=args.test_forces,
    )


if __name__ == "__main__":
    main()
