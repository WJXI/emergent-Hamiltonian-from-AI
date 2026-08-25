"""Fast integrity and paper-number audit for the public code bundle."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(actual: float, expected: float, label: str, tolerance: float = 5.0e-12) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"{label}: {actual} != {expected}")


def verify_spin(paper: dict[str, Any]) -> None:
    root = ROOT / "spin_glass"
    checkpoint = root / "checkpoints" / "epoch_26.pt"
    if sha256(checkpoint) != paper["checkpoint_sha256"]:
        raise AssertionError("epoch-26 checkpoint hash mismatch")

    test_dir = root / "data" / "test_2000"
    manifest = load_json(test_dir / "CANONICAL_TESTSET_MANIFEST.json")
    if manifest["num_sequences"] != 2000 or manifest["training_sequence_overlap"] != 0:
        raise AssertionError("canonical independent-test manifest changed")
    for item in manifest["chunks"]:
        if sha256(test_dir / item["name"]) != item["sha256"]:
            raise AssertionError(f"canonical test chunk changed: {item['name']}")

    wj = load_json(root / "results" / "wj_metrics.json")
    close(wj["pearson_W_J"], paper["pearson_W_J"], "spin W-J Pearson")
    close(wj["scale_W_per_J"], paper["class_balanced_a_star"], "spin a_star")
    close(wj["class_ratio_relative_rms"], paper["class_ratio_relative_rms"], "spin class RMS")
    close(wj["pointwise_relative_rmse"], paper["pointwise_relative_rmse"], "spin pointwise RMSE")
    close(wj["configuration_relative_rms"], paper["configuration_relative_rms"], "spin configuration RMS")

    locality = load_json(root / "results" / "locality_metrics.json")
    ratio = locality["far_to_support_absolute_mass_ratio_all_r_ge_3"]
    close(ratio, paper["far_to_support_absolute_weight_ratio"], "spin locality mass ratio")
    close(1.0 / (1.0 + ratio), paper["support_absolute_weight_fraction"], "spin support fraction")

    diagnostics = load_json(root / "results" / "identifiability.json")
    close(
        diagnostics["representation"]["shared_six_class_design_condition_number"],
        paper["shared_six_class_design_condition_number"],
        "six-class design condition number",
    )
    overlap = load_json(root / "results" / "aligned_overlap_audit.json")
    close(
        overlap["within_run_by_saved_lag"]["1"]["mean_overlap"]["point"],
        paper["aligned_overlap"]["within_run_100_sweeps"],
        "within-run overlap",
    )
    close(
        overlap["independent_run_baseline"]["point"],
        paper["aligned_overlap"]["independent_run_baseline"],
        "independent-run overlap",
    )


def verify_phenol_data() -> None:
    phenol = ROOT / "phenol"
    manifest_path = phenol / "data" / "fresh_physical_v1" / "model_pilot_v1" / "dataset_manifest.json"
    manifest = load_json(manifest_path)
    for split in ("train", "validation"):
        item = manifest["split_files"][split]
        path = manifest_path.parent / item["path"]
        if sha256(path) != item["sha256"]:
            raise AssertionError(f"phenol {split} hash mismatch")
        with np.load(path, allow_pickle=False) as values:
            if sorted(values.files) != ["family_id", "phenol_coordinates_nm"]:
                raise AssertionError(f"unexpected arrays in phenol {split}")
            if values["phenol_coordinates_nm"].shape != (item["frames"], 13, 3):
                raise AssertionError(f"unexpected phenol {split} shape")

    holdout = phenol / "phenol_dsm_controlled" / "results" / "final_holdout_v1"
    with np.load(holdout / "selected_coordinates.npz", allow_pickle=False) as values:
        if values["phenol_coordinates_nm"].shape != (1200, 13, 3):
            raise AssertionError("final phenol holdout coordinate shape changed")
        if sorted(map(int, np.unique(values["family_id"]))) != list(range(8)):
            raise AssertionError("final phenol holdout families changed")
    with np.load(holdout / "raw_openmm_force_components.npz", allow_pickle=False) as values:
        for name in ("total", "bond", "angle", "proper", "improper", "nonbonded"):
            if values[name].shape != (1200, 13, 3):
                raise AssertionError(f"physical holdout component changed: {name}")


def verify_phenol_metrics(paper: dict[str, Any]) -> None:
    phenol = ROOT / "phenol"
    evaluation = load_json(phenol / "paper_evaluation.json")
    source = evaluation["metrics"]
    source_keys = {
        "total": "total",
        "bond": "bond",
        "angle": "angle",
        "proper_torsion": "proper",
        "out_of_plane": "out_of_plane_vs_improper",
        "graph_distance_3_pair": "graph_distance_3_vs_exact_exception",
        "longer_range_pair": "graph_distance_gt_3_vs_pme_real",
    }
    for network, reported in paper["metrics"].items():
        for sector, expected in reported.items():
            actual = source[network][source_keys[sector]]
            close(actual["cosine"], expected["cosine"], f"{network} {sector} cosine", 1.0e-5)
            close(actual["nrmse"], expected["nrmse"], f"{network} {sector} NRMSE", 1.0e-5)
    close(
        evaluation["network_3_P_R_const_squared"],
        paper["network_3_P_R_const_squared"],
        "Network 3-P R_const_squared",
        1.0e-5,
    )
    if evaluation["physical_labels_used_for_training_projection_or_selection"]:
        raise AssertionError("Phenol physical-label firewall changed")

    manifest = load_json(phenol / "frozen_models.json")
    for item in manifest["artifacts"]:
        path = phenol / item["path"]
        if sha256(path) != item["sha256"]:
            raise AssertionError(f"frozen artifact changed: {path}")


def main() -> None:
    paper = load_json(ROOT / "paper_results.json")
    verify_spin(paper["spin_glass"])
    verify_phenol_data()
    verify_phenol_metrics(paper["phenol"])
    forbidden = list(ROOT.rglob("*.running.pt"))
    if forbidden:
        raise AssertionError(f"unfinished checkpoints were packaged: {forbidden}")
    legacy_paths = [
        ROOT / "phenol" / "phenol_v2",
        ROOT / "phenol" / "src" / "thermo_repro",
        ROOT / "phenol" / "data" / "fresh_physical_v1" / "model_pilot_v1" / "test.npz",
    ]
    if any(path.exists() for path in legacy_paths):
        raise AssertionError("a superseded Phenol experiment was packaged")
    print("Release verification passed: hashes, data splits, and paper metrics are consistent.")


if __name__ == "__main__":
    main()
