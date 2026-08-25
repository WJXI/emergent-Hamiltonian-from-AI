from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "unified_dsm_experiment.py"
SPEC = importlib.util.spec_from_file_location("unified_dsm_experiment", SOURCE)
assert SPEC is not None and SPEC.loader is not None
dsm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dsm)


def test_protocol_enforces_the_fixed_structured_network() -> None:
    config = dsm.load_protocol(ROOT / "configs" / "clean_limit_protocol_v2.json")
    assert tuple(config["arms"]) == dsm.ARM_NAMES
    assert config["common_model"]["raw_basis_count"] == 270
    assert config["guardrails"]["energy_or_force_labels_for_fit_selection"] is False
    assert config["guardrails"]["shared_law_residual_decomposition"] is False


def test_coordinate_loader_never_opens_legacy_test() -> None:
    config = dsm.load_protocol(ROOT / "configs" / "clean_limit_protocol_v2.json")
    train, train_family, validation, validation_family, audit = dsm.load_coordinate_splits(config)
    assert train.shape == (5000, 13, 3)
    assert validation.shape == (1000, 13, 3)
    assert sorted(np.unique(train_family).tolist()) == [0, 1, 2, 3, 4]
    assert np.unique(validation_family).tolist() == [5]
    assert audit["legacy_test_opened"] is False
    assert audit["force_energy_or_component_arrays_opened"] is False


def test_structured_dictionary_has_270_scalar_functions() -> None:
    config = dsm.load_protocol(ROOT / "configs" / "clean_limit_protocol_v2.json")
    train, *_ = dsm.load_coordinate_splits(config)
    coordinates = train[:8]
    for arm in dsm.ARM_NAMES:
        dictionary, audit = dsm.build_dictionary(arm, config, coordinates)
        value = dictionary(torch.as_tensor(coordinates[:2], dtype=torch.float64))
        assert dictionary.basis_count == 270
        assert value.shape == (2, 270)
        assert torch.isfinite(value).all()
        assert audit["basis_count"] == 270
class ToyDictionary(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("pair", torch.tensor([[0, 1], [1, 2]], dtype=torch.long))
        self.basis_count = 2
        self.basis_block = ("toy", "toy")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = x[:, self.pair[:, 1]] - x[:, self.pair[:, 0]]
        return torch.linalg.vector_norm(delta, dim=-1)


def test_dsm_moments_are_symmetric_finite_and_reproducible() -> None:
    rng = np.random.default_rng(7)
    coordinates = rng.normal(size=(5, 3, 3))
    coordinates -= coordinates.mean(axis=1, keepdims=True)
    dictionary = ToyDictionary().to(dtype=torch.float64)
    transform = np.eye(2)
    first = dsm.compute_dsm_moments(dictionary, transform, coordinates, [0.001, 0.002], 0.002, 1, 99, 3)
    second = dsm.compute_dsm_moments(dictionary, transform, coordinates, [0.001, 0.002], 0.002, 1, 99, 2)
    assert first["a"].shape == (4, 4)
    assert first["b"].shape == (4,)
    assert np.allclose(first["a"], first["a"].T)
    eigenvalues = torch.linalg.eigvalsh(torch.as_tensor(first["a"], dtype=torch.float64)).cpu().numpy()
    assert np.all(eigenvalues > -1.0e-10)
    assert np.allclose(first["a"], second["a"], atol=1.0e-12)
    assert np.allclose(first["b"], second["b"], atol=1.0e-12)
    assert first["first_noise_sha256"] == second["first_noise_sha256"]


def test_ridge_solver_improves_a_well_conditioned_quadratic() -> None:
    matrix = np.diag([2.0, 3.0, 4.0, 5.0])
    vector = np.asarray([1.0, -2.0, 0.5, 1.5])
    moments = {"a": matrix, "b": vector, "baseline": 10.0}
    theta, summary = dsm.solve_ridges(moments, moments, [1.0e-8, 1.0e-3])
    assert dsm.objective(theta, moments) < moments["baseline"]
    assert summary["selected_validation_fractional_gain_over_zero_score"] > 0.0
