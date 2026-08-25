from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pairformer_arm3 as arm3


def fixture() -> tuple[dict, np.ndarray, arm3.TrueTopologyBasisPairformer]:
    config = arm3.load_config(ROOT / "configs" / "protocol_v1.json")
    train, _, _, _, _ = arm3.load_coordinate_splits(config)
    transform, blocks, center, scale, _ = arm3.prepare_basis_artifact(config, train, ROOT / "results" / "test_artifact")
    model = arm3.make_model(config, train, transform, blocks, center, scale).float().eval()
    return config, train, model


def test_frozen_basis_counts() -> None:
    config, _, model = fixture()
    assert config["basis"]["raw_count"] == 270
    assert config["basis"]["retained_count"] == 240
    assert sum(model.block_sizes) == 240
    assert model.block_sizes == (35, 49, 41, 21, 53, 41)


def test_scalar_action_is_e3_invariant() -> None:
    _, train, model = fixture()
    positions = torch.as_tensor(train[:2], dtype=torch.float32)
    q, _ = torch.linalg.qr(torch.tensor([[0.3, -0.7, 0.2], [0.8, 0.1, -0.5], [0.2, 0.6, 0.9]]))
    shifted = positions @ q + torch.tensor([0.4, -0.2, 0.1])
    sigma = torch.full((2,), 0.00075)
    first = model(positions, sigma)
    second = model(shifted, sigma)
    assert torch.allclose(first, second, atol=2.0e-5, rtol=2.0e-5)


def test_score_equals_basis_chain_rule() -> None:
    _, train, model = fixture()
    positions = torch.as_tensor(train[:1], dtype=torch.float32).requires_grad_(True)
    sigma = torch.full((1,), 0.00075)
    action = model(positions, sigma)
    direct = -torch.autograd.grad(action.sum(), positions, create_graph=False)[0][0]

    point = positions.detach()[0].requires_grad_(True)
    basis = model.standardized_basis(point.unsqueeze(0))[0]
    leaf = basis.detach().requires_grad_(True)
    action_leaf = model.action_from_basis(leaf.unsqueeze(0), sigma)
    eta = torch.autograd.grad(action_leaf.sum(), leaf)[0]
    jacobian = torch.autograd.functional.jacobian(
        lambda value: model.standardized_basis(value.unsqueeze(0))[0], point, vectorize=True
    )
    chain = -torch.einsum("r,rnc->nc", eta, jacobian)
    assert torch.allclose(direct, chain, atol=2.0e-3, rtol=2.0e-4)


def test_no_constant_law_bias_or_oracle_input() -> None:
    config, _, model = fixture()
    assert not config["architecture"]["constant_coefficient_penalty"]
    assert not config["architecture"]["linear_constant_skip"]
    assert not config["guardrails"]["energy_or_force_labels_used_for_fit_selection"]
    assert not config["guardrails"]["OpenMM_or_GAFF_parameters_used"]
    assert model.audit()["configuration_dependent_effective_coefficients_allowed"]


if __name__ == "__main__":
    tests = [
        test_frozen_basis_counts,
        test_scalar_action_is_e3_invariant,
        test_score_equals_basis_chain_rule,
        test_no_constant_law_bias_or_oracle_input,
    ]
    for test in tests:
        test()
        print(json.dumps({"passed": test.__name__}), flush=True)
