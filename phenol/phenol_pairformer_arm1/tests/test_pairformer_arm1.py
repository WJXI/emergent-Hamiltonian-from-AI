from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("pairformer_arm1", ROOT / "src" / "pairformer_arm1.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def config():
    return MODULE.load_config(ROOT / "configs" / "protocol_v1.json")


def test_scalar_action_and_score_shape():
    model = MODULE.GraphFreeScalarPairformer(config())
    positions = torch.randn(3, 13, 3) * 0.1
    sigma = torch.full((3,), 0.00075)
    action, score = MODULE.action_score(model, positions, sigma, create_graph=False)
    assert action.shape == (3,)
    assert score.shape == positions.shape


def test_translation_and_rotation_invariance():
    torch.manual_seed(11)
    model = MODULE.GraphFreeScalarPairformer(config()).eval()
    positions = torch.randn(2, 13, 3) * 0.1
    sigma = torch.full((2,), 0.00075)
    angle = torch.tensor(0.73)
    rotation = torch.tensor([[torch.cos(angle), -torch.sin(angle), 0.0], [torch.sin(angle), torch.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    translated = positions @ rotation.T + torch.tensor([1.2, -0.5, 0.8])
    assert torch.allclose(model(positions, sigma), model(translated, sigma), atol=2.0e-5, rtol=2.0e-5)


def test_identical_element_permutation_invariance():
    torch.manual_seed(17)
    model = MODULE.GraphFreeScalarPairformer(config()).eval()
    positions = torch.randn(2, 13, 3) * 0.1
    sigma = torch.full((2,), 0.00075)
    permutation = torch.tensor([2, 1, 0, 3, 4, 5, 6, 9, 8, 7, 10, 11, 12])
    assert torch.equal(model.element_id, model.element_id[permutation])
    assert torch.allclose(model(positions, sigma), model(positions[:, permutation], sigma), atol=2.0e-5, rtol=2.0e-5)


def test_no_topology_inputs_or_atom_index_embedding():
    audit = MODULE.GraphFreeScalarPairformer(config()).audit()
    assert not audit["bond_graph_used"]
    assert not audit["atom_index_embedding_used"]
    assert not audit["component_heads"]
