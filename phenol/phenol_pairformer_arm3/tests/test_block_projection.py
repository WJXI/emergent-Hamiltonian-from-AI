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

import block_projection as block  # noqa: E402
import pairformer_arm3 as arm3  # noqa: E402
import projection_utils as emergence  # noqa: E402


def fixture() -> tuple[dict, np.ndarray, arm3.TrueTopologyBasisPairformer]:
    config = arm3.load_config(ROOT / "configs" / "protocol_v1.json")
    train, _, _, _, _ = arm3.load_coordinate_splits(config)
    transform, blocks, center, scale, _ = arm3.prepare_basis_artifact(
        config, train, ROOT / "results" / "test_artifact"
    )
    model = arm3.make_model(config, train, transform, blocks, center, scale).float().eval()
    return config, train, model


def test_protocol_is_sector_preserving_and_label_free() -> None:
    parent = arm3.load_config(ROOT / "configs" / "protocol_v1.json")
    protocol = block.load_projection_protocol(ROOT / "configs" / "block_projection_protocol_v1.json")
    assert protocol["definition"].startswith("Network 3-P")
    assert protocol["projection"]["fit_families"] == parent["splits"]["fit_families"]
    assert protocol["projection"]["selection_families"] == parent["splits"]["selection_families"]
    assert not protocol["projection"]["physical_labels_used_for_fit_or_selection"]


def test_separate_constant_blocks_sum_to_one_constant_action() -> None:
    config, train, model = fixture()
    device = torch.device("cpu")
    model = model.to(device)
    rng = np.random.default_rng(260822)
    alpha = rng.normal(scale=0.01, size=sum(model.block_sizes))
    pieces = []
    for left, right in model.block_slices:
        pieces.append(block.constant_block_score(model, train[:3], alpha[left:right], (left, right), 3, device))
    reconstructed = np.sum(np.stack(pieces), axis=0)
    direct = emergence.constant_score(model, train[:3], alpha, 3, device)
    audit = block.exact_sum_audit(reconstructed, direct)
    assert audit["rms_error_over_direct_score_rms"] < 2.0e-5
    assert config["basis"]["retained_count"] == alpha.size


if __name__ == "__main__":
    tests = [test_protocol_is_sector_preserving_and_label_free, test_separate_constant_blocks_sum_to_one_constant_action]
    for test in tests:
        test()
        print(json.dumps({"passed": test.__name__}), flush=True)
