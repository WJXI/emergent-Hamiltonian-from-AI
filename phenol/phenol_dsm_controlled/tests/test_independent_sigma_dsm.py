from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "independent_sigma_dsm_experiment.py"
SPEC = importlib.util.spec_from_file_location("independent_sigma_dsm_experiment", SOURCE)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_v2_protocol_is_independent_sigma_and_label_free() -> None:
    config = module.load_protocol(ROOT / "configs" / "clean_limit_protocol_v2.json")
    assert config["common_model"]["coefficient_coupling_across_sigma"] is False
    assert config["dsm"]["noise_sigmas_nm"] == [0.000375, 0.00075, 0.0015]
    assert len(config["dsm"]["fit_noise_seeds"]) == 3
    assert config["guardrails"]["energy_or_force_labels_for_fit_selection"] is False


def test_sigma_squared_extrapolation_recovers_clean_coefficient() -> None:
    clean = np.asarray([1.0, -2.0, 0.5])
    bias = np.asarray([4.0, 3.0, -7.0])
    low, high = 0.000375, 0.00075
    eta_low = clean + low**2 * bias
    eta_high = clean + high**2 * bias
    recovered = module.extrapolate_clean(eta_low, low, eta_high, high)
    assert np.allclose(recovered, clean, atol=1.0e-12, rtol=1.0e-12)


def test_array_metric_identity() -> None:
    vector = np.asarray([1.0, 2.0])
    metric = module.array_metric(vector, vector, np.eye(2))
    assert metric["score_cosine"] == 1.0
    assert metric["score_norm_ratio_prediction_to_reference"] == 1.0
    assert metric["score_rms_difference_over_reference"] == 0.0
