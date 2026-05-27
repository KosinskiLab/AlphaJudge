from __future__ import annotations

import math

from alphajudge.meta_score import (
    META_SCORE_FEATURES,
    calibrated_feature_percentile,
    interface_meta_score,
)


def _complete_row() -> dict[str, float]:
    return {
        "interface_LIS": 0.30,
        "interface_ipSAE": 0.45,
        "interface_pDockQ2": 0.06,
        "iptm": 0.55,
        "confidence_score": 0.62,
        "average_interface_pae": 10.0,
        "pDockQ/mpDockQ": 0.40,
        "interface_sc": 0.50,
        "interface_area": 2300.0,
        "interface_solv_en": -32.0,
    }


def test_meta_score_is_bounded_for_complete_row() -> None:
    score = interface_meta_score(_complete_row())

    assert 0.0 <= score <= 1.0


def test_meta_score_ignores_missing_or_nan_inputs() -> None:
    row = {feature: float("nan") for feature in META_SCORE_FEATURES}
    row["interface_LIS"] = 0.30

    assert interface_meta_score(row) == calibrated_feature_percentile("interface_LIS", 0.30)


def test_meta_score_returns_nan_when_all_inputs_missing() -> None:
    assert math.isnan(interface_meta_score({}))


def test_inverted_features_score_in_the_expected_direction() -> None:
    low_pae = calibrated_feature_percentile("average_interface_pae", 4.0)
    high_pae = calibrated_feature_percentile("average_interface_pae", 30.0)
    strong_solvation = calibrated_feature_percentile("interface_solv_en", -50.0)
    weak_solvation = calibrated_feature_percentile("interface_solv_en", -5.0)

    assert low_pae is not None and high_pae is not None
    assert strong_solvation is not None and weak_solvation is not None
    assert low_pae > high_pae
    assert strong_solvation > weak_solvation


def test_feature_percentiles_clamp_to_unit_interval() -> None:
    assert calibrated_feature_percentile("interface_LIS", -1.0) == 0.0
    assert calibrated_feature_percentile("interface_LIS", 10.0) == 1.0
