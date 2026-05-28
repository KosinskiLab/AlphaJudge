from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Mapping
from typing import Any

META_SCORE_FEATURES = (
    "interface_LIS",
    "interface_ipSAE",
    "interface_pDockQ2",
    "iptm",
    "confidence_score",
    "average_interface_pae",
    "pDockQ/mpDockQ",
    "interface_sc",
    "interface_hb",
    "interface_solv_en",
)

FEATURE_DIRECTIONS = {
    "interface_LIS": 1.0,
    "interface_ipSAE": 1.0,
    "interface_pDockQ2": 1.0,
    "iptm": 1.0,
    "confidence_score": 1.0,
    "average_interface_pae": -1.0,
    "pDockQ/mpDockQ": 1.0,
    "interface_sc": 1.0,
    "interface_hb": 1.0,
    "interface_area": 1.0,
    "interface_solv_en": -1.0,
}

CALIBRATION_LEVELS = (
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
)

# Frozen deciles from the benchmark_26 final synchronized best-interface run
# (7,756 AF2/AF3 positive/negative rows; final_sync_20260523_225722, after
# the missing pair-matched predictions were back-filled). Values are already
# oriented so larger is better; e.g. PAE and solvation energy are stored
# after sign flip.
BENCHMARK_QUANTILES = {
    "interface_LIS": (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.04087949643906165,
        0.13930053033306905,
        0.2748901434822449,
        0.3884159952769791,
        0.5095176486875836,
        0.7683597309793258,
    ),
    "interface_ipSAE": (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.01145996318417455,
        0.04849651380227124,
        0.39191523664041816,
        0.631408256291724,
        0.7685336920803372,
        0.955598788837354,
    ),
    "interface_pDockQ2": (
        0.0,
        0.0088894923938337,
        0.0093154249757626,
        0.0097853519732417,
        0.0105167871915631,
        0.0121137962077202,
        0.01700470113788611,
        0.044079106104955995,
        0.147310180270119,
        0.4604512911372348,
        0.950422615923692,
    ),
    "iptm": (
        0.04,
        0.15,
        0.18,
        0.21,
        0.2422413975000381,
        0.2993520200252533,
        0.38812878727913014,
        0.51,
        0.66,
        0.8086847960948944,
        0.9710875749588012,
    ),
    "confidence_score": (
        -99.73,
        0.24,
        0.279434748489446,
        0.32,
        0.3648142071999158,
        0.42,
        0.5,
        0.6002808797233267,
        0.7262544258927405,
        0.84,
        1.17,
    ),
    "average_interface_pae": (
        -31.466666666666665,
        -28.001159075666123,
        -26.179307692307695,
        -24.495731538992413,
        -22.704047619047614,
        -20.205151515151503,
        -17.110826882477134,
        -11.98133340586016,
        -7.214102564102556,
        -3.7483788429752054,
        -1.0446969696969706,
    ),
    "pDockQ/mpDockQ": (
        0.0,
        0.04342955378732375,
        0.0744205498951194,
        0.11156935841275703,
        0.154663210767691,
        0.20778370916876399,
        0.28249571167245185,
        0.3758437409692674,
        0.4920927853075837,
        0.6177077320794855,
        0.7403745371795283,
    ),
    "interface_sc": (
        -0.0909274325112387,
        0.2727699720080689,
        0.3812050398280308,
        0.4213154557548237,
        0.4488086853592169,
        0.47065393187337334,
        0.4938328663057683,
        0.5188898407198262,
        0.54885285443392,
        0.5914616385120754,
        0.744024124091351,
    ),
    # Deciles for interface_hb replace interface_area in METASCORE: H-bond
    # count is interpretable, only weakly correlated with interface_sc, and
    # captures specific polar interactions, whereas area was strongly
    # redundant with solvation energy (Pearson rho = -0.80 on the same
    # benchmark).
    "interface_hb": (
        0.0,
        2.0,
        4.0,
        6.0,
        8.0,
        10.0,
        12.0,
        15.0,
        20.0,
        28.0,
        129.0,
    ),
    "interface_area": (
        0.0,
        542.1321074137181,
        813.3952069762071,
        1038.3245889227087,
        1289.939806795484,
        1559.6427719907642,
        1897.2281626043305,
        2361.926012313381,
        2928.3984815923395,
        3943.2489991553953,
        19027.209490273777,
    ),
    "interface_solv_en": (
        -26.14067293563187,
        0.03920834959131975,
        2.8962665146694064,
        5.443213873630101,
        8.057221500635805,
        10.930657812354951,
        14.625751361301866,
        19.243051691872445,
        26.38771636631324,
        38.319393210666135,
        233.00683345812263,
    ),
}


def _safe_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed if math.isfinite(parsed) else float("nan")


def calibrated_feature_percentile(feature: str, value: Any) -> float | None:
    """Map a raw feature value onto the frozen benchmark percentile scale."""
    if feature not in BENCHMARK_QUANTILES:
        raise KeyError(f"unknown metascore feature: {feature}")

    raw = _safe_float(value)
    if math.isnan(raw):
        return None

    oriented = raw * FEATURE_DIRECTIONS[feature]
    quantiles = BENCHMARK_QUANTILES[feature]
    levels = CALIBRATION_LEVELS

    if oriented <= quantiles[0]:
        return levels[0]
    if oriented >= quantiles[-1]:
        return levels[-1]

    lower_idx = bisect_right(quantiles, oriented) - 1
    lower_idx = max(0, min(lower_idx, len(quantiles) - 2))
    q0 = quantiles[lower_idx]
    q1 = quantiles[lower_idx + 1]
    p0 = levels[lower_idx]
    p1 = levels[lower_idx + 1]

    if oriented == q0 or q1 <= q0:
        return p0
    fraction = (oriented - q0) / (q1 - q0)
    return p0 + fraction * (p1 - p0)


def interface_meta_score(row: Mapping[str, Any]) -> float:
    """
    Transparent rank-style interface metascore.

    Each selected AlphaJudge feature is converted to a frozen benchmark
    percentile where higher means stronger interaction evidence. Missing or
    non-finite inputs are ignored. The final score is the mean percentile.
    """
    percentiles = [
        percentile
        for feature in META_SCORE_FEATURES
        if (percentile := calibrated_feature_percentile(feature, row.get(feature))) is not None
    ]
    if not percentiles:
        return float("nan")
    return float(sum(percentiles) / len(percentiles))
