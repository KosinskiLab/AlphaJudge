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
    "interface_area",
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

# Frozen deciles from the benchmark_26 all-organism best-interface run
# (7,345 AF2/AF3 positive/negative rows; April 22, 2026 merged table).
# Values are already oriented so larger is better; e.g. PAE and solvation
# energy are stored after sign flip.
BENCHMARK_QUANTILES = {
    "interface_LIS": (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.059720202020202,
        0.16705226076310281,
        0.29334500115446349,
        0.40119327202506005,
        0.51623890758784829,
        0.75338038596492163,
    ),
    "interface_ipSAE": (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0131470059312316,
        0.10584161048830307,
        0.43384587935260738,
        0.64412107036419464,
        0.77087459523243229,
        0.95559878883735405,
    ),
    "interface_pDockQ2": (
        0.0075494887493505998,
        0.0089492092351328,
        0.0093933511858568398,
        0.0098769987230639799,
        0.010703449712070559,
        0.0124802347229461,
        0.018771163348387087,
        0.050761707493646967,
        0.17282606942497017,
        0.47580370600410854,
        0.95042261592369204,
    ),
    "iptm": (
        0.040000000000000001,
        0.14999999999999999,
        0.17999999999999999,
        0.21093810200691221,
        0.25285084247589107,
        0.31,
        0.40999999999999998,
        0.53254582881927492,
        0.67266757488250806,
        0.81000000000000005,
        0.97108757495880116,
    ),
    "confidence_score": (
        0.12678638003217149,
        0.23979955960633545,
        0.28000000000000003,
        0.32000000000000001,
        0.37,
        0.42999999999999999,
        0.51000000000000001,
        0.61354833086340177,
        0.73788607560146302,
        0.84101645584207918,
        1.1499999999999999,
    ),
    "average_interface_pae": (
        -31.5,
        -27.859964220839984,
        -26.073899561846982,
        -24.438900954877365,
        -22.36344086021505,
        -20.006129032258062,
        -16.573059698130209,
        -11.591519818964887,
        -6.7923660714285621,
        -3.6440909090909086,
        -1.0446969696969697,
    ),
    "pDockQ/mpDockQ": (
        0.018258893561979499,
        0.048401733659705799,
        0.080583999776873999,
        0.11763635331753999,
        0.1609442064816457,
        0.21622604840653531,
        0.28893407672657195,
        0.39005367493568538,
        0.502151558453238,
        0.62055885656958631,
        0.74034050365479986,
    ),
    "interface_sc": (
        -0.20978600009076619,
        0.19474781603546867,
        0.25907145776977002,
        0.32226136048686279,
        0.38997362215252516,
        0.42865133273885592,
        0.46107348285381024,
        0.49758078915313065,
        0.54187397531551566,
        0.59373881458964451,
        0.74402412409164909,
    ),
    "interface_area": (
        12.696379714313343,
        581.59180907251502,
        830.76128351670551,
        1067.0180525575927,
        1311.9445584680248,
        1586.0792814579392,
        1912.6781164897349,
        2333.6612364858556,
        2898.2619149565764,
        3914.9127720160723,
        23847.404872232858,
    ),
    "interface_solv_en": (
        -26.140681977743796,
        0.38162628724614497,
        3.1598266879255246,
        5.8459112057851663,
        8.5967875314312085,
        11.475426312687474,
        15.228084355320659,
        20.067672952580592,
        27.302842021122294,
        39.330244540335536,
        400.66271273444442,
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
