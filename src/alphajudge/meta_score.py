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
    "interface_contact_prob_top10_mean",
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
    "interface_contact_prob_top10_mean": 1.0,
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

# Frozen deciles from the AlphaJudge interacting reference set. Calibrated on
# POSITIVE (interacting) pairs ONLY: 3,878 AF2/AF3 positive rows out of the
# 7,756-row balanced table. The database-negative re-pairings are deliberately
# excluded so a new prediction is ranked against the distribution of real
# interfaces, not against a 50% non-interacting decoy population. Regenerate
# manually with
# `python test/manual/freeze_metascore_quantiles.py --input-csv ... --label-filter positive`.
# Values are already oriented so larger is better; e.g. PAE and solvation
# energy are stored after sign flip.
BENCHMARK_QUANTILES = {
    "interface_LIS": (
        0.0,
        0.0,
        0.04060645514906991,
        0.16033887103863478,
        0.2576658136092226,
        0.32442128815185134,
        0.3851385587805126,
        0.44345538655101113,
        0.5088408754804322,
        0.5774309078720992,
        0.7683597309793258,
    ),
    "interface_ipSAE": (
        0.0,
        0.0,
        0.010959105568327996,
        0.11185110880195845,
        0.3555340463896866,
        0.5186285859409205,
        0.6285148344492705,
        0.702100751375003,
        0.7684464485394504,
        0.8257356696917769,
        0.955598788837354,
    ),
    "interface_pDockQ2": (
        0.0,
        0.009453880310358351,
        0.0111891126550556,
        0.016886274027138628,
        0.03397041297897107,
        0.07344648145371385,
        0.14133308419757337,
        0.2696698397779966,
        0.45400387748598986,
        0.6963144657009852,
        0.950422615923692,
    ),
    "iptm": (
        0.08,
        0.2,
        0.271808648109436,
        0.37,
        0.4707051336765289,
        0.5603788185119629,
        0.6500772428512572,
        0.73,
        0.8022430896759034,
        0.87,
        0.9710875749588012,
    ),
    "confidence_score": (
        -99.73,
        0.29009815345375634,
        0.38,
        0.4718669364580657,
        0.5582932754510956,
        0.6378477968232499,
        0.7118574504742852,
        0.78,
        0.8399975446618876,
        0.8912135719685363,
        1.16,
    ),
    "average_interface_pae": (
        -31.39791666666667,
        -25.547096774193545,
        -21.96980016550708,
        -17.9543,
        -13.584722222222224,
        -10.055666561613616,
        -7.456405228758169,
        -5.489171195652174,
        -3.7889943074003773,
        -2.4152747252747253,
        -1.0446969696969706,
    ),
    "pDockQ/mpDockQ": (
        0.0,
        0.08588056107981762,
        0.16023056468016583,
        0.24182412751855897,
        0.3247870448351307,
        0.4041803181799995,
        0.48010708308928685,
        0.5508260817138925,
        0.6163167605164049,
        0.6713316493073628,
        0.7403745371795283,
    ),
    "interface_sc": (
        -0.0909274325112387,
        0.3767070462297797,
        0.43997561327658047,
        0.46854106915590277,
        0.4903169925194296,
        0.5111733070545363,
        0.5327125306473834,
        0.5566747944324164,
        0.5840561089134118,
        0.6193315076367524,
        0.744024124091351,
    ),
    # Deciles for interface_hb replace interface_area in METASCORE: H-bond
    # count is interpretable, only weakly correlated with interface_sc, and
    # captures specific polar interactions, whereas area was strongly
    # redundant with solvation energy (Pearson rho = -0.80 on the same
    # benchmark).
    "interface_hb": (
        0.0,
        4.0,
        6.0,
        9.0,
        11.0,
        13.0,
        16.0,
        20.0,
        25.0,
        34.0,
        129.0,
    ),
    "interface_area": (
        0.0,
        811.1528034047047,
        1107.604424428278,
        1420.6931573271686,
        1716.9149871911516,
        2073.670645394412,
        2488.5877792892425,
        2942.3373065313444,
        3529.63873252237,
        4669.550388777939,
        17039.462876150043,
    ),
    "interface_solv_en": (
        -26.14067293563187,
        2.3417733842449655,
        6.058883462167398,
        9.549507514423109,
        13.314712839646953,
        17.308406303470562,
        22.06634539018819,
        28.33697255981416,
        36.16186391880063,
        48.35158049464502,
        233.00683345812263,
    ),
    # Calibrated on the FULL UNFILTERED v3 benchmark's positives (n=12,163;
    # af2 6,036 + af3 6,127), pooled over both backends -- unlike the entries
    # above, which come from the filtered benchmark. That benchmark predates the
    # contact-probability score and carries no interface_contact_prob_* column,
    # so this ladder can only come from full v3. Cost of the mixed calibration
    # source is small: refreezing the ten older features on full-v3 positives
    # moves an assigned percentile by 0.044 on average and changes the metascore
    # AUROC by at most 0.0011.
    #
    # NOTE (backend scale): AF2 contact probabilities are distogram-derived while
    # AF3 uses the model's native contact head, so the two are not on a common
    # scale and AF2 rows land at a systematically higher percentile on this
    # pooled ladder. Within-backend ranking -- what the score is used for -- is
    # unaffected.
    "interface_contact_prob_top10_mean": (
        0.0,
        0.19286754713058474,
        0.39218458712100984,
        0.6249195265769958,
        0.777816823720932,
        0.875,
        0.937,
        0.974,
        0.991,
        0.9983325207233429,
        1.0,
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
