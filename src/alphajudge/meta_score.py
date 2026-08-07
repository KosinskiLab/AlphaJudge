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
    "interface_ccc": 1.0,
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
# `python test/manual/freeze_metascore_quantiles.py --input-csv ... --ccc-csv ...
# --label-filter positive`.
# Values are already oriented so larger is better; e.g. PAE and solvation
# energy are stored after sign flip.
BENCHMARK_QUANTILES = {
    # full non-downsampled v3, POSITIVES only, pooled af2+af3 (n=12163)
    "interface_LIS": (
        0.0,
        0.0,
        0.0,
        0.07756424252283722,
        0.18524380120629177,
        0.2694273631458537,
        0.3394504436714632,
        0.4034279791029212,
        0.47235041723583504,
        0.5545265096282466,
        0.766488622764907,
    ),
    "interface_ipSAE": (
        0.0,
        0.0,
        0.0,
        0.014342347529995107,
        0.16443610007454396,
        0.3840602212414786,
        0.5442805932504046,
        0.6507097630647123,
        0.7327642399042628,
        0.8024249781037394,
        0.9624138446279216,
    ),
    "interface_pDockQ2": (
        0.0075248486093233095,
        0.009213655629047478,
        0.010122835597296627,
        0.01225301812975495,
        0.018289299933913487,
        0.03438269221147416,
        0.07830205723803,
        0.16821218947135655,
        0.337451821492259,
        0.6003359400361069,
        0.9509054047439837,
    ),
    "iptm": (
        0.06,
        0.1846581518650055,
        0.24,
        0.31,
        0.4,
        0.49638617038726807,
        0.59,
        0.68,
        0.77,
        0.854150688648224,
        0.9712996482849121,
    ),
    "confidence_score": (
        -99.76,
        0.2744943321025725,
        0.34,
        0.41905546686959233,
        0.4969541073015846,
        0.58,
        0.66,
        0.74,
        0.81,
        0.88,
        1.16,
    ),
    "average_interface_pae": (
        -31.55882352941177,
        -26.88919240506329,
        -23.994221938775514,
        -20.9049403794038,
        -17.38557638514011,
        -13.536519607843147,
        -9.828304639626491,
        -6.992307692307693,
        -4.763490534427875,
        -2.910195121951216,
        -1.0173913043478262,
    ),
    "pDockQ/mpDockQ": (
        0.018258893561979524,
        0.07822316003812757,
        0.1396137274727143,
        0.2079517713066971,
        0.2793161002766122,
        0.35370198617585397,
        0.4298773056578562,
        0.5051912964577867,
        0.5780019217731105,
        0.645862480041125,
        0.7402866068322936,
    ),
    "interface_sc": (
        -0.20150967675772263,
        0.24047402147688657,
        0.3303891211815527,
        0.4030438532707223,
        0.4446699591794004,
        0.47664727757059616,
        0.5067974239228107,
        0.5387955071430409,
        0.5724287410635969,
        0.6128145255311569,
        0.7616072028183929,
    ),
    "interface_hb": (
        0.0,
        3.0,
        6.0,
        8.0,
        10.0,
        13.0,
        16.0,
        19.399999999999636,
        25.0,
        34.0,
        219.0,
    ),
    "interface_solv_en": (
        -27.256826262904255,
        1.936796343828263,
        5.658609310544915,
        8.746550074367514,
        11.707498196341898,
        15.680071964927663,
        20.082169059498153,
        25.510605941898184,
        33.17123316941061,
        45.624946979545825,
        404.59852857526937,
    ),
    "interface_contact_prob_top10_mean": (
        0.0,
        0.14026100000000002,
        0.30199999999999994,
        0.532,
        0.725,
        0.8389999999999999,
        0.909,
        0.9577936,
        0.982,
        0.9944468,
        1.0,
    ),
    # full non-downsampled v3, POSITIVES with a parsed CCC value and a scored
    # interface, pooled af2+af3 (n=12160)
    "interface_ccc": (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        16.0,
        38.0,
        56.0,
        78.0,
        115.0,
        781.0,
    ),
}


BENCHMARK_QUANTILES_BY_BACKEND = {
    # full non-downsampled v3, POSITIVES only, af2 (n=6036)
    "af2": {
        "interface_LIS": (
            0.0,
            0.0,
            0.0161036036036036,
            0.1094453138066084,
            0.2143402556137805,
            0.3018936325360668,
            0.36712178162341574,
            0.4246419059059224,
            0.4908848196394652,
            0.567843979908057,
            0.766488622764907,
        ),
        "interface_ipSAE": (
            0.0,
            0.0,
            0.0,
            0.018010338960813776,
            0.24099553282511965,
            0.4590036102851777,
            0.6010500212828368,
            0.7019194429940772,
            0.7664003285906783,
            0.8216770525309647,
            0.9624138446279216,
        ),
        "interface_pDockQ2": (
            0.0075248486093233095,
            0.009062808932344572,
            0.010065060645680511,
            0.012138641657128076,
            0.017494532140134873,
            0.03379250106250077,
            0.08615090635639835,
            0.19385413618733893,
            0.4138520536017136,
            0.667989817076277,
            0.9509054047439837,
        ),
        "iptm": (
            0.08530029654502869,
            0.2029327005147934,
            0.2612302899360657,
            0.33858415484428406,
            0.43042290210723877,
            0.526773989200592,
            0.6227998733520508,
            0.7111912369728088,
            0.80478835105896,
            0.8701068460941315,
            0.9712996482849121,
        ),
        "confidence_score": (
            0.1071063974154037,
            0.2505752651840195,
            0.2965757196787984,
            0.35603736631561206,
            0.43408799105104373,
            0.5214252811748283,
            0.6104664943617232,
            0.6925474739319109,
            0.7738532142342129,
            0.8453791995008948,
            0.9724877374956733,
        ),
        "average_interface_pae": (
            -31.441935483870967,
            -26.436170767306074,
            -23.590740740740724,
            -20.673112507637256,
            -17.179670329670337,
            -13.385132684230623,
            -9.28230088495575,
            -6.284635416666665,
            -4.082692307692309,
            -2.5245403190075324,
            -1.0173913043478262,
        ),
        "pDockQ/mpDockQ": (
            0.018258893561979524,
            0.06277007963683032,
            0.10881854328425011,
            0.16480232570866313,
            0.2354248449091356,
            0.3127768906978083,
            0.39850803135205826,
            0.4877464419834684,
            0.574595566914355,
            0.6480062549920766,
            0.7402866068322936,
        ),
        "interface_sc": (
            -0.20150967675772263,
            0.185749901432075,
            0.23969240414099632,
            0.2856536354496385,
            0.33372359322334405,
            0.38964615355204946,
            0.4441354327058579,
            0.4932446108607873,
            0.5367910679850174,
            0.583726581653264,
            0.7554651752879871,
        ),
        "interface_hb": (
            0.0,
            3.0,
            6.0,
            8.0,
            10.0,
            13.0,
            15.0,
            19.0,
            24.0,
            32.0,
            219.0,
        ),
        "interface_solv_en": (
            -19.437019114569807,
            1.9811161614067316,
            5.7340100751852106,
            8.747040846269769,
            11.632867247047898,
            15.561733300911527,
            19.690202825755705,
            24.735973338818354,
            31.948630682685803,
            43.33489351789842,
            361.3415837314151,
        ),
        "interface_contact_prob_top10_mean": (
            0.011296,
            0.16358650000000002,
            0.365848,
            0.608114,
            0.78215,
            0.886109,
            0.945688,
            0.9750965,
            0.989495,
            0.9966170000000001,
            0.999966,
        ),
        # n=6034: positive interface-present rows with a parsed CCC value
        "interface_ccc": (
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            25.0,
            42.0,
            60.0,
            83.0,
            120.0,
            688.0,
        ),
    },
    # full non-downsampled v3, POSITIVES only, af3 (n=6127)
    "af3": {
        "interface_LIS": (
            0.0,
            0.0,
            0.0,
            0.049114069371161855,
            0.15646202258105246,
            0.24196739649043122,
            0.31185213076582613,
            0.37733956871516555,
            0.45224274405522175,
            0.5376073370013275,
            0.7382252435615914,
        ),
        "interface_ipSAE": (
            0.0,
            0.0,
            0.0,
            0.011817622985615358,
            0.09207364968424916,
            0.31454670298597853,
            0.48670166965609724,
            0.6039841210988671,
            0.6872034959540964,
            0.7720316261669447,
            0.9471774759371085,
        ),
        "interface_pDockQ2": (
            0.00806745494045585,
            0.009312217771674921,
            0.010169702951052002,
            0.012343243638441548,
            0.01898230478175482,
            0.03490995707564033,
            0.07337637114469373,
            0.14642103377841104,
            0.28227388212231264,
            0.5136692998734235,
            0.9349438164290506,
        ),
        "iptm": (
            0.06,
            0.17,
            0.22,
            0.29,
            0.37,
            0.46,
            0.56,
            0.65,
            0.74,
            0.83,
            0.96,
        ),
        "confidence_score": (
            -99.76,
            0.32,
            0.4,
            0.47,
            0.55,
            0.63,
            0.71,
            0.78,
            0.84,
            0.91,
            1.16,
        ),
        "average_interface_pae": (
            -31.55882352941177,
            -27.29042582417583,
            -24.431281296023567,
            -21.23466230001073,
            -17.598288416075647,
            -13.688535031847127,
            -10.35936363636364,
            -7.4944622093023225,
            -5.369848015916876,
            -3.4195044955044955,
            -1.0749999999999997,
        ),
        "pDockQ/mpDockQ": (
            0.018258893561979524,
            0.10646776734305421,
            0.17925487086255473,
            0.25309125025507356,
            0.3197250826717975,
            0.38613165005859335,
            0.452697317414381,
            0.5167125519126162,
            0.5805716401291584,
            0.6431658840865384,
            0.7395871148386612,
        ),
        "interface_sc": (
            0.0,
            0.4132603282043353,
            0.4441885025726823,
            0.46915794447991194,
            0.4899097726388194,
            0.5142648409033658,
            0.5393892153521872,
            0.5639078767470249,
            0.594452141877825,
            0.6302878187891571,
            0.7616072028183929,
        ),
        "interface_hb": (
            0.0,
            4.0,
            6.0,
            9.0,
            11.0,
            13.0,
            16.0,
            20.0,
            26.0,
            36.0,
            201.0,
        ),
        "interface_solv_en": (
            -27.256826262904255,
            1.9196611270768926,
            5.580423437772015,
            8.746521905894863,
            11.776864127807466,
            15.773898668424408,
            20.561330219245985,
            26.263961502937395,
            34.64776936849817,
            48.403697136379016,
            404.59852857526937,
        ),
        "interface_contact_prob_top10_mean": (
            0.0,
            0.1236,
            0.25020000000000003,
            0.45999999999999996,
            0.671,
            0.79,
            0.8710000000000001,
            0.9279999999999999,
            0.968,
            0.99,
            1.0,
        ),
        # n=6126: positive interface-present rows with a parsed CCC value
        "interface_ccc": (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            6.0,
            33.0,
            52.0,
            73.0,
            110.0,
            781.0,
        ),
    },
}


def _safe_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed if math.isfinite(parsed) else float("nan")


def calibrated_feature_percentile(
    feature: str, value: Any, backend: str | None = None
) -> float | None:
    """Map a raw feature value onto the frozen benchmark percentile scale.

    ``backend`` selects a per-backend ladder ("af2"/"af3"). AlphaFold2 and
    AlphaFold3 put the same feature on measurably different scales -- shape
    complementarity differs by 0.69 of its IQR between them and the overall
    confidence score by 0.27 -- so a pooled ladder systematically flatters one
    backend. Pass the backend when it is known; omitting it keeps the pooled
    scale, which is what callers that cannot tell the backend apart should use.
    """
    if feature not in BENCHMARK_QUANTILES:
        raise KeyError(f"unknown metascore feature: {feature}")

    raw = _safe_float(value)
    if math.isnan(raw):
        return None

    oriented = raw * FEATURE_DIRECTIONS[feature]
    quantiles = BENCHMARK_QUANTILES_BY_BACKEND.get(backend, {}).get(
        feature, BENCHMARK_QUANTILES[feature]
    )
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


def infer_backend(row: Mapping[str, Any]) -> str | None:
    """Best-effort backend tag for a scored row ("af2"/"af3", else None)."""
    src = str(row.get("interface_contact_prob_source") or "")
    if src.startswith("af2"):
        return "af2"
    if src.startswith("af3"):
        return "af3"
    model = str(row.get("model_used") or "")
    if model.startswith("seed-"):
        return "af3"
    if "multimer" in model or model.startswith("model_"):
        return "af2"
    return None


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
        if (
            percentile := calibrated_feature_percentile(
                feature, row.get(feature), infer_backend(row)
            )
        )
        is not None
    ]
    if not percentiles:
        return float("nan")
    return float(sum(percentiles) / len(percentiles))
