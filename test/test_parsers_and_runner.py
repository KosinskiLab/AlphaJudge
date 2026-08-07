from __future__ import annotations

import csv
import gzip
import json
import logging
import math
import pickle
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from alphajudge.parsers import pick_parser
from alphajudge.parsers.af2 import AF2Parser
from alphajudge.parsers.af3 import AF3Parser
from alphajudge.contact_probs import contact_probs_from_distogram
from alphajudge.runner import process, process_many


# -------------------------
# Expected output schema (based on real AlphaJudge output CSV)
# -------------------------

EXPECTED_OUTPUT_COLUMNS = {
    "jobs",
    "model_used",
    "interface",
    "iptm_ptm",
    "iptm",
    "ptm",
    "confidence_score",
    "pDockQ/mpDockQ",
    "average_interface_pae",
    "interface_average_plddt",
    "interface_num_intf_residues",
    "interface_polar",
    "interface_hydrophobic",
    "interface_charged",
    "interface_contact_pairs",
    "interface_contact_prob_source",
    "interface_contact_prob_max",
    "interface_contact_prob_top10_mean",
    "interface_score",
    "interface_pDockQ2",
    "interface_ipSAE",
    "interface_LIS",
    "interface_cLIS",
    "interface_iLIS",
    "interface_hb",
    "interface_sb",
    "interface_ss",
    "interface_sc",
    "interface_area",
    "interface_solv_en",
}

# columns that are expected to be numeric (but may be NaN)
EXPECTED_NUMERIC_COLUMNS = EXPECTED_OUTPUT_COLUMNS - {
    "jobs",
    "model_used",
    "interface",
    "interface_contact_prob_source",
}


# -------------------------
# Helpers
# -------------------------

def _repo_path(p: str) -> Path:
    return Path(p)


def _ensure_exists(p: Path) -> Path:
    if not p.exists():
        pytest.skip(f"Missing test data: {p}")
    return p


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def to_float_or_nan(x: Any) -> float:
    if x is None:
        return float("nan")
    s = str(x).strip().lower()
    if s in ("", "nan", "none"):
        return float("nan")
    return float(s)


def nearly_equal(a: Any, b: Any, tol: float = 1e-6) -> bool:
    a = to_float_or_nan(a)
    b = to_float_or_nan(b)
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isnan(a) != math.isnan(b):
        return False
    return abs(a - b) <= tol


def assert_has_expected_headers(rows: list[dict[str, str]], *, where: str) -> None:
    assert rows, f"{where} must contain at least one row"
    header = set(rows[0].keys())

    missing = sorted(EXPECTED_OUTPUT_COLUMNS - header)
    assert not missing, f"{where} missing expected columns: {missing}"

    # Allow extra columns (forward-compatible), but ensure no empty header keys
    assert "" not in header, f"{where} contains an empty column name"


def assert_numeric_columns_parse(rows: list[dict[str, str]], *, where: str) -> None:
    # verify numeric columns parse as float or NaN for every row
    for i, r in enumerate(rows):
        for col in EXPECTED_NUMERIC_COLUMNS:
            try:
                _ = to_float_or_nan(r.get(col))
            except Exception as e:
                raise AssertionError(f"{where}: row {i} col {col} not parseable: {r.get(col)!r}") from e


def copy_run_dir(src: Path, dst_root: Path) -> Path:
    dst = dst_root / src.name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


def make_official_af3_layout(src: Path, dst_root: Path, job_name: str = "hello_fold") -> Path:
    dst = dst_root / job_name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    root_renames = {
        "ranking_scores.csv": f"{job_name}_ranking_scores.csv",
        "ranked_0_model.cif": f"{job_name}_model.cif",
        "ranked_0_confidences.json": f"{job_name}_confidences.json",
        "ranked_0_summary_confidences.json": f"{job_name}_summary_confidences.json",
    }
    for old_name, new_name in root_renames.items():
        old = dst / old_name
        if old.exists():
            old.rename(dst / new_name)

    for model_dir in sorted(p for p in dst.glob("seed-*_sample-*") if p.is_dir()):
        sample_renames = {
            "model.cif": f"{job_name}_{model_dir.name}_model.cif",
            "confidences.json": f"{job_name}_{model_dir.name}_confidences.json",
            "summary_confidences.json": f"{job_name}_{model_dir.name}_summary_confidences.json",
        }
        for old_name, new_name in sample_renames.items():
            old = model_dir / old_name
            if old.exists():
                old.rename(model_dir / new_name)

    return dst


def _expected_models_for(run, models_to_analyse: str) -> list[str]:
    return [run.order[0]] if models_to_analyse == "best" else list(run.order)


def _assert_pae_pngs_exist(run_dir: Path, model_names: list[str]) -> None:
    for m in model_names:
        png = run_dir / f"pae_{m}.png"
        assert png.exists() and png.stat().st_size > 0, f"Missing/empty PAE png: {png}"


def _get_rows_by_model(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        m = str(r.get("model_used", "")).strip()
        out.setdefault(m, []).append(r)
    return out


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _load_npz_array(path: Path, preferred_key: str) -> np.ndarray:
    with np.load(path) as payload:
        key = preferred_key if preferred_key in payload else payload.files[0]
        return np.array(payload[key], dtype=float)


def _af3_expected_rank(summary: dict[str, Any]) -> Any:
    # AF3: ranking_score is the ranking metric; some files may expose iptm+ptm too
    if summary.get("ranking_score") is not None:
        return summary["ranking_score"]
    if summary.get("iptm+ptm") is not None:
        return summary["iptm+ptm"]
    return None


def _af3_expected_iptm_ptm(summary: dict[str, Any]) -> Any:
    # AF3: iptm_ptm may exist; otherwise iptm+ptm is often the "iptm_ptm-like" metric
    if summary.get("iptm_ptm") is not None:
        return summary["iptm_ptm"]
    if summary.get("iptm+ptm") is not None:
        return summary["iptm+ptm"]
    return None


def _cli_supports(exe: str, flag: str) -> bool:
    try:
        r = subprocess.run([exe, "--help"], capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return False
    txt = (r.stdout or "") + (r.stderr or "")
    return flag in txt


def _run_cli(exe: str, args: list[str]) -> None:
    r = subprocess.run([exe, *args], capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise AssertionError(
            "CLI failed\n"
            f"cmd: {exe} {' '.join(args)}\n"
            f"returncode: {r.returncode}\n"
            f"stdout:\n{r.stdout}\n"
            f"stderr:\n{r.stderr}\n"
        )


# -------------------------
# Fixtures (source dirs)
# -------------------------

@pytest.fixture(scope="module")
def af2_dir_src() -> Path:
    return _ensure_exists(_repo_path("test_data/af2/pos_dimers/Q13148+Q92900"))


@pytest.fixture(scope="module")
def af3_dir_src() -> Path:
    return _ensure_exists(_repo_path("test_data/af3/pos_dimers/Q13148+Q92900"))


@pytest.fixture(scope="module")
def boltz2_dir_src() -> Path:
    return _ensure_exists(_repo_path("test_data/boltz2/6OGE_ABC_DSSO_CDI_seed_3"))


@pytest.fixture(scope="module")
def af2_pos_sample_src() -> list[Path]:
    srcs = [
        _repo_path("test_data/af2/pos_dimers/Q13148+Q92900"),
        _repo_path("test_data/af2/pos_dimers/Q9BUL8+Q13033"),
    ]
    return [_ensure_exists(p) for p in srcs]


@pytest.fixture(scope="module")
def af2_neg_sample_src() -> list[Path]:
    srcs = [_repo_path("test_data/af2/neg_dimers/Q14974+Q13033")]
    return [_ensure_exists(p) for p in srcs]


@pytest.fixture(scope="module")
def af3_pos_sample_src() -> list[Path]:
    srcs = [
        _repo_path("test_data/af3/pos_dimers/Q13148+Q92900"),
        _repo_path("test_data/af3/pos_dimers/Q9BUL8+Q13033"),
    ]
    return [_ensure_exists(p) for p in srcs]


@pytest.fixture(scope="module")
def af3_neg_sample_src() -> list[Path]:
    srcs = [_repo_path("test_data/af3/neg_dimers/Q14974+Q13033")]
    return [_ensure_exists(p) for p in srcs]


# -------------------------
# AF2 runner: best/all + score checks (monomer vs multimer)
# -------------------------

@pytest.mark.parametrize("models_to_analyse", ["best", "all"])
def test_af2_runner_outputs_have_expected_scores(tmp_path: Path, af2_dir_src: Path, models_to_analyse: str):
    af2_dir = copy_run_dir(af2_dir_src, tmp_path)

    parser = pick_parser(af2_dir)
    assert parser.name == "af2"

    process(str(af2_dir), 8.0, 100.0, models_to_analyse, 10.0)

    run = parser.parse_run(af2_dir)
    expected_models = _expected_models_for(run, models_to_analyse)

    out = af2_dir / "interfaces.csv"
    assert out.exists() and out.stat().st_size > 0

    rows = read_csv_rows(out)
    assert_has_expected_headers(rows, where=str(out))
    assert_numeric_columns_parse(rows, where=str(out))

    by_model = _get_rows_by_model(rows)
    assert set(by_model.keys()) == set(expected_models), f"Expected models {expected_models}, got {sorted(by_model.keys())}"

    _assert_pae_pngs_exist(af2_dir, expected_models)

    ranking_path = af2_dir / "ranking_debug.json"
    assert ranking_path.exists(), f"Missing {ranking_path}"
    ranking = _load_json(ranking_path)

    # AF2 variability: monomers may have only ptm; multimers have iptm and iptm+ptm.
    iptm_map = ranking.get("iptm", {}) or {}
    iptm_ptm_map = ranking.get("iptm+ptm", {}) or {}
    ptm_map = ranking.get("ptm", {}) or {}

    for m in expected_models:
        r0 = by_model[m][0]

        exp_ptm = ptm_map.get(m)
        if exp_ptm is not None:
            assert nearly_equal(r0["ptm"], float(exp_ptm)), f"AF2 ptm mismatch for {m}"

        if m in iptm_map and iptm_map[m] is not None:
            # AF2 multimer-like: confidence == iptm+ptm, and iptm_ptm should match that
            assert m in iptm_ptm_map, f"ranking_debug.json missing iptm+ptm for model {m}"
            exp_iptm = float(iptm_map[m])
            exp_conf = float(iptm_ptm_map[m])

            assert nearly_equal(r0["iptm"], exp_iptm), f"AF2 iptm mismatch for {m}"
            assert nearly_equal(r0["confidence_score"], exp_conf), f"AF2 confidence_score mismatch for {m}"
            assert nearly_equal(r0["iptm_ptm"], exp_conf), f"AF2 iptm_ptm mismatch for {m}"
        else:
            # AF2 monomer-like: ranking score is ptm; iptm is undefined (NaN)
            assert math.isnan(to_float_or_nan(r0["iptm"])), f"AF2 monomer expected NaN iptm for {m}"
            if exp_ptm is not None:
                assert nearly_equal(r0["confidence_score"], float(exp_ptm)), f"AF2 monomer confidence_score should equal ptm for {m}"
                # iptm_ptm may be NaN or ptm depending on writer; accept both
                got_iptm_ptm = to_float_or_nan(r0["iptm_ptm"])
                if not math.isnan(got_iptm_ptm):
                    assert nearly_equal(got_iptm_ptm, float(exp_ptm)), f"AF2 monomer iptm_ptm should be NaN or ptm for {m}"


class _FakeResidue:
    """Minimal hashable Bio.PDB-like residue with .id and .get_parent().id."""

    def __init__(self, chain_id: str, res_id: Any):
        self.id = res_id
        self._parent = type("_Chain", (), {"id": chain_id})()

    def get_parent(self):
        return self._parent


def _make_residue(chain_id: str, res_id: Any) -> _FakeResidue:
    return _FakeResidue(chain_id, res_id)


def test_clis_ilis_math_is_deterministic():
    """
    cLIS restricts the LIS PAE transform to contacting residue pairs and iLIS is
    the geometric mean sqrt(LIS * cLIS) (AFM-LIS, Kim et al.). Build a tiny 2+2
    complex with a known PAE matrix so the arithmetic is checkable by hand.
    """
    from alphajudge.interface import Interface

    a1, a2 = _make_residue("A", (" ", 1, " ")), _make_residue("A", (" ", 2, " "))
    b1, b2 = _make_residue("B", (" ", 1, " ")), _make_residue("B", (" ", 2, " "))

    # PAE indices: A1->0, A2->1, B1->2, B2->3. Intra-chain PAE is irrelevant.
    pae = np.array(
        [
            [0.0, 0.0, 0.0, 6.0],
            [0.0, 0.0, 24.0, 24.0],
            [0.0, 24.0, 0.0, 0.0],
            [6.0, 24.0, 0.0, 0.0],
        ]
    )

    iface = object.__new__(Interface)  # bypass __init__; set only what we exercise
    iface._pae = pae
    iface._idx1 = np.array([0, 1])
    iface._idx2 = np.array([2, 3])
    iface._rim = {("A", a1.id): 0, ("A", a2.id): 1, ("B", b1.id): 2, ("B", b2.id): 3}
    iface._pairs = {(a1, b1)}  # only A1-B1 is in physical contact

    # LIS: A->B valid entries (12-pae)/12 = [1.0, 0.5]; B->A = [1.0, 0.5]; mean 0.75.
    assert nearly_equal(iface.lis(), 0.75)
    # cLIS: only the A1-B1 contact, pae 0 both directions -> 1.0.
    assert nearly_equal(iface.clis(), 1.0)
    # iLIS = sqrt(0.75 * 1.0).
    assert nearly_equal(iface.ilis(), math.sqrt(0.75))

    # No contacts -> cLIS 0 -> iLIS 0 regardless of LIS.
    iface._pairs = set()
    assert iface.clis() == 0.0
    assert iface.ilis() == 0.0


def test_contact_probability_scores_math_is_deterministic():
    """
    Contact-probability summaries are computed over all residue pairs between
    the two chains: max and mean of the ten largest values.
    """
    from alphajudge.interface import Interface

    iface = object.__new__(Interface)
    iface._idx1 = np.array([0, 1])
    iface._idx2 = np.array([2, 3])
    iface._contact_prob = np.array(
        [
            [0.0, 0.0, 0.8, 0.4],
            [0.0, 0.0, 0.2, 0.1],
            [0.8, 0.2, 0.0, 0.0],
            [0.4, 0.1, 0.0, 0.0],
        ]
    )

    assert nearly_equal(iface.contact_prob_max, 0.8)
    assert nearly_equal(iface.contact_prob_top10_mean, 0.375)

    missing = object.__new__(Interface)
    missing._idx1 = np.array([0, 1])
    missing._idx2 = np.array([2, 3])
    missing._contact_prob = None
    assert math.isnan(missing.contact_prob_max)
    assert math.isnan(missing.contact_prob_top10_mean)


def test_af2_distogram_contact_probs_softmax_cutoff_is_deterministic(tmp_path: Path):
    """
    AF2 contact probabilities are distogram softmax mass for bins lying entirely
    below the contact cutoff, i.e. bins whose UPPER bound is below
    ``AF2_DISTOGRAM_CONTACT_CUTOFF`` (8 A).

    This replaces the earlier convention (mass in bins whose LOWER bound was
    below 12 A, after Humphreys et al., which included the bin straddling the
    cutoff). Both the threshold and the bin rule changed in 1.3.0: sweeping the
    contact threshold from 4 to 20 A on the four-organism benchmark puts peak
    positive-vs-negative discrimination at 6-8 A for both AlphaFold versions,
    with the old 12 A setting scoring about 0.005 AUROC lower on AlphaFold2.
    """
    probs = np.array(
        [
            [[0.70, 0.20, 0.05, 0.05], [0.10, 0.20, 0.30, 0.40]],
            [[0.20, 0.30, 0.10, 0.40], [0.05, 0.05, 0.20, 0.70]],
        ],
        dtype=float,
    )
    logits = np.log(probs)
    bin_edges = np.array([4.0, 8.0, 12.0])

    # Bin upper bounds are [4, 8, 12, inf]; only the first lies below 8 A.
    direct = contact_probs_from_distogram(logits, bin_edges)
    expected_asym = np.array([[0.70, 0.10], [0.20, 0.05]])
    assert np.allclose(direct, expected_asym)

    # Standard AF2/AF3 breaks: 19 of the 64 bins end below 8 A.
    standard_edges = np.linspace(2.3125, 21.6875, 63)
    standard_logits = np.zeros((1, 1, 64), dtype=float)
    assert np.allclose(
        contact_probs_from_distogram(standard_logits, standard_edges),
        np.array([[19 / 64]]),
    )
    # Boundary: bin 18 is the last one ending below 8 A and counts in full,
    # while bin 19 straddles the cutoff and is excluded entirely.
    inside_logits = np.full((1, 1, 64), -1000.0, dtype=float)
    inside_logits[..., 18] = 0.0
    assert np.allclose(
        contact_probs_from_distogram(inside_logits, standard_edges),
        np.array([[1.0]]),
    )
    straddling_logits = np.full((1, 1, 64), -1000.0, dtype=float)
    straddling_logits[..., 19] = 0.0
    assert np.allclose(
        contact_probs_from_distogram(straddling_logits, standard_edges),
        np.array([[0.0]]),
        atol=1e-6,
    )

    run_dir = tmp_path / "af2_result"
    run_dir.mkdir()
    with gzip.open(run_dir / "result_model_1.pkl.gz", "wb") as f:
        pickle.dump({"distogram": {"logits": logits, "bin_edges": bin_edges}}, f)

    parsed = AF2Parser._load_contact_probs_from_result_pkl(
        run_dir, "model_1", expected_shape=(2, 2)
    )
    assert parsed is not None
    assert np.allclose(parsed, expected_asym)


def test_af2_distogram_loader_uses_requested_model_not_last_glob(tmp_path: Path):
    run_dir = tmp_path / "af2_result"
    run_dir.mkdir()
    bin_edges = np.array([4.0, 8.0, 12.0])

    def write_distogram(model: str, contact_prob: float) -> None:
        logits = np.full((1, 1, 4), -1000.0, dtype=float)
        logits[..., 0] = np.log(contact_prob)
        logits[..., 3] = np.log(1.0 - contact_prob)
        with (run_dir / f"result_{model}.pkl").open("wb") as f:
            pickle.dump(
                {"distogram": {"logits": logits, "bin_edges": bin_edges}},
                f,
            )

    write_distogram("model_1", 0.25)
    write_distogram("model_2", 0.75)

    parsed = AF2Parser._load_contact_probs_from_result_pkl(
        run_dir, "model_1", expected_shape=(1, 1)
    )

    assert parsed is not None
    assert np.allclose(parsed, np.array([[0.25]]))


def test_af2_missing_distogram_warns_once_and_returns_none(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    run_dir = tmp_path / "af2_result"
    run_dir.mkdir()
    with (run_dir / "result_model_1.pkl").open("wb") as f:
        pickle.dump({"plddt": [90.0]}, f)
    with (run_dir / "result_model_2.pkl").open("wb") as f:
        pickle.dump({"plddt": [80.0]}, f)

    old_warned = AF2Parser._warned_missing_distogram
    AF2Parser._warned_missing_distogram = False
    try:
        caplog.set_level(logging.WARNING, logger="alphajudge.parsers.af2")
        parsed = AF2Parser._load_contact_probs_from_result_pkl(
            run_dir, "model_1", expected_shape=(1, 1)
        )
        assert parsed is None
        assert "has no distogram" in caplog.text
        assert "--remove_keys_from_pickles" in caplog.text

        caplog.clear()
        parsed = AF2Parser._load_contact_probs_from_result_pkl(
            run_dir, "model_2", expected_shape=(1, 1)
        )
        assert parsed is None
        assert "has no distogram" not in caplog.text
    finally:
        AF2Parser._warned_missing_distogram = old_warned


# -------------------------
# AF3 runner: best/all + score checks
# -------------------------

def test_af3_job_prefix_from_ranking_file_handles_plain_prefixed_and_weird():
    assert AF3Parser._job_prefix_from_ranking_file(Path("ranking_scores.csv")) is None
    assert (
        AF3Parser._job_prefix_from_ranking_file(Path("hello_fold_ranking_scores.csv"))
        == "hello_fold"
    )
    assert (
        AF3Parser._job_prefix_from_ranking_file(Path("hello_fold_ranking_scores_backup.csv"))
        is None
    )


def test_af3_pae_shape_warns_but_unknown_schema_raises(caplog: pytest.LogCaptureFixture):
    class Chain:
        def __init__(self, chain_id: str):
            self.id = chain_id

    chains = [Chain("A"), Chain("B")]
    cid = {"A": [0], "B": [1]}

    caplog.set_level(logging.WARNING, logger="alphajudge.parsers.af3")
    pae, max_pae = AF3Parser._normalize_pae_af3(
        {"predicted_aligned_error": [[3.0]], "max_predicted_aligned_error": 3.0},
        chains,
        cid,
    )

    assert pae.shape == (2, 2)
    assert np.all(pae == 100.0)
    assert max_pae == 3.0
    assert "predicted_aligned_error shape" in caplog.text

    caplog.clear()
    with pytest.raises(ValueError, match="unknown AF3 confidences schema"):
        AF3Parser._normalize_pae_af3({"unexpected_schema": True}, chains, cid)


def test_af3_parser_accepts_alphapulldown_layout(af3_dir_src: Path):
    assert (af3_dir_src / "ranking_scores.csv").exists()

    parser = pick_parser(af3_dir_src)
    assert parser.name == "af3"

    run = parser.parse_run(af3_dir_src)
    assert len(run.order) == 5
    assert run.order[0] == "seed-19698302_sample-1"

    _, conf = run.load_model(run.order[0])
    assert conf.pae_matrix.shape[0] == len(conf.plddt_residue)
    assert conf.contact_prob_matrix is not None
    assert conf.contact_prob_matrix.shape == conf.pae_matrix.shape
    assert conf.contact_prob_source == "af3_contact_probs"
    assert np.isfinite(conf.confidence_score)


def test_af3_contact_probability_scores_match_raw_contact_probs(
    tmp_path: Path, af3_dir_src: Path
):
    from alphajudge.complex import Complex

    parser = pick_parser(af3_dir_src)
    run = parser.parse_run(af3_dir_src)
    best_model = run.order[0]
    structure, conf = run.load_model(best_model)

    raw = _load_json(af3_dir_src / best_model / "confidences.json")
    raw_contact_probs = np.asarray(raw["contact_probs"], dtype=float)
    raw_sym = 0.5 * (raw_contact_probs + raw_contact_probs.T)

    assert conf.contact_prob_matrix is not None
    assert np.allclose(conf.contact_prob_matrix, raw_sym)

    comp = Complex(structure, conf, 8.0, 100.0, 10.0)
    assert comp.interfaces
    iface = comp.interfaces[0]
    label = f"{iface.chain1[0].get_parent().id}_{iface.chain2[0].get_parent().id}"

    af3_dir = copy_run_dir(af3_dir_src, tmp_path)
    process(
        str(af3_dir),
        8.0,
        100.0,
        "best",
        10.0,
        per_run_csv_name="interfaces_contact_probs.csv",
        skip_pae_png=True,
        skip_biophysical_scores=True,
    )

    rows = read_csv_rows(af3_dir / "interfaces_contact_probs.csv")
    row = next(r for r in rows if r["interface"] == label)
    assert row["interface_contact_prob_source"] == "af3_contact_probs"
    assert nearly_equal(row["interface_contact_prob_max"], iface.contact_prob_max)
    assert nearly_equal(row["interface_contact_prob_top10_mean"], iface.contact_prob_top10_mean)


def test_af3_contact_probs_alignment_uses_token_res_ids_with_extra_tokens():
    from Bio.PDB.Atom import Atom
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Residue import Residue

    def residue(chain: Chain, resseq: int, serial: int) -> None:
        res = Residue((" ", resseq, " "), "ALA", "")
        res.add(Atom("CA", np.zeros(3), 1.0, 1.0, " ", "CA", serial, element="C"))
        chain.add(res)

    chain_a = Chain("A")
    chain_b = Chain("B")
    residue(chain_a, 1, 1)
    residue(chain_a, 2, 2)
    residue(chain_b, 1, 3)

    token_matrix = np.arange(16, dtype=float).reshape(4, 4)
    aligned = AF3Parser._align_token_pair_matrix_to_residues(
        token_matrix,
        ["A", "A", "A", "B"],
        [1, 99, 2, 1],
        [chain_a, chain_b],
        {"A": [0, 1], "B": [2]},
        (3, 3),
    )

    assert aligned is not None
    expected = token_matrix[np.ix_([0, 2, 3], [0, 2, 3])]
    assert np.allclose(aligned, expected)


@pytest.mark.parametrize("models_to_analyse", ["best", "all"])
def test_af3_runner_outputs_have_expected_scores(tmp_path: Path, af3_dir_src: Path, models_to_analyse: str):
    af3_dir = copy_run_dir(af3_dir_src, tmp_path)

    parser = pick_parser(af3_dir)
    assert parser.name == "af3"

    process(str(af3_dir), 8.0, 100.0, models_to_analyse, 10.0)

    run = parser.parse_run(af3_dir)
    expected_models = _expected_models_for(run, models_to_analyse)

    out = af3_dir / "interfaces.csv"
    assert out.exists() and out.stat().st_size > 0

    rows = read_csv_rows(out)
    assert_has_expected_headers(rows, where=str(out))
    assert_numeric_columns_parse(rows, where=str(out))

    header = set(rows[0].keys())
    has_ranking_score = "ranking_score" in header

    by_model = _get_rows_by_model(rows)
    assert set(by_model.keys()) == set(expected_models), f"Expected models {expected_models}, got {sorted(by_model.keys())}"

    _assert_pae_pngs_exist(af3_dir, expected_models)

    ranked0_path = af3_dir / "ranked_0_summary_confidences.json"
    ranked0 = _load_json(ranked0_path) if ranked0_path.exists() else None

    for m in expected_models:
        model_dir = af3_dir / m
        summary_path = model_dir / "summary_confidences.json"
        summary = None
        if summary_path.exists():
            summary = _load_json(summary_path)
        elif ranked0 is not None and m == expected_models[0]:
            summary = ranked0

        if summary is None:
            pytest.skip(f"Missing AF3 summary for model {m} (no {summary_path} and no ranked_0 fallback)")

        r0 = by_model[m][0]
        got_iptm = to_float_or_nan(r0.get("iptm"))
        got_ptm = to_float_or_nan(r0.get("ptm"))
        got_conf = to_float_or_nan(r0.get("confidence_score"))
        got_iptm_ptm = to_float_or_nan(r0.get("iptm_ptm"))
        got_rankcol = to_float_or_nan(r0.get("ranking_score")) if has_ranking_score else float("nan")

        exp_iptm = summary.get("iptm")
        exp_ptm = summary.get("ptm")
        exp_rank = _af3_expected_rank(summary)
        exp_iptm_ptm = _af3_expected_iptm_ptm(summary)

        if exp_iptm is not None:
            assert nearly_equal(got_iptm, float(exp_iptm)), f"AF3 iptm mismatch for {m}"
        else:
            assert math.isnan(got_iptm), f"Expected NaN iptm for {m}, got {got_iptm}"

        assert exp_ptm is not None, f"AF3 summary missing ptm for {m}"
        assert nearly_equal(got_ptm, float(exp_ptm)), f"AF3 ptm mismatch for {m}"

        # AF3: confidence_score follows ranking_score (NOT iptm_ptm)
        if exp_rank is not None:
            assert nearly_equal(got_conf, float(exp_rank)), f"AF3 confidence_score mismatch for {m}"
            if has_ranking_score:
                assert nearly_equal(got_rankcol, float(exp_rank)), f"AF3 ranking_score mismatch for {m}"

        # AF3: iptm_ptm is its own metric; compare to its corresponding key
        if exp_iptm_ptm is not None:
            assert nearly_equal(got_iptm_ptm, float(exp_iptm_ptm)), f"AF3 iptm_ptm mismatch for {m}"


def test_af3_empty_csv_is_explained_when_no_contacts(
    tmp_path: Path, af3_dir_src: Path, caplog: pytest.LogCaptureFixture
):
    """Regression for issue #17: when no inter-chain contact is within contact_thresh,
    the CSV is empty (no header) but the log must say *why* instead of being silent."""
    af3_dir = copy_run_dir(af3_dir_src, tmp_path)

    caplog.set_level(logging.WARNING, logger="alphajudge.runner")
    # A sub-Angstrom contact threshold guarantees no inter-chain contacts -> no interfaces.
    process(str(af3_dir), 0.01, 100.0, "best", 10.0)

    out = af3_dir / "interfaces.csv"
    assert out.exists(), "an (empty) interfaces.csv should still be written"
    assert out.stat().st_size == 0, "no contacts -> empty CSV (no header)"

    assert "no interface rows" in caplog.text
    assert "contact_thresh" in caplog.text


def test_af3_parser_accepts_official_prefixed_layout(tmp_path: Path, af3_dir_src: Path):
    af3_dir = make_official_af3_layout(af3_dir_src, tmp_path, job_name="hello_fold")

    assert not (af3_dir / "ranking_scores.csv").exists()
    parser = pick_parser(af3_dir)
    assert parser.name == "af3"

    run = parser.parse_run(af3_dir)
    assert run.order

    process(
        str(af3_dir),
        8.0,
        100.0,
        "all",
        10.0,
        per_run_csv_name="interfaces_official_af3.csv",
        skip_pae_png=True,
        skip_biophysical_scores=True,
    )

    rows = read_csv_rows(af3_dir / "interfaces_official_af3.csv")
    assert rows
    assert set(_get_rows_by_model(rows)) == set(run.order)


def test_boltz2_parser_processes_ranked_prediction_dir(tmp_path: Path, boltz2_dir_src: Path):
    boltz_dir = copy_run_dir(boltz2_dir_src, tmp_path)
    model_name = "6OGE_ABC_DSSO_CDI_Boltz2_model_0"

    raw_pae = _load_npz_array(boltz_dir / f"pae_{model_name}.npz", "pae")
    raw_plddt = _load_npz_array(boltz_dir / f"plddt_{model_name}.npz", "plddt")
    assert raw_pae.shape == (1070, 1070)
    assert raw_plddt.shape == (1070,)

    parser = pick_parser(boltz_dir)
    assert parser.name == "boltz2"
    run = parser.parse_run(boltz_dir)
    assert run.order == [model_name]

    _, conf = run.load_model(run.order[0])
    assert conf.pae_matrix.shape == (1058, 1058)
    assert len(conf.plddt_residue) == 1058
    assert conf.pae_matrix.shape[0] < raw_pae.shape[0]
    assert len(conf.plddt_residue) < raw_plddt.size
    assert np.allclose(conf.pae_matrix, raw_pae[:1058, :1058])
    assert np.allclose(conf.plddt_residue, raw_plddt[:1058])
    assert nearly_equal(conf.confidence_score, 0.8897088170051575)

    process(
        str(boltz_dir),
        8.0,
        100.0,
        "best",
        10.0,
        per_run_csv_name="interfaces_boltz2.csv",
        skip_pae_png=True,
        skip_biophysical_scores=True,
    )

    rows = read_csv_rows(boltz_dir / "interfaces_boltz2.csv")
    assert rows
    assert set(_get_rows_by_model(rows)) == {model_name}


# -------------------------
# Headers sanity across AF2/AF3
# -------------------------

def test_headers_include_expected_schema(tmp_path: Path, af2_dir_src: Path, af3_dir_src: Path):
    af2_dir = copy_run_dir(af2_dir_src, tmp_path / "af2")
    af3_dir = copy_run_dir(af3_dir_src, tmp_path / "af3")

    process(str(af2_dir), 8.0, 100.0, "best", 10.0)
    process(str(af3_dir), 8.0, 100.0, "best", 10.0)

    r2 = read_csv_rows(af2_dir / "interfaces.csv")
    r3 = read_csv_rows(af3_dir / "interfaces.csv")
    assert_has_expected_headers(r2, where=str(af2_dir / "interfaces.csv"))
    assert_has_expected_headers(r3, where=str(af3_dir / "interfaces.csv"))


# -------------------------
# process_many
# -------------------------

def test_process_many_aggregates_rows(
    tmp_path: Path,
    af2_pos_sample_src: list[Path],
    af2_neg_sample_src: list[Path],
    af3_pos_sample_src: list[Path],
    af3_neg_sample_src: list[Path],
):
    root = tmp_path / "data"
    af2_pos = [copy_run_dir(p, root / "af2_pos") for p in af2_pos_sample_src]
    af2_neg = [copy_run_dir(p, root / "af2_neg") for p in af2_neg_sample_src]
    af3_pos = [copy_run_dir(p, root / "af3_pos") for p in af3_pos_sample_src]
    af3_neg = [copy_run_dir(p, root / "af3_neg") for p in af3_neg_sample_src]

    summary = tmp_path / "summary.csv"
    paths = [*(str(p) for p in af2_pos), *(str(p) for p in af2_neg), *(str(p) for p in af3_pos), *(str(p) for p in af3_neg)]

    got = process_many(
        paths,
        contact_thresh=8.0,
        pae_filter=100.0,
        models_to_analyse="best",
        recursive=False,
        summary_csv=str(summary),
    )
    assert got is not None and summary.exists() and summary.stat().st_size > 0

    rows = read_csv_rows(summary)
    assert_has_expected_headers(rows, where=str(summary))
    assert_numeric_columns_parse(rows, where=str(summary))

    expected_rows = 0
    for p in af2_pos + af2_neg + af3_pos + af3_neg:
        per = p / "interfaces.csv"
        assert per.exists(), f"Missing per-run interfaces.csv at {per}"
        expected_rows += len(read_csv_rows(per))

    assert len(rows) == expected_rows, f"Expected {expected_rows} rows, got {len(rows)}"


def test_process_many_recursive_discovers_runs(tmp_path: Path, af2_pos_sample_src: list[Path], af3_pos_sample_src: list[Path]):
    af2_root = tmp_path / "af2" / "pos_dimers"
    af3_root = tmp_path / "af3" / "pos_dimers"
    af2_root.mkdir(parents=True, exist_ok=True)
    af3_root.mkdir(parents=True, exist_ok=True)

    for src in af2_pos_sample_src:
        shutil.copytree(src, af2_root / src.name)
    for src in af3_pos_sample_src:
        shutil.copytree(src, af3_root / src.name)

    # 1) recursive on single root containing multiple runs
    summary = tmp_path / "recursive_summary.csv"
    got = process_many(
        [str(af2_root.parent)],  # AF2 root only
        contact_thresh=8.0,
        pae_filter=100.0,
        models_to_analyse="best",
        recursive=True,
        summary_csv=str(summary),
    )
    assert got is not None and summary.exists() and summary.stat().st_size > 0
    rows = read_csv_rows(summary)
    assert_has_expected_headers(rows, where=str(summary))

    expected = 0
    for src in af2_pos_sample_src:
        per = (af2_root / src.name) / "interfaces.csv"
        assert per.exists()
        expected += len(read_csv_rows(per))
    assert len(rows) >= expected

    # 2) recursive on mixed AF2+AF3 roots
    summary2 = tmp_path / "recursive_summary2.csv"
    got2 = process_many(
        [str(af2_root.parent), str(af3_root.parent)],
        contact_thresh=8.0,
        pae_filter=100.0,
        models_to_analyse="best",
        recursive=True,
        summary_csv=str(summary2),
    )
    assert got2 is not None and summary2.exists() and summary2.stat().st_size > 0
    rows2 = read_csv_rows(summary2)
    assert_has_expected_headers(rows2, where=str(summary2))

    expected2 = 0
    for src in af2_pos_sample_src:
        expected2 += len(read_csv_rows((af2_root / src.name) / "interfaces.csv"))
    for src in af3_pos_sample_src:
        expected2 += len(read_csv_rows((af3_root / src.name) / "interfaces.csv"))
    assert len(rows2) >= expected2


# -------------------------
# CLI integration: --cores and --recursive edge cases
# -------------------------

def test_cli_cores_two_for_one_directory(tmp_path: Path, af3_dir_src: Path):
    exe = shutil.which("alphajudge")
    if not exe:
        pytest.skip("alphajudge CLI not found in PATH")

    if not _cli_supports(exe, "--cores"):
        pytest.skip("alphajudge CLI does not advertise --cores")

    run_dir = copy_run_dir(af3_dir_src, tmp_path / "run")
    out1 = tmp_path / "out_cores1.csv"
    out2 = tmp_path / "out_cores2.csv"

    base_args = [str(run_dir), "--models_to_analyse", "best", "-o"]

    _run_cli(exe, base_args + [str(out1), "--cores", "1"])
    _run_cli(exe, base_args + [str(out2), "--cores", "2"])  # edge case: 2 cores, 1 dir

    r1 = read_csv_rows(out1)
    r2 = read_csv_rows(out2)
    assert_has_expected_headers(r1, where=str(out1))
    assert_has_expected_headers(r2, where=str(out2))
    assert len(r1) == len(r2), "Changing --cores must not change result row count"


def test_cli_recursive_single_directory_root(tmp_path: Path, af2_dir_src: Path):
    exe = shutil.which("alphajudge")
    if not exe:
        pytest.skip("alphajudge CLI not found in PATH")

    if not _cli_supports(exe, "--recursive"):
        pytest.skip("alphajudge CLI does not advertise --recursive")

    # Root contains exactly one run directory
    root = tmp_path / "root"
    root.mkdir(parents=True, exist_ok=True)
    run_dir = copy_run_dir(af2_dir_src, root)

    out_nonrec = tmp_path / "out_nonrec.csv"
    out_rec = tmp_path / "out_rec.csv"

    _run_cli(exe, [str(run_dir), "--models_to_analyse", "best", "-o", str(out_nonrec)])
    _run_cli(exe, [str(root), "--recursive", "--models_to_analyse", "best", "-o", str(out_rec)])

    r1 = read_csv_rows(out_nonrec)
    r2 = read_csv_rows(out_rec)
    assert_has_expected_headers(r1, where=str(out_nonrec))
    assert_has_expected_headers(r2, where=str(out_rec))
    assert len(r1) == len(r2), "Recursive root with one run should match direct-run output row count"


# -------------------------
# Compressed-confidences reading (AlphaPulldown slim/minimal storage modes)
# -------------------------

import gzip
import lzma

from alphajudge.parsers import BaseParser


def test_read_json_reads_plain_xz_and_gz(tmp_path):
    payload = {"a": 1, "pae": [[0.0, 1.0], [1.0, 0.0]]}
    plain = tmp_path / "confidences.json"
    plain.write_text(json.dumps(payload))
    xz = tmp_path / "x.json.xz"
    with lzma.open(xz, "wt") as fh:
        json.dump(payload, fh)
    gz = tmp_path / "x.json.gz"
    with gzip.open(gz, "wt") as fh:
        json.dump(payload, fh)

    assert BaseParser._read_json(plain) == payload
    assert BaseParser._read_json(xz) == payload
    assert BaseParser._read_json(gz) == payload


def test_read_json_detects_compression_by_magic_not_extension(tmp_path):
    # xz bytes stored under a plain .json name must still decode.
    payload = {"k": "v"}
    mislabeled = tmp_path / "confidences.json"
    with lzma.open(mislabeled, "wt") as fh:
        json.dump(payload, fh)
    assert BaseParser._read_json(mislabeled) == payload


def test_read_json_falls_back_to_compressed_sibling(tmp_path):
    payload = {"pae": [[0.0]]}
    # Only the .xz sibling exists; the plain path is requested.
    with lzma.open(tmp_path / "confidences.json.xz", "wt") as fh:
        json.dump(payload, fh)
    assert BaseParser._read_json(tmp_path / "confidences.json") == payload


def test_read_json_missing_returns_empty(tmp_path):
    assert BaseParser._read_json(tmp_path / "nope.json") == {}


def _write_af3_sample(model_dir: Path, *, compress: bool, with_summary: bool = True):
    model_dir.mkdir(parents=True, exist_ok=True)
    conf = {"pae": [[0.0, 5.0], [5.0, 0.0]], "token_chain_ids": ["A", "B"]}
    if compress:
        with lzma.open(model_dir / "confidences.json.xz", "wt") as fh:
            json.dump(conf, fh)
    else:
        (model_dir / "confidences.json").write_text(json.dumps(conf))
    if with_summary:
        (model_dir / "summary_confidences.json").write_text(json.dumps({"iptm": 0.5, "ptm": 0.5}))


def test_find_af3_json_finds_compressed_confidences(tmp_path):
    md = tmp_path / "seed-1_sample-0"
    _write_af3_sample(md, compress=True)
    found = AF3Parser._find_af3_json(tmp_path, "seed-1_sample-0", "confidences", None, False)
    assert found.name == "confidences.json.xz"
    assert found.exists()


def test_find_af3_json_does_not_shadow_confidences_with_summary(tmp_path):
    # Even when only summary_confidences.json is present, the confidences search
    # must not return it (it carries only coarse per-chain-pair PAE). The caller
    # then falls back to summary explicitly.
    md = tmp_path / "seed-1_sample-0"
    md.mkdir(parents=True)
    (md / "summary_confidences.json").write_text(json.dumps({"iptm": 0.5}))
    found = AF3Parser._find_af3_json(tmp_path, "seed-1_sample-0", "confidences", None, False)
    assert not found.name.startswith("summary_")
    assert not found.exists()  # genuinely no confidences file present


def test_find_af3_json_excludes_job_prefixed_summary(tmp_path):
    # Official AF3 layout uses "<job>_summary_confidences.json", which does NOT
    # start with "summary_". The confidences search must still not return it,
    # and must not let it shadow the real "<job>_<model>_confidences.json".
    job = "hello_fold"
    model = "seed-1_sample-0"
    md = tmp_path / model
    md.mkdir(parents=True)
    (md / f"{job}_{model}_confidences.json").write_text(
        json.dumps({"pae": [[0.0]], "token_chain_ids": ["A"]})
    )
    (md / f"{job}_{model}_summary_confidences.json").write_text(json.dumps({"iptm": 0.5}))
    found = AF3Parser._find_af3_json(tmp_path, model, "confidences", job, False)
    assert found.name == f"{job}_{model}_confidences.json"
    assert "summary" not in found.name


def test_confident_contact_count_math_is_deterministic():
    """
    CCC counts inter-chain contacting residue pairs whose PAE is at or below the
    cutoff. The PAE matrix is asymmetric, so the direction convention decides
    which pairs qualify; the published default scores chain1 -> chain2 only.
    """
    from alphajudge.confident_contacts import ContactGeometry, PaeDirection
    from alphajudge.interface import Interface

    a1, a2 = _make_residue("A", (" ", 1, " ")), _make_residue("A", (" ", 2, " "))
    b1, b2 = _make_residue("B", (" ", 1, " ")), _make_residue("B", (" ", 2, " "))

    # PAE indices A1->0, A2->1, B1->2, B2->3. A1-B1 is confident both ways;
    # A2-B2 is confident only in the A->B direction (2.0 vs 9.0).
    pae = np.array(
        [
            [0.0, 0.0, 1.0, 20.0],
            [0.0, 0.0, 20.0, 2.0],
            [1.0, 20.0, 0.0, 0.0],
            [20.0, 9.0, 0.0, 0.0],
        ]
    )

    iface = object.__new__(Interface)
    iface.chain1, iface.chain2 = [a1, a2], [b1, b2]
    iface._pae = pae
    iface._cid1_id = "A"
    iface._rim = {("A", a1.id): 0, ("A", a2.id): 1, ("B", b1.id): 2, ("B", b2.id): 3}
    iface._pairs = {(a1, b1), (a2, b2)}

    rep = ContactGeometry.REPRESENTATIVE_ATOM
    # A->B: PAE 1.0 and 2.0, both < 4 -> 2 confident contacts.
    assert iface.confident_contacts(geometry=rep) == 2
    # B->A: PAE 1.0 and 9.0 -> only the first qualifies.
    assert iface.confident_contacts(geometry=rep, direction=PaeDirection.BA) == 1
    # Requiring both directions is the stricter max convention.
    assert iface.confident_contacts(geometry=rep, direction=PaeDirection.MAX) == 1
    # A tighter cutoff drops the 2.0 pair even in the A->B direction.
    assert iface.confident_contacts(geometry=rep, pae_cutoff=1.5) == 1
    # The default comparison is strict, matching the authors' code, so a pair
    # sitting exactly on the cutoff does not count.
    assert iface.confident_contacts(geometry=rep, pae_cutoff=2.0) == 1
    assert iface.confident_contacts(geometry=rep, pae_cutoff=2.0, inclusive=True) == 2

    # No contacts -> no confident contacts, whatever the PAE says.
    iface._pairs = set()
    assert iface.confident_contacts(geometry=rep) == 0


def test_interactome3d_contact_rules_are_element_and_distance_specific():
    """
    The Interactome3D-style geometry accepts a residue pair on any one of three
    atom-pair rules (C-C <= 5.0, N-O <= 5.5, Cys S-S <= 2.56) and counts the
    pair once however many atom pairs qualify.
    """
    from Bio.PDB import PDBParser
    from io import StringIO

    from alphajudge.confident_contacts import interactome3d_contact_pairs

    # A: one ALA at the origin. B: one ALA whose CB sits 4.5 A away (C-C rule,
    # inside 5.0) and one GLY whose CA sits 7.0 A away (outside every rule).
    pdb = StringIO(
        "ATOM      1  CB  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  CB  ALA B   1       4.500   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  CA  GLY B   2       7.000   0.000   0.000  1.00  0.00           C\n"
        "END\n"
    )
    model = next(PDBParser(QUIET=True).get_structure("x", pdb).get_models())
    chain_a = list(model["A"])
    chain_b = list(model["B"])

    pairs = interactome3d_contact_pairs(chain_a, chain_b)
    assert len(pairs) == 1
    (res1, res2), = pairs
    # Orientation is always (chain1 residue, chain2 residue).
    assert res1.get_parent().id == "A" and res2.get_parent().id == "B"
    assert res2.id[1] == 1  # the 4.5 A partner, not the 7.0 A one


def test_confident_contacts_boundary_convention_is_selectable():
    """
    The publication's main text says PAE <= 4 while its Methods heading says
    PAE < 4; the authors' released code names the column n_contacts_PAE_lt_4A,
    so strict is the operational definition and the default here. Both are
    reachable and they differ exactly by the pairs sitting on the cutoff.
    """
    from alphajudge.confident_contacts import ContactGeometry
    from alphajudge.interface import Interface

    a1 = _make_residue("A", (" ", 1, " "))
    b1 = _make_residue("B", (" ", 1, " "))
    pae = np.array([[0.0, 4.0], [4.0, 0.0]])  # exactly on the cutoff

    iface = object.__new__(Interface)
    iface.chain1, iface.chain2 = [a1], [b1]
    iface._pae = pae
    iface._cid1_id = "A"
    iface._rim = {("A", a1.id): 0, ("B", b1.id): 1}
    iface._pairs = {(a1, b1)}

    rep = ContactGeometry.REPRESENTATIVE_ATOM
    assert iface.confident_contacts(geometry=rep) == 0                  # strict
    assert iface.confident_contacts(geometry=rep, inclusive=True) == 1  # inclusive
