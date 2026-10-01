"""Regression tests for input provenance, cache validity and failed-model isolation."""
from __future__ import annotations

import gzip
import json
import lzma
import os
import pickle
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from Bio.PDB import Atom, Chain, MMCIFIO, Model, PDBIO, Residue, Structure

from alphajudge import cache, meta_score, report, runner
from alphajudge.complex import Complex
from alphajudge.parsers import BaseParser
from alphajudge.parsers.af2 import AF2Parser
from alphajudge.parsers.af3 import AF3Parser
from alphajudge.parsers.boltz import Boltz2Parser


def write_structure(path: Path, separation: float = 5., plddt: float = 90., chains: str = "AB") -> None:
    structure = Structure.Structure("test")
    model = Model.Model(0)
    structure.add(model)
    serial = 1
    for chain_no, chain_id in enumerate(chains):
        chain = Chain.Chain(chain_id)
        model.add(chain)
        for res_no in (1, 2):
            residue = Residue.Residue((" ", res_no, " "), "ALA", " ")
            chain.add(residue)
            for name, dx in (("CA", 0.), ("CB", .5)):
                residue.add(Atom.Atom(
                    name, np.array([res_no * 3.8 + dx, chain_no * separation, 0.]),
                    plddt, 1., " ", name, serial, element="C",
                ))
                serial += 1
    writer = PDBIO()
    writer.set_structure(structure)
    writer.save(str(path))


@pytest.fixture
def af2_run(tmp_path):
    models = ["model_1_multimer_v3_pred_0", "model_2_multimer_v3_pred_0"]
    ranking = {"order": models, "iptm+ptm": dict.fromkeys(models, .68),
               "iptm": dict.fromkeys(models, .7), "ptm": dict.fromkeys(models, .6)}
    (tmp_path / "ranking_debug.json").write_text(json.dumps(ranking))
    for index, model in enumerate(models):
        write_structure(tmp_path / f"unrelaxed_{model}.pdb", separation=7. if index == 0 else 5.)
        (tmp_path / f"pae_{model}.json").write_text(json.dumps([
            {"predicted_aligned_error": np.full((4, 4), 2.).tolist()}
        ]))
    return tmp_path, models


def score(directory, **changes):
    options = dict(contact_thresh=8., pae_filter=100., models_to_analyse="best",
                   summary_csv="summary.csv", ipsae_pae_cutoff=10., force_recompute=False,
                   per_run_csv_name="interfaces.csv", skip_pae_png=True,
                   skip_biophysical_scores=True)
    options.update(changes)
    return runner._process_one_run(str(directory), **options)[1]


def test_filtered_residues_retain_chain_identity(af2_run):
    directory, models = af2_run
    structure, confidence = AF2Parser().parse_run(directory).load_model(models[0])
    comp = Complex(structure, confidence, 8., 100., 10.)
    first, second = (chain.child_list[0] for chain in comp._chains)
    assert first.id == second.id  # Same residue number, different chains.
    assert first != second
    assert first not in list(comp._chains[1])
    residues = [res for chain in comp._chains for res in chain]
    for a in residues:
        for b in residues:
            assert (a == b) == (a in {b})
    assert len(set(residues)) == 4
    assert first.get_parent().get_parent().get_parent() is not None
    # Filtering must not reparent the input structure.
    assert structure[0]["A"].child_list[0] is not first
    assert structure[0]["A"].get_parent() is structure[0]


@pytest.mark.parametrize("contact_thresh,expected_status", [(8., "default"), (9., "custom")])
def test_csv_records_score_provenance(af2_run, contact_thresh, expected_status):
    directory, _ = af2_run
    row = score(directory, contact_thresh=contact_thresh)[0]
    assert float(row["contact_thresh"]) == contact_thresh
    assert float(row["pae_filter"]) == 100.
    assert float(row["ipsae_pae_cutoff"]) == 10.
    assert row["metascore_calibration"] == meta_score.CALIBRATION_ID
    assert row["metascore_calibration_status"] == expected_status
    features = row["metascore_features"].split(";")
    assert int(row["metascore_feature_count"]) == len(features) == 7
    assert "interface_hb" not in features
    assert float(row["interface_meta_score"]) == pytest.approx(
        sum(meta_score.calibrated_feature_percentile(f, row[f], "af2") for f in features) / len(features)
    )


def test_cache_changed_selection_and_threshold(af2_run):
    directory, models = af2_run
    assert {r["model_used"] for r in score(directory)} == {models[0]}
    assert {r["model_used"] for r in score(directory, contact_thresh=6., models_to_analyse="all")} == {models[1]}


@pytest.mark.parametrize("change", [
    {"pae_filter": 1.}, {"ipsae_pae_cutoff": 1.},
    {"skip_biophysical_scores": False},
])
def test_cache_invalidates_scoring_options(af2_run, monkeypatch, change):
    directory, _ = af2_run
    score(directory)
    calls = []
    original = runner.process
    def recording_process(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(runner, "process", recording_process)
    score(directory)
    assert calls == []
    score(directory, **change)
    assert calls == [1]


@pytest.mark.parametrize("cache_validation", ["stat", "content"])
def test_cache_checks_input_and_csv_contents(af2_run, cache_validation):
    directory, models = af2_run
    first = score(directory, cache_validation=cache_validation)[0]
    structure_path = directory / f"unrelaxed_{models[0]}.pdb"
    before = structure_path.stat()
    write_structure(structure_path, separation=7., plddt=40.)
    # Content mode must detect a same-size edit even with a preserved mtime.
    # Stat mode needs an observable metadata change; the shared filesystem can
    # quantize both mtime and ctime to whole seconds.
    mtime = before.st_mtime_ns if cache_validation == "content" else before.st_mtime_ns + 2_000_000_000
    os.utime(structure_path, ns=(before.st_atime_ns, mtime))
    assert structure_path.stat().st_size == before.st_size
    second = score(directory, cache_validation=cache_validation)[0]
    assert float(first["interface_average_plddt"]) == 90.
    assert float(second["interface_average_plddt"]) == 40.
    path = directory / "interfaces.csv"
    path.write_text(path.read_text().replace(models[0], "edited_model"))
    assert score(directory, cache_validation=cache_validation)[0]["model_used"] == models[0]


def test_warm_cache_does_not_read_input_payloads(af2_run, monkeypatch):
    directory, _ = af2_run
    original = score(directory)
    digest = cache.file_digest
    def guarded_digest(path):
        assert path.parent != directory or path.name == "interfaces.csv", f"Re-read input: {path}"
        return digest(path)
    monkeypatch.setattr(cache, "file_digest", guarded_digest)
    assert score(directory) == original


def test_content_validation_hashes_inputs_on_warm_reuse(af2_run, monkeypatch):
    directory, _ = af2_run
    score(directory, cache_validation="content")
    calls = []
    digest = cache.file_digest
    def recording_digest(path):
        calls.append(path)
        return digest(path)
    monkeypatch.setattr(cache, "file_digest", recording_digest)
    score(directory, cache_validation="content")
    assert directory / "ranking_debug.json" in calls
    assert any(path.suffix == ".pdb" for path in calls)


def test_content_validation_detects_edits_hidden_from_metadata(af2_run, monkeypatch):
    directory, models = af2_run
    csv_path = directory / "interfaces.csv"
    stat_request = cache.request_identity(directory, csv_path)
    content_request = cache.request_identity(directory, csv_path, cache_validation="content")
    original = cache.file_fingerprint
    def frozen_input_metadata(path):
        key = str(path.relative_to(directory))
        return stat_request["inputs"].get(key) or original(path)
    monkeypatch.setattr(cache, "file_fingerprint", frozen_input_metadata)
    write_structure(directory / f"unrelaxed_{models[0]}.pdb", plddt=40.)
    assert cache.request_identity(directory, csv_path) == stat_request
    assert cache.request_identity(directory, csv_path, cache_validation="content") != content_request


def test_changing_validation_mode_revalidates_scores(af2_run, monkeypatch):
    directory, _ = af2_run
    score(directory)
    calls = []
    original = runner.process
    def record(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(runner, "process", record)
    score(directory, cache_validation="content")
    score(directory, cache_validation="content")
    assert calls == [1]


def test_cache_detects_added_removed_and_replaced_inputs(af2_run):
    directory, models = af2_run
    csv_path = directory / "interfaces.csv"
    before = cache.request_identity(directory, csv_path)
    extra = directory / "extra.json"
    extra.write_text("{}")
    assert cache.request_identity(directory, csv_path) != before
    extra.unlink()
    assert cache.request_identity(directory, csv_path) == before
    structure = directory / f"unrelaxed_{models[0]}.pdb"
    stat = structure.stat()
    replacement = structure.with_suffix(".tmp")
    replacement.write_bytes(structure.read_bytes())
    os.utime(replacement, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    replacement.replace(structure)
    assert cache.request_identity(directory, csv_path) != before


def test_png_toggle_generates_missing_plot_without_rescoring(af2_run, monkeypatch):
    directory, models = af2_run
    original = score(directory)
    csv_path = directory / "interfaces.csv"
    csv_stat = csv_path.stat()
    def no_rescore(*args, **kwargs):
        pytest.fail("PNG request recomputed scores")
    monkeypatch.setattr(runner, "process", no_rescore)
    monkeypatch.setattr(runner, "Complex", no_rescore)
    assert score(directory, skip_pae_png=False) == original
    assert (directory / f"pae_{models[0]}.png").is_file()
    assert csv_path.stat().st_mtime_ns == csv_stat.st_mtime_ns
    # A second request should not even reload the model when the plot is current.
    monkeypatch.setattr(AF2Parser, "parse_run", no_rescore)
    assert score(directory, skip_pae_png=False) == original
    assert score(directory, skip_pae_png=True) == original


@pytest.mark.parametrize("failure", ["exception", "no_output"])
def test_failed_png_is_retried_without_invalidating_scores(af2_run, monkeypatch, failure):
    directory, _ = af2_run
    attempts = []
    def broken_plot(*args, **kwargs):
        attempts.append(1)
        if failure == "exception":
            raise OSError("PNG destination unavailable")
        return None
    monkeypatch.setattr(runner, "render_pae_png", broken_plot)
    original = score(directory, skip_pae_png=False)
    manifest = json.loads(cache.manifest_path(directory / "interfaces.csv").read_text())
    assert manifest["complete"] is True
    monkeypatch.setattr(runner, "process", lambda *a, **kw: pytest.fail("Rescored after PNG failure"))
    assert score(directory, skip_pae_png=False) == original
    assert attempts == [1, 1]


def test_skipped_png_is_refreshed_after_input_change(af2_run, monkeypatch):
    directory, models = af2_run
    score(directory, skip_pae_png=False, cache_validation="content")
    (directory / f"pae_{models[0]}.json").write_text(json.dumps([
        {"predicted_aligned_error": np.full((4, 4), 7.).tolist()}
    ]))
    score(directory, skip_pae_png=True, cache_validation="content")
    plotted = []
    original = runner.render_pae_png
    def recording_plot(path, matrix, **kwargs):
        plotted.append(float(matrix[0, 0]))
        return original(path, matrix, **kwargs)
    monkeypatch.setattr(runner, "render_pae_png", recording_plot)
    monkeypatch.setattr(runner, "process", lambda *a, **kw: pytest.fail("Rescored for stale PNG"))
    score(directory, skip_pae_png=False, cache_validation="content")
    assert plotted == [7.]


def test_explicit_unrelaxed_preference_invalidates_cache(af2_run):
    directory, models = af2_run
    write_structure(directory / f"relaxed_{models[0]}.pdb", plddt=40.)
    relaxed = score(directory)[0]
    unrelaxed = score(directory, af2_structure="unrelaxed")[0]
    assert float(relaxed["interface_average_plddt"]) == 40.
    assert float(unrelaxed["interface_average_plddt"]) == 90.
    assert unrelaxed["structure_file"] == f"unrelaxed_{models[0]}.pdb"
    assert score(directory, af2_structure="unrelaxed")[0] == unrelaxed


def test_unrelaxed_preference_falls_back_to_available_relaxed_file(af2_run):
    directory, models = af2_run
    model = models[0]
    (directory / f"unrelaxed_{model}.pdb").rename(directory / f"relaxed_{model}.pdb")
    row = score(directory, af2_structure="unrelaxed")[0]
    assert row["structure_file"] == f"relaxed_{model}.pdb"


def test_copied_cache_does_not_keep_the_old_job_name(af2_run):
    directory, _ = af2_run
    original_job = score(directory)[0]["jobs"]
    relocated = directory.parent / (directory.name + "_relocated")
    shutil.copytree(directory, relocated)
    row = score(relocated)[0]
    assert row["jobs"] == relocated.name != original_job


@pytest.mark.parametrize("invalidate", ["software", "missing_manifest", "corrupt_manifest", "force"])
def test_cache_recomputes_without_current_provenance(af2_run, monkeypatch, invalidate):
    directory, _ = af2_run
    filename = "scores/custom.csv"
    score(directory, per_run_csv_name=filename)
    manifest = cache.manifest_path(directory / filename)
    if invalidate == "software":
        monkeypatch.setattr(cache, "software_identity", lambda: {"changed_calibration": True})
    elif invalidate == "missing_manifest":
        manifest.unlink()
    elif invalidate == "corrupt_manifest":
        manifest.write_text("{BROKEN")
    calls = []
    original = runner.process
    def recording_process(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(runner, "process", recording_process)
    score(directory, per_run_csv_name=filename, force_recompute=invalidate == "force")
    assert calls == [1]
    score(directory, per_run_csv_name=filename)
    assert calls == [1]


def test_concurrent_csv_replacement_cannot_receive_wrong_provenance(af2_run, monkeypatch):
    directory, models = af2_run
    original = cache.write_manifest
    def replace_before_manifest(csv_path, *args, **kwargs):
        csv_path.write_text(csv_path.read_text().replace(models[0], "other_writer"))
        return original(csv_path, *args, **kwargs)
    monkeypatch.setattr(cache, "write_manifest", replace_before_manifest)
    score(directory)
    path = directory / "interfaces.csv"
    manifest = json.loads(cache.manifest_path(path).read_text())
    assert not cache.matches(path, manifest["request"])


@pytest.mark.parametrize("suffix,compress", [("", lambda x: x), (".gz", gzip.compress), (".xz", lzma.compress)])
def test_native_multimer_ranking_and_pickle_pae(af2_run, suffix, compress):
    directory, models = af2_run
    model = models[0]
    (directory / "ranking_debug.json").write_text(json.dumps({"order": [model], "iptm+ptm": {model: .68}}))
    (directory / f"pae_{model}.json").unlink()
    (directory / f"result_{model}.pkl{suffix}").write_bytes(compress(pickle.dumps({
        "iptm": .7, "ptm": .6, "predicted_aligned_error": np.full((4, 4), 2.)
    })))
    _, confidence = AF2Parser().parse_run(directory).load_model(model)
    assert (confidence.iptm, confidence.ptm, confidence.confidence_score) == (.7, .6, .68)
    assert confidence.pae_matrix.shape == (4, 4)
    assert len(score(directory)) == 1


def test_native_multimer_missing_individual_scores_remain_missing(af2_run):
    directory, models = af2_run
    model = models[0]
    (directory / "ranking_debug.json").write_text(json.dumps({"order": [model], "iptm+ptm": {model: .68}}))
    _, confidence = AF2Parser().parse_run(directory).load_model(model)
    assert confidence.iptm is None and confidence.ptm is None
    assert confidence.confidence_score == .68


def test_boltz_backend_does_not_depend_on_model_name(tmp_path):
    rows = []
    for name in ("model_foo_model_0", "myinput_model_0"):
        directory = tmp_path / name
        directory.mkdir()
        write_structure(directory / f"{name}.pdb")
        (directory / f"confidence_{name}.json").write_text(json.dumps({"iptm": .7, "ptm": .6, "confidence_score": .68}))
        np.savez(directory / f"pae_{name}.npz", pae=np.full((4, 4), 2.))
        row = score(directory)[0]
        assert row["backend"] == "boltz2"
        assert meta_score.infer_backend(row) is None  # pooled ladder for Boltz
        assert report._detect_backend([row]) == "Boltz-2"
        rows.append(row)
    assert rows[0]["interface_meta_score"] == rows[1]["interface_meta_score"]


@pytest.mark.parametrize("suffix,compress", [("", lambda x: x), (".gz", gzip.compress), (".xz", lzma.compress)])
def test_corrupt_json_reports_actual_file(tmp_path, suffix, compress):
    path = tmp_path / f"confidences.json{suffix}"
    path.write_bytes(compress(b"{BROKEN"))
    with pytest.raises(ValueError, match="confidences.json"):
        BaseParser._read_json(tmp_path / "confidences.json")
    assert BaseParser._read_json(tmp_path / "optional.json") == {}


@pytest.mark.parametrize("suffix,compress", [(".gz", gzip.compress), (".xz", lzma.compress)])
def test_corrupt_compressed_stream_reports_actual_file(tmp_path, suffix, compress):
    path = tmp_path / f"confidences.json{suffix}"
    data = bytearray(compress(b'{"pae": [[0.0]]}'))
    data[-5] ^= 0xff  # damage the compression trailer, not the JSON
    path.write_bytes(data)
    with pytest.raises(ValueError, match="confidences.json"):
        BaseParser._read_json(tmp_path / "confidences.json")


@pytest.mark.parametrize("payload", [None, "{BROKEN", "[]", "{}"])
def test_af3_required_confidence_errors_keep_file_context(tmp_path, payload):
    directory = tmp_path / "seed-1_sample-0"
    directory.mkdir()
    write_structure(directory / "model.pdb")
    (tmp_path / "ranking_scores.csv").write_text("seed,sample,ranking_score\n1,0,0.8\n")
    (directory / "summary_confidences.json").write_text(json.dumps({"iptm": .7, "ptm": .6}))
    if payload is not None:
        (directory / "confidences.json").write_text(payload)
    run = AF3Parser().parse_run(tmp_path)
    with pytest.raises((FileNotFoundError, ValueError), match="confidences.json") as caught:
        run.load_model(run.order[0])
    assert "only chain-pair minima" not in str(caught.value)


def test_af2_corrupt_pae_does_not_silently_fall_back_to_pickle(af2_run):
    directory, models = af2_run
    model = models[0]
    (directory / f"pae_{model}.json").write_text("{BROKEN")
    (directory / f"result_{model}.pkl").write_bytes(pickle.dumps({"predicted_aligned_error": np.zeros((4, 4))}))
    with pytest.raises(ValueError, match=f"pae_{model}.json"):
        AF2Parser().parse_run(directory).load_model(model)


@pytest.mark.parametrize("pae", [np.zeros((3, 3)), np.zeros((4, 3)), np.zeros(4),
                                  np.full((4, 4), np.nan), np.full((4, 4), -1.)])
def test_complex_rejects_invalid_pae_before_scoring(af2_run, pae):
    directory, models = af2_run
    structure, confidence = AF2Parser().parse_run(directory).load_model(models[0])
    with pytest.raises(ValueError, match="PAE"):
        Complex(structure, replace(confidence, pae_matrix=pae), 8., 100.)


@pytest.mark.parametrize("pae", [np.zeros((5, 5)), np.zeros((3, 3)), np.full((4, 4), np.nan)])
def test_boltz_rejects_unmappable_pae(tmp_path, pae):
    path = tmp_path / "pae.npz"
    np.savez(path, pae=pae)
    with pytest.raises(ValueError, match="pae.npz"):
        Boltz2Parser._load_pae(path, 4, 4, np.arange(4))


def test_boltz_corrupt_pae_is_distinct_from_missing_optional_pae(tmp_path):
    path = tmp_path / "pae.npz"
    path.write_bytes(b"BROKEN")
    with pytest.raises(ValueError, match="pae.npz"):
        Boltz2Parser._load_pae(path, 4, 4, np.arange(4))
    matrix, _ = Boltz2Parser._load_pae(None, 4, 4, np.arange(4))
    assert np.all(matrix == 100.)


def test_af2_structure_choice_and_provenance(af2_run, monkeypatch):
    directory, models = af2_run
    model = models[0]
    score(directory)
    relaxed = directory / f"relaxed_{model}.pdb"
    write_structure(relaxed, plddt=40.)
    glob = Path.glob
    monkeypatch.setattr(Path, "glob", lambda self, pattern, **kwargs: iter(reversed(list(glob(self, pattern, **kwargs)))))
    assert Path(BaseParser._guess_struct(directory, model)) == relaxed
    row = score(directory)[0]
    assert float(row["interface_average_plddt"]) == 40.
    assert row["structure_file"] == relaxed.name


def test_af2_structure_lookup_does_not_match_model_10(tmp_path):
    write_structure(tmp_path / "unrelaxed_model_10.pdb")
    with pytest.raises(ValueError, match="model_1"):
        BaseParser._guess_struct(tmp_path, "model_1")


def test_af2_relaxation_precedence_across_formats_and_rank_fallback(af2_run):
    directory, models = af2_run
    model = models[0]
    unrelaxed = directory / f"unrelaxed_{model}.pdb"
    writer = MMCIFIO()
    writer.set_structure(BaseParser._load_structure(unrelaxed))
    unrelaxed_cif = unrelaxed.with_suffix(".cif")
    writer.save(str(unrelaxed_cif))
    relaxed = directory / f"relaxed_{model}.pdb"
    write_structure(relaxed, plddt=40.)
    assert Path(BaseParser._guess_struct(directory, model)) == relaxed
    unrelaxed.unlink()
    unrelaxed_cif.unlink()
    relaxed.rename(directory / "ranked_0.pdb")
    row = score(directory)[0]
    assert row["structure_file"] == "ranked_0.pdb"
    assert float(row["interface_average_plddt"]) == 40.


def test_legacy_backend_inference_still_works():
    assert meta_score.infer_backend({"model_used": "model_1_multimer_v3_pred_0"}) == "af2"
    assert meta_score.infer_backend({"model_used": "seed-1_sample-0"}) == "af3"
    assert meta_score.infer_backend({"backend": "af3", "model_used": "model_misleading"}) == "af3"
    assert report._detect_backend([{"model_used": "model_1_multimer_v3_pred_0"}]) == "AlphaFold 2"


def test_failed_model_is_retried_and_logs_traceback(af2_run, caplog, monkeypatch):
    directory, models = af2_run
    broken = directory / f"pae_{models[1]}.json"
    broken.write_text("{BROKEN")
    rows = score(directory, models_to_analyse="all")
    assert {r["model_used"] for r in rows} == {models[0]}
    assert any(r.exc_info and str(broken) in r.message for r in caplog.records)
    calls = []
    original = runner.process
    def recording_process(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(runner, "process", recording_process)
    score(directory, models_to_analyse="all")
    assert calls == [1]


def test_failure_after_one_interface_does_not_leave_partial_model_rows(af2_run, monkeypatch):
    directory, models = af2_run
    write_structure(directory / f"unrelaxed_{models[0]}.pdb", chains="ABC")
    (directory / f"pae_{models[0]}.json").write_text(json.dumps([
        {"predicted_aligned_error": np.full((6, 6), 2.).tolist()}
    ]))
    original = runner._interface_row
    calls = []
    def fail_second_interface(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError("synthetic second-interface failure")
        return original(*args, **kwargs)
    monkeypatch.setattr(runner, "_interface_row", fail_second_interface)
    rows = score(directory, models_to_analyse="all")
    assert len(calls) == 3
    assert {row["model_used"] for row in rows} == {models[1]}
