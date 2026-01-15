import pytest
import polars as pl
from pathlib import Path
import subprocess

from alphajudge.runner import process_many
from alphajudge.parsers import pick_parser

# Define paths
ACQUITTED_CSV = Path("test_data/af2/pos_dimers/CCP4_benchmarks.csv")
DEFENDANT_CSV = Path("test_data/af2/pos_dimers/AJ_summary.csv")

@pytest.fixture(scope="module")
def af2_benchmark() -> Path:
    """The 200 AF2 positive dimer for regression checks."""
    return Path("test_data/af2/pos_dimers/")

def judge_benchmark(af2_benchmark: Path):
    parser = pick_parser(af2_benchmark)
    assert parser.name == "af2"
    process_many(
        str(af2_benchmark),
        12.0,
        100.0,
        recursive = True,
        summary_csv = af2_benchmark / "AJ_summary.csv",
    )

def test_black_box_comparison_test():
    """
    Compares AlphaJudge output (summary.csv) with the acquitted reference from prior runs.
    Verifies that specific columns match within a tolerance of 1e-6.
    """
    assert ACQUITTED_CSV.exists(), f"{ACQUITTED_CSV} not found"
    assert DEFENDANT_CSV.exists(), f"{DEFENDANT_CSV} not found"

    # Load dataframes
    df_acquitted = pl.read_csv(ACQUITTED_CSV)   # Acquitted is the reference
    df_defendant = pl.read_csv(DEFENDANT_CSV)   # Defendant is the new run

    # Column mapping: Defendant -> Acquitted
    column_mapping = {
        "jobs": "jobs",
        "iptm_ptm": "iptm_ptm",
        "iptm": "iptm",
        "pDockQ/mpDockQ": "pDockQ/mpDockQ",
        "average_interface_pae": "average_interface_pae",
        "interface_average_plddt": "average_interface_plddt",
        "interface_num_intf_residues": "Num_intf_residues",
        "interface_polar": "Polar",
        "interface_hydrophobic": "Hydrophobic",
        "interface_charged": "Charged",
        "interface_contact_pairs": "contact_pairs",
        "interface_sc": "sc",
        "interface_hb": "hb",
        "interface_sb": "sb",
        "interface_solv_en": "int_solv_en",
        "interface_area": "int_area",
        "interface_score": "pi_score"
    }
    
    # Rename Acquitted DF columns to match Defendant DF columns for easier comparison or just join.
    aj_cols = list(column_mapping.keys())
    df_selected = df_acquitted.select(aj_cols)
    
    # Rename columns in Acquitted DF to match Defendant DF column names.
    rename_map = {k: v for k, v in column_mapping.items() if k != v}
    if rename_map:
        df_selected = df_selected.rename(rename_map)

    # Check if 'jobs' exists in both
    assert "jobs" in df_defendant.columns
    assert "jobs" in df_selected.columns

    # Join on "jobs"
    joined_df = df_defendant.join(df_selected, on="jobs", how="inner", suffix="_judged")

    # Columns to compare (values from column_mapping values, excluding 'jobs')
    cols_to_compare = [v for k, v in column_mapping.items() if k != "jobs"]

    for col in cols_to_compare:
        col_judged = f"{col}_judged"
        
        # Check if columns exist
        if col not in joined_df.columns:
            pytest.fail(f"Expected column {col} missing in joined dataframe")
                    
        s_defendant = joined_df.get_column(col)
        s_judged = joined_df.get_column(col_judged)

        # Filter out where either is null
        valid_mask = s_defendant.is_not_null() & s_judged.is_not_null()
        
        diff = (s_defendant.filter(valid_mask) - s_judged.filter(valid_mask)).abs()
        max_diff = diff.max()
        
        if max_diff is not None and max_diff > 1e-6:
            # Find failing rows for appropriate error message
            failing = joined_df.filter(
                valid_mask & ((pl.col(col) - pl.col(col_judged)).abs() > 1e-6)
            )
            pytest.fail(f"Column '{col}' mismatch. Max diff: {max_diff}. Failing rows:\n{failing.select(['jobs', col, col_judged])}")
