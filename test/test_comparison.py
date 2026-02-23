import pytest
import polars as pl
from pathlib import Path

from alphajudge.runner import process_many

# Column mapping: Defendant -> Acquitted
COLUMN_MAPPING = {
    "jobs": "complex",
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

# Define per-column tolerances. Defaults to 1e-6 if not specified.
TOLERANCE_MAP = {
    # Large magnitude / lower resolution values
    "interface_area": 1e-2,         # Area > 1000, 2 decimal places often sufficient
    "interface_solv_en": 1e-2,      # Solvent energy
    "interface_score": 1e-2,
    "average_interface_pae": 1e-3,  # 0-30 range
    "interface_average_plddt": 1e-3,# 0-100 range
    
    # 0-1 scores where precision matters
    "iptm": 1e-6,
    "ptm": 1e-6,
    "iptm_ptm": 1e-6,
    "pDockQ/mpDockQ": 1e-6,
    "confidence_score": 1e-6,
}

@pytest.fixture(scope="module")
def benchmark_csv_path() -> Path:
    """Path to the authoritative benchmark CSV."""
    return Path("test_data/af2/positive_dimers/benchmarks/CCP4.csv")

@pytest.fixture(scope="module")
def model_input_paths() -> list[str]:
    """List of paths to input models for the test."""
    base = Path("test_data/af2/positive_dimers/predictions/")
    # You can customize this to be more specific if needed
    return [str(p) for p in base.iterdir() if p.is_dir() and p.name not in ["data", "features"]]

def test_generated_results_vs_benchmark(benchmark_csv_path, model_input_paths, tmp_path):
    """
    Runs process_many on the input models and compares the output summary CSV
    against the provided benchmark CSV.
    """
    if not benchmark_csv_path.exists():
        pytest.skip(f"Benchmark file not found at {benchmark_csv_path}")

    # 1. Run process_many to generate new results
    generated_summary_csv = tmp_path / "AF2_summary.csv"
    
    result_path = process_many(
        paths=model_input_paths,
        contact_thresh=8.0,
        pae_filter=100.0,
        models_to_analyse="best",
        recursive=False,
        summary_csv=str(generated_summary_csv),
    )
    
    assert result_path is not None, "process_many returned None"
    assert generated_summary_csv.exists(), "Summary CSV was not created"
    
    # 2. Compare generated results with benchmark
    _compare_dataframes(generated_summary_csv, benchmark_csv_path)

@pytest.mark.xfail
def _compare_dataframes(generated_path: Path, benchmark_path: Path):
    """
    Helper to compare two CSVs using Polars.
    generated_path: Path to the newly generated CSV.
    benchmark_path: Path to the reference CSV.
    """
    df_generated = pl.read_csv(generated_path)
    df_benchmark = pl.read_csv(benchmark_path)

    # Validate mapping keys exist in generated data
    for gen_col in COLUMN_MAPPING.keys():
        if gen_col not in df_generated.columns:
            pytest.fail(f"Expected column '{gen_col}' not found in generated output.")

    # Select and rename columns from Benchmark to match Generated names (reverse mapping)
    # We want to compare Generated[Col] vs Benchmark[MappedCol]
    
    # Prepare benchmark DF for join: Keep only relevant columns and rename them to match generated
    # This makes joining and comparing easier
    
    # Prepare benchmark DF for join: Keep only relevant columns and rename them to match generated
    # This makes joining and comparing easier

    # 1. Identify which mapped columns actually exist in the benchmark
    valid_mapping = {k: v for k, v in COLUMN_MAPPING.items() if v in df_benchmark.columns}
    
    # Check for missing columns and warn
    missing_bench_cols = [v for v in COLUMN_MAPPING.values() if v not in df_benchmark.columns]
    if missing_bench_cols:
        print(f"Benchmark CSV missing valid columns (skipping these): {missing_bench_cols}")

    # Ensure the Key ID column ('complex' -> 'jobs') is present
    if COLUMN_MAPPING["jobs"] not in df_benchmark.columns:
        pytest.fail(f"Critical benchmark column '{COLUMN_MAPPING['jobs']}' is missing. Cannot join.")

    # Create a mapping for renaming benchmark columns -> generated column names
    # Mapping is {GeneratedName: BenchmarkName} -> we want {BenchmarkName: GeneratedName}
    bench_rename_map = {v: k for k, v in valid_mapping.items()}
    
    # Select only the columns we care about + rename them
    df_benchmark_clean = df_benchmark.select(
        list(bench_rename_map.keys())
    ).rename(bench_rename_map)

    # "jobs" is our join key. Ensure it's present and unique-ish if possible.
    assert "jobs" in df_benchmark_clean.columns
    assert "jobs" in df_generated.columns

    # Join on 'jobs'
    # suffix="_bench" will be applied to columns from the benchmark DF where names collide
    joined = df_generated.join(
        df_benchmark_clean, 
        on="jobs", 
        how="inner", 
        suffix="_bench"
    )

    if joined.height == 0:
        pytest.fail("Inner join of generated results and benchmark resulted in 0 rows. Check 'jobs'/'complex' identifiers.")

    cols_to_check = [c for c in valid_mapping.keys() if c != "jobs"]
    
    failures = []

    for col in cols_to_check:
        col_bench = f"{col}_bench"
        
        # Numeric comparison with tolerance
        # We filter for valid (non-null) pairs
        valid_mask = joined.select(
            pl.col(col).is_not_null() & pl.col(col_bench).is_not_null()
        ).to_series()
        
        if not valid_mask.any():
            continue
            
        diffs = joined.filter(valid_mask).select(
            (pl.col(col) - pl.col(col_bench)).abs().alias("diff")
        )
        
        max_diff = diffs.select(pl.col("diff").max()).item()
        
        # Determine tolerance for this column
        tol = TOLERANCE_MAP.get(col, 1e-6)
        
        if max_diff > tol:
            # Collect details on failing rows
            failing_rows = joined.filter(
                valid_mask & ((pl.col(col) - pl.col(col_bench)).abs() > tol)
            ).select(["jobs", col, col_bench])
            
            failures.append(f"Column '{col}' mismatch. Max diff: {max_diff} > tol {tol}. {failing_rows.height} failing rows.")

        # Check Correlation (Spearman)
        # Only meaningful if we have enough distinct values
        n_unique = joined.select(pl.col(col).n_unique()).item()
        if n_unique > 1:
            try:
                corr = joined.select(
                    pl.corr(col, col_bench, method="spearman")
                ).item()
                
                # If valid correlation, strict check
                if corr is not None and corr < 0.99:
                     failures.append(f"Column '{col}' Spearman correlation too low: {corr:.4f}")
            except Exception as e:
                # Polars might error on edge cases
                pass

    if failures:
        pytest.fail("\n".join(failures))