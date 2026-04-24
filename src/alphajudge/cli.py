from __future__ import annotations

import argparse
import logging

from .runner import process_many


def main() -> None:
    p = argparse.ArgumentParser("AlphaJudge interface scoring")
    p.add_argument("paths", nargs="*", help="One or more run directories or roots")
    p.add_argument("--contact_thresh", type=float, default=8.0)
    p.add_argument("--pae_filter", type=float, default=100.0)
    p.add_argument("--ipsae_pae_cutoff", type=float, default=10.0)
    p.add_argument("--models_to_analyse", choices=["best","all"], default="best")
    p.add_argument("-r","--recursive", action="store_true", help="Recursively search for runs under given PATHS")
    p.add_argument("-o","--summary", help="Write aggregated CSV across runs to this path")
    p.add_argument(
        "--force_recompute",
        action="store_true",
        help="Ignore existing per-run CSVs and recompute scores",
    )
    p.add_argument(
        "--skip_pae_png",
        action="store_true",
        help="Do not write per-model PAE heatmap PNG files",
    )
    p.add_argument(
        "--per_run_csv_name",
        default="interfaces.csv",
        help="Filename to write inside each processed run directory",
    )
    p.add_argument(
        "--cores",
        type=int,
        default=1,
        help="Number of processes to use across run directories (0 = all available cores)",
    )
    p.add_argument(
        "--voroif_gnn_path",
        default=None,
        help="Path to the voroif-gnn-v2-app directory (if provided, VoroIF scores are computed)",
    )
    p.add_argument(
        "--voroif_conda_path",
        default=None,
        help="Path to the conda installation or environment for voroif-gnn-v2-env",
    )
    p.add_argument(
        "--voroif_conda_env",
        default=None,
        help="Name of the conda environment for voroif-gnn-v2",
    )
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    if args.paths:
        process_many(
            args.paths,
            args.contact_thresh,
            args.pae_filter,
            args.models_to_analyse,
            recursive=args.recursive,
            summary_csv=args.summary,
            cores=args.cores,
            ipsae_pae_cutoff=args.ipsae_pae_cutoff,
            force_recompute=args.force_recompute,
            per_run_csv_name=args.per_run_csv_name,
            skip_pae_png=args.skip_pae_png,
            voroif_gnn_path=args.voroif_gnn_path,
            voroif_conda_path=args.voroif_conda_path,
            voroif_conda_env=args.voroif_conda_env,
        )
    else:
        p.error("Provide PATHS")


if __name__ == "__main__":
    main()
