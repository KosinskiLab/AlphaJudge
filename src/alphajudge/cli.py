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
        "--skip_biophysical_scores",
        action="store_true",
        help="Skip expensive biophysical calculations (hydrogen bonds, salt bridges, disulfides, shape complementarity, buried surface area, solvation energy) to save time",
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
            skip_biophysical_scores=args.skip_biophysical_scores,
        )
    else:
        p.error("Provide PATHS")


if __name__ == "__main__":
    main()
