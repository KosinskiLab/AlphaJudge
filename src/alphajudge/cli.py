from __future__ import annotations

import argparse
import logging

from .report import generate_aggregate_report
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
    report_group = p.add_mutually_exclusive_group()
    report_group.add_argument(
        "--report",
        dest="report",
        action="store_true",
        help="Write an RCSB-style report.pdf next to each per-run interfaces.csv "
             "(default on for single-run scoring, off when --summary is used).",
    )
    report_group.add_argument(
        "--no-report",
        dest="report",
        action="store_false",
        help="Skip per-run report.pdf generation.",
    )
    p.set_defaults(report=None)
    p.add_argument(
        "--aggregate_report",
        default=None,
        help="If set, write an aggregate validation PDF to this path. "
             "Reads the --summary CSV after scoring.",
    )
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    if not args.paths:
        p.error("Provide PATHS")

    write_per_run_report = args.report
    if write_per_run_report is None:
        write_per_run_report = args.summary is None

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
        write_per_run_report=write_per_run_report,
    )

    if args.aggregate_report:
        if not args.summary:
            p.error("--aggregate_report requires --summary")
        generate_aggregate_report(args.summary, out_pdf=args.aggregate_report)


if __name__ == "__main__":
    main()
