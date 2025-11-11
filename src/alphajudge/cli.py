import argparse, logging
from .runner import process_many

def main():
    p = argparse.ArgumentParser("AlphaJudge interface scoring")
    p.add_argument("paths", nargs="*", help="One or more run directories or roots")
    p.add_argument("--contact_thresh", type=float, default=8.0)
    p.add_argument("--pae_filter", type=float, default=100.0)
    p.add_argument("--models_to_analyse", choices=["best","all"], default="best")
    p.add_argument("-r","--recursive", action="store_true", help="Recursively search for runs under given PATHS")
    p.add_argument("-o","--summary", help="Write aggregated CSV across runs to this path")
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
        )
    else:
        p.error("Provide PATHS")
