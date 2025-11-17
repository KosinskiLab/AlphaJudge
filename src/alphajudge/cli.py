import argparse, logging
from .runner import process

def main():
    p = argparse.ArgumentParser("AlphaJudge interface scoring")
    p.add_argument("--path_to_dir", required=True)
    p.add_argument("--contact_thresh", type=float, default=8.0)
    p.add_argument("--pae_filter", type=float, default=100.0)
    p.add_argument("--models_to_analyse", choices=["best","all"], default="best")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    process(args.path_to_dir, args.contact_thresh, args.pae_filter, args.models_to_analyse)
