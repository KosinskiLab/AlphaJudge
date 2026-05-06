#!/usr/bin/env python3
"""Build a target-only benchmark prediction tree from a frozen pair list."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ORGANISMS = ("arabidopsis", "ecoli", "human", "yeast")
MODELS = ("af2", "af3")
PAIRSETS = (("pos_pairs", "pos"), ("neg_pairs", "neg"))

DEFAULT_PAIR_ROOT = Path(
    "/g/transform/kosinski/dima/IntAct_BioGRID_STRING/benchmark_26/"
    "frozen_inputs/pairmatched_20260505_fetch3_march2"
)
DEFAULT_SOURCE_ROOT = Path("/scratch/dima/benchmark_26/predictions")
DEFAULT_VIEW_ROOT = Path("/scratch/dima/benchmark_26_pairmatched_20260505")


def read_pairs(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def make_link(source: Path, dest: Path, *, force: bool) -> str:
    if dest.is_symlink():
        if dest.resolve() == source.resolve():
            return "existing"
        if not force:
            raise FileExistsError(f"{dest} already points to {dest.resolve()}")
        dest.unlink()
    elif dest.exists():
        if not force:
            raise FileExistsError(f"{dest} already exists")
        if dest.is_dir():
            shutil.rmtree(dest)
        else:
            dest.unlink()

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.symlink_to(source.resolve(), target_is_directory=True)
    return "linked"


def maybe_link_file(source: Path, dest: Path, *, force: bool) -> str:
    if not source.exists():
        return "missing"
    if dest.is_symlink():
        if dest.resolve() == source.resolve():
            return "existing"
        if not force:
            raise FileExistsError(f"{dest} already points to {dest.resolve()}")
        dest.unlink()
    elif dest.exists():
        if not force:
            return "existing"
        dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.symlink_to(source.resolve())
    return "linked"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-root", type=Path, default=DEFAULT_PAIR_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--view-root", type=Path, default=DEFAULT_VIEW_ROOT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--require-completed",
        action="store_true",
        help="Only link pair directories that already contain completed_fold.txt.",
    )
    args = parser.parse_args()

    linked = existing = missing = incomplete = 0
    for organism in ORGANISMS:
        for model in MODELS:
            for pairset, short in PAIRSETS:
                pairs = read_pairs(args.pair_root / "by_org" / f"{organism}_{short}.pairs.txt")
                source_group = args.source_root / organism / model / pairset
                view_group = args.view_root / "predictions" / organism / model / pairset
                (view_group / "predictions").mkdir(parents=True, exist_ok=True)

                for aux_name in ("ipsae_10_8.csv",):
                    maybe_link_file(source_group / aux_name, view_group / aux_name, force=args.force)

                for pair in pairs:
                    source_pair = source_group / "predictions" / pair
                    dest_pair = view_group / "predictions" / pair
                    if not source_pair.exists():
                        missing += 1
                        continue
                    if args.require_completed and not (source_pair / "completed_fold.txt").exists():
                        incomplete += 1
                        continue
                    status = make_link(source_pair, dest_pair, force=args.force)
                    if status == "linked":
                        linked += 1
                    else:
                        existing += 1

    print(f"view_root={args.view_root}")
    print(f"linked={linked}")
    print(f"existing={existing}")
    print(f"missing_source_dirs={missing}")
    print(f"incomplete_skipped={incomplete}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
