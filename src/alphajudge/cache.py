"""Versioned provenance for per-run CSV reuse.

Input metadata is checked by default; content validation hashes compressed
inputs as stored. Package source (including calibration) and CSV output are
always hashed. Missing/old manifests are cache misses.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import tempfile

SCHEMA_VERSION = 2
_INPUT_SUFFIXES = (".json", ".json.gz", ".json.xz", ".pkl", ".pkl.gz", ".pkl.xz",
                   ".npz", ".cif", ".pdb")


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_path(csv_path: Path) -> Path:
    return csv_path.with_name(csv_path.name + ".meta.json")


def file_fingerprint(path: Path) -> dict[str, int]:
    """Cheap change detection, including file replacements.

    ctime supplements mtime, but coarse/preserved timestamps can still hide
    same-size rewrites. Use content validation when metadata is insufficient.
    """
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns,
            "device": stat.st_dev, "inode": stat.st_ino}


def software_identity() -> dict:
    root = Path(__file__).parent
    dependencies = {}
    for package in ("alphajudge", "numpy", "scipy", "biopython", "matplotlib"):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = "uninstalled"
    return {
        "python": platform.python_version(), "packages": dependencies,
        "source": {str(p.relative_to(root)): file_digest(p) for p in sorted(root.rglob("*.py"))},
    }


def request_identity(directory: Path, csv_path: Path, *, cache_validation: str = "stat", **settings) -> dict:
    if cache_validation not in {"stat", "content"}:
        raise ValueError("cache_validation must be 'stat' or 'content'")
    inputs = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path == csv_path or path.name.endswith(".meta.json"):
            continue
        if path.name.endswith(_INPUT_SUFFIXES) or path.name.endswith("ranking_scores.csv"):
            inputs[str(path.relative_to(directory))] = (
                file_digest(path) if cache_validation == "content" else file_fingerprint(path)
            )
    return {"schema": SCHEMA_VERSION, "directory": str(directory.resolve()),
            "cache_validation": cache_validation,
            "settings": settings, "inputs": inputs,
            "software": software_identity()}


@contextmanager
def atomic_text(path: Path):
    """Publish a complete file with a same-directory atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", newline="", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            yield handle
            handle.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def write_manifest(csv_path: Path, request: dict, *, csv_sha256: str, complete: bool,
                   backend: str, models: list[str], structure_files: dict[str, str],
                   pae_pngs: dict | None = None) -> None:
    manifest = {"request": request, "complete": complete,
                "csv_sha256": csv_sha256, "backend": backend,
                "models": models, "structure_files": structure_files, "pae_pngs": pae_pngs or {}}
    with atomic_text(manifest_path(csv_path)) as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def matches(csv_path: Path, request: dict) -> bool:
    try:
        manifest = json.loads(manifest_path(csv_path).read_text())
        return (manifest.get("complete") is True and manifest.get("request") == request
                and manifest.get("csv_sha256") == file_digest(csv_path))
    except (OSError, ValueError, AttributeError):
        return False
