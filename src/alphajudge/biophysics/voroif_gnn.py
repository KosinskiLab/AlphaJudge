from __future__ import annotations
import subprocess
import logging
from pathlib import Path
import os

logger = logging.getLogger(__name__)

class VoroIFGNNWrapper:
    """Wrapper for the external voroif-gnn-v2-app tool."""

    def __init__(
        self,
        app_path: str | Path = "../voroif-gnn-v2-app",
        conda_path: str | Path = "~/miniforge3/envs/voroif-gnn-v2-env",
        conda_env: str = "voroif-gnn-v2-env",
    ):
        self.app_path = Path(os.path.expanduser(str(app_path)))
        self.conda_path = Path(os.path.expanduser(str(conda_path)))
        self.conda_env = conda_env
        self.executable = self.app_path / "voronota-js-voroif-gnn-v2"

    def run(self, model_path: Path) -> dict[str, float]:
        """
        Run voroif-gnn-v2-app on a given model file and return scores.
        Only supports PDB files (as per tool documentation).
        """
        if not self.executable.exists():
            logger.warning(f"VoroIF-GNN executable not found at {self.executable}")
            return {}

        if model_path.suffix.lower() != ".pdb":
            logger.warning(f"VoroIF-GNN wrapper currently only supports .pdb files, got {model_path}")
            # If it's a .cif, we could convert it if needed!
            return {}

        cmd = [
            str(self.executable),
            "--conda-path", str(self.conda_path),
            "--conda-env", self.conda_env
        ]

        try:
            # Must pass the *list* of input PDB files via stdin (i.e., path itself).
            input_paths = str(model_path) + "\n"
            result = subprocess.run(
                cmd,
                input=input_paths,
                capture_output=True,
                text=True,
                check=True,
                cwd=str(self.app_path)
            )
            return self._parse_output(result.stdout)
        except subprocess.CalledProcessError as e:
            logger.error(f"VoroIF-GNN execution failed: {e.stderr}")
            return {}
        except Exception as e:
            logger.error(f"Error running VoroIF-GNN: {e}")
            return {}

    def _parse_output(self, stdout: str) -> dict[str, float]:
        """
        Parse the space-separated table output of the tool.
        Example output header: ID pgoodness area pgoodness_average pcadscore num_of_residues residue_pcadscore
        """
        lines = stdout.strip().splitlines()
        if len(lines) < 2:
            return {}

        headers = lines[0].split()
        values = lines[1].split()

        if len(headers) != len(values):
            return {}

        scores = {}
        for h, v in zip(headers, values):
            if h == "ID":
                continue
            try:
                scores[h] = float(v)
            except ValueError:
                continue
        return scores
