import os
import subprocess
import tempfile
from pathlib import Path


def test_cli():
    repo_root = Path(__file__).parent.parent.resolve()
    bids_dir = repo_root / "data"

    with tempfile.TemporaryDirectory() as tmp_output_dir:
        output_dir = Path(tmp_output_dir) / "derivatives"
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = f"python3 run.py --bids_dir {bids_dir} --output_dir {output_dir}"
        exit_code = subprocess.call(cmd, shell=True)
        assert exit_code == 0


def test_cli_group_level():
    repo_root = Path(__file__).parent.parent.resolve()
    bids_dir = repo_root / "data"
    output_dir = bids_dir / "derivatives"

    cmd = (
        f"python3 run.py --bids_dir {bids_dir} --output_dir {output_dir} "
        f"--analysis_level group"
    )
    exit_code = subprocess.call(cmd, shell=True)
    assert exit_code == 0
    assert (output_dir / "group_report.html").exists()