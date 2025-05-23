import os
import subprocess
import tempfile
from pathlib import Path
import json


def test_cli():
    repo_root = Path(__file__).parent.parent.resolve()
    bids_dir = repo_root / "data"

    with tempfile.TemporaryDirectory() as tmp_output_dir:
        output_dir = Path(tmp_output_dir) / "derivatives"
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = f"python3 run.py --bids_dir {bids_dir} --output_dir {output_dir}"
        exit_code = subprocess.call(cmd, shell=True)
        assert exit_code == 0

        json_files = list(output_dir.rglob("*desc-mc_pet.json"))
        assert json_files, "No output JSON files found"
        with open(json_files[0]) as jf:
            data = json.load(jf)
            assert "FrameDuration" in data
            # check that metadata from source JSON is retained
            assert "Manufacturer" in data
            # ensure QC report path stored and file exists
            assert "QC" in data
            assert os.path.exists(data["QC"])
