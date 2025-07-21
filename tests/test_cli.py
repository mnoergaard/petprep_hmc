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

        txt_files = list(output_dir.rglob("*_xfm.txt"))
        assert txt_files, "No transform text files found"

        petref_files = list(output_dir.rglob("*_desc-hmc_petref.nii.gz"))
        assert petref_files, "No reference image found"


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


def test_cli_group_high_motion():
    repo_root = Path(__file__).parent.parent.resolve()
    bids_dir = repo_root / "data"

    with tempfile.TemporaryDirectory() as tmp_output_dir:
        derivatives = Path(tmp_output_dir) / "derivatives" / "petprep_hmc" / "sub-01"
        derivatives.mkdir(parents=True, exist_ok=True)

        confounds_tsv = derivatives / "sub-01_desc-confounds_timeseries.tsv"
        confounds_tsv.write_text("median_tot\n6\n7\n")

        cmd = f"python3 run.py {bids_dir} {Path(tmp_output_dir) / 'derivatives'} group"
        exit_code = subprocess.call(cmd, shell=True)
        assert exit_code == 0

        report_file = Path(tmp_output_dir) / "derivatives" / "group_report.html"
        plot_file = Path(tmp_output_dir) / "derivatives" / "group_motion.png"
        assert report_file.exists()
        assert plot_file.exists()
        html = report_file.read_text()
        assert "sub-01" in html
