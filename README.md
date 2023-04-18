[![PyPI](https://img.shields.io/pypi/v/petprep-hmc)](https://pypi.org/project/petprep-hmc/0.0.1/)

# PETPrep Head Motion Correction workflow (BIDS App)
This BIDS App provides a pipeline for preprocessing and head motion correction of Positron Emission Tomography (PET) data following the Brain Imaging Data Structure (BIDS) standard.

## Features

- BIDS compliant PET data input and output
- Rigid body head motion correction using robust registration
- Motion parameter estimation for further analysis
- Integration with BIDS Derivatives for compatibility with other BIDS Apps
- Compatible with Python 3

## Requirements (for more info see the environment.yml file)

- Python 3.9+
- Nipype
- NiBabel
- FSL
- FreeSurfer

## Installation

Clone the repository and install the required packages:

<pre>
git clone https://github.com/mnoergaard/petprep_hmc.git
cd petprep_hmc
pip install -e .
</pre>

The package is also pip installable and can be installed using the following command

<pre>
pip install petprep-hmc==0.0.1
</pre>

## Usage

To run the PETPrep Head Motion Correction BIDS App, use the following command:

`python3 run.py --bids_dir /path/to/bids_input --output_dir /path/to/bids_output --analysis_level participant [--participant_label PARTICIPANT_LABEL]`

- `--bids_dir`: Path to the input BIDS dataset
- `--output_dir`: Path to the output directory for preprocessed data
- `--analysis_level`: Level of the analysis that will be performed. Multiple participant level analyses can be run independently (in parallel) using the same output_dir.
- `--participant_label`: (Optional) A single participant label or a space-separated list of participant labels to process. If not provided, all participants in the dataset will be processed.
- `--n_procs`: (Optional) number of processors allocated to be used when running the workflow
- `--skip_bids_validator`: (Optional) Whether or not to perform BIDS dataset validation.

For example, to process participant `sub-01`, use the following command:

`python3 run.py --bids_dir /data/bids_input --output_dir /data/bids_output --participant_label 01`

## Output

The output will be saved in the specified `--output_dir` following the BIDS Derivatives standard. Preprocessed PET data will be stored in the `petprep_hmc` directory in the derivatives directory, along with the estimated motion parameters (confounds) and motion plots.

## Support

For questions or bug reports, please open an issue on the GitHub repository.

## License

This BIDS App is released under the Apache 2.0 license. Please see the `Apache 2.0 license` file for more details.
