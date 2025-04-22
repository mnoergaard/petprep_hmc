[![PyPI](https://img.shields.io/pypi/v/petprep-hmc)](https://pypi.org/project/petprep-hmc/0.0.8/) [![Docker Hub](https://img.shields.io/docker/automated/martinnoergaard/petprep_hmc)](https://hub.docker.com/r/martinnoergaard/petprep_hmc) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14168647.svg)](https://doi.org/10.5281/zenodo.14168647)

![Read the Docs](https://readthedocs.org/projects/petprep_hmc/badge/?version=latest)

| CI  | Status |
|---------| ------ |
| `docker build` | ![Docker Build](https://codebuild.us-east-1.amazonaws.com/badges?uuid=eyJlbmNyeXB0ZWREYXRhIjoiSlFmZFpmNVZRbXFXc1ZLMXZVRnpnVVdvVVZucEZ5cS9ib0JyOGhLNTJsRWtROU12Y0hUUUlRaXBwd0ZDbTR2MGJzeGZFYmJNdlRqdndMRVRQWFZzQ2M4PSIsIml2UGFyYW1ldGVyU3BlYyI6IkVzMzhLTlFBelNrMXl6Tm4iLCJtYXRlcmlhbFNldFNlcmlhbCI6MX0%3D&branch=main) |
| `docker run` | ![Docker Run](https://codebuild.us-east-1.amazonaws.com/badges?uuid=eyJlbmNyeXB0ZWREYXRhIjoiOEYxenJydy8wQWJrUDFQU3dZWGlkRjhMbW16OHl0RWs4VlF6R0NtR0FLZGZHQkw3emEzRGxzMkg4SkJ2T2lZKzQ0Zk9PblZUeFIvSmNZWHlLTEtVU29zPSIsIml2UGFyYW1ldGVyU3BlYyI6IjgwSzVpTDY0NmltcGd3MU8iLCJtYXRlcmlhbFNldFNlcmlhbCI6MX0%3D&branch=main) |

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

```bash
git clone https://github.com/mnoergaard/petprep_hmc.git
cd petprep_hmc
pip install -e .
```

The package is also pip installable and can be installed using the following command

```bash
pip install petprep-hmc
```

## Usage

To run the PETPrep Head Motion Correction BIDS App, use the following command:

`python3 run.py /path/to/bids_input /path/to/bids_output participant [--participant_label PARTICIPANT_LABEL]`

- `bids_dir`: Path to the input BIDS dataset
- `output_dir`: Path to the output directory for preprocessed data
- `analysis_level`: Level of the analysis that will be performed. Multiple participant level analyses can be run independently (in parallel) using the same output_dir.
- `--participant_label`: (Optional) A single participant label or a space-separated list of participant labels to process (e.g. sub-01 sub02). If not provided, all participants in the dataset will be processed.
- `--participant_label_exclude`: (Optional) A single participant label or a space-separated list of participant labels to exclude in the processing (e.g. sub-01 sub02). If not provided, all participants in the dataset will be processed.
- `--session_label`: (Optional) A single session label or a space-separated list of session labels to process (e.g. ses-01 ses-02). If not provided, all sessions in the dataset will 
be processed
- `--session_label_exclude`: (Optional) A single sessien label or a space-separated list of session labels to exclude in the processing (e.g. ses-01 ses-02). If not provided, all sessions in the dataset will be processed.
- `--mc_start_time`: (Optional) Start time for when to perform motion correction (subsequent frame will be chosen) in seconds (default = 120 seconds).
- `--mc_fwhm`: (Optional) FWHM for smoothing of frames prior to estimating motion (default = 10mm).
- `--mc_thresh`: (Optional) Threshold below the following percentage (0-100) of framewise ROBUST RANGE prior to estimating motion correction (default = 20).
- `--n_procs`: (Optional) Number of processors allocated to be used when running the workflow.
- `--no_resample`: (Optional) Whether or not to resample the motion corrected PET data to lowest x/y/z dim in original data (default = False). 
- `--skip_bids_validator`: (Optional) Whether or not to perform BIDS dataset validation.

For example, to process participant `sub-01`, use the following command:

`python3 run.py /data/bids_input /data/bids_output participant --participant_label sub-01`

## Outputs

Preprocessed PET data along with the estimated motion parameters (confounds) and motion plots will be stored in the directory specified by the second command line argument. If no output directory is specified the outputs will saved to `<bids_dir>/derivatives/petprep_hmc` with `<bids_dir>` corresponding to the first command line argument.

## Installation and Running the Code using Docker

### Prerequisites
To run petprep_hmc workflow using Docker, you must first have Docker installed on your system. You can download and install Docker from https://www.docker.com/.

### Pulling the Docker Image
Once you have Docker installed, you can pull the petprep_hmc Docker image from Docker Hub by running the following command:

```bash
docker pull martinnoergaard/petprep_hmc:latest
```

### Running the Docker Container
To run the petprep_hmc Docker container, use the following command:

```bash
docker run -it --rm \
    -v /path/to/bids_input:/data/input \
    -v /path/to/bids_output:/data/output \
    -v /path/to/freesurfer_license:/opt/freesurfer/license.txt \
    martinnoergaard/petprep_hmc:latest \
    /data/input /data/output participant [--participant_label PARTICIPANT_LABEL]
```

This command mounts your local input and output directories, as well as the FreeSurfer license file, to the Docker container. The petprep_hmc script is then executed within the container, processing the input data and saving the results to the specified output directory.

```bash
singularity exec -e --bind license.txt:/opt/freesufer/license.txt docker://martinnoergaard/petprep_hmc:latest python3 /opt/petprep_hmc/run.py
```

## Boilerplate for publications
Motion correction of the dynamic PET data was performed using PETPrep HMC (petprep_hmc version X.X.X, https://github.com/mnoergaard/petprep_hmc; Nørgaard et al., 20XX), a Nipype-based BIDS App (Gorgolewski et al., 2017).
The head motion was estimated using a frame-based robust registration approach to an unbiased mean volume implemented in FreeSurfer's mri_robust_register (Reuter et al., 2010), combined with preprocessing steps using tools from FSL (Jenkinson et al., 2012). Specifically, for the estimation of head motion, each frame was initially smoothed with a Gaussian filter (full-width half-maximum [FWHM] = 10 mm), followed by thresholding at 20% of the intensity range to reduce noise and improve registration accuracy (removing stripe artefacts from filtered back projection reconstructions). Motion was estimated selectively of frames acquired after 120 seconds post-injection of the tracer, as frames before this often contain low count statistics. Frames preceding 120 seconds were corrected using identical transformations as derived for the first frame after 120 seconds. The robust registration (mri_robust_register) algorithm utilized settings optimized for PET data: intensity scaling was enabled, automated sensitivity detection was activated, and the Frobenius norm threshold for convergence was set at 0.0001, ensuring precise and consistent alignment across frames. After head motion estimation, the obtained transforms were applied to the original data and resampled, providing the final motion corrected output data.

Visual quality control outputs, including plots illustrating translational and rotational motion parameters, and framewise displacement, were generated to assess correction quality and ensure robustness of the registration procedure.

References:

Nørgaard et al., 20XX. PETPrep HMC: Head Motion Correction Workflow for Dynamic PET Data. Zenodo. https://doi.org/XXXX (Replace with actual DOI upon availability).

Gorgolewski, K.J. et al., 2017. The Brain Imaging Data Structure: A standard for organizing and describing outputs of neuroimaging experiments. Scientific Data, 3, 160044. https://doi.org/10.1038/sdata.2016.44

Reuter, M. et al., 2010. Highly accurate inverse consistent registration: A robust approach. NeuroImage, 53(4), 1181–1196. https://doi.org/10.1016/j.neuroimage.2010.09.016

Jenkinson, M. et al., 2012. FSL. NeuroImage, 62(2), 782–790. https://doi.org/10.1016/j.neuroimage.2011.09.015

## Support

For questions or bug reports, please open an issue on the GitHub repository.

## License

This BIDS App is released under the Apache 2.0 license. Please see the `Apache 2.0 license` file for more details.
