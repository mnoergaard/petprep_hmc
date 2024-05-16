import argparse
import os
import sys
from bids import BIDSLayout
import nipype.interfaces.freesurfer as fs
import nipype.interfaces.fsl as fsl
from nipype.interfaces.utility import IdentityInterface
from nipype.pipeline import Workflow
from nipype import Node, Function, MapNode
from nipype.interfaces.io import SelectFiles
from pathlib import Path
import glob
import re
import shutil
import json
from niworkflows.utils.misc import check_valid_fs_license
from petprep_hmc.utils import plot_mc_dynamic_pet
from petprep_hmc.bids import collect_data
import os.path as op
import yaml
import petprep_hmc


__version__ = open(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'version')).read()


def main(args):
    # Check whether BIDS directory exists and instantiate BIDSLayout
    if os.path.exists(args.bids_dir):
        if not args.skip_bids_validator:
            layout = BIDSLayout(args.bids_dir, validate=True)
        else:
            layout = BIDSLayout(args.bids_dir, validate=False)
    else:
        raise Exception('BIDS directory does not exist')

    # Check whether FreeSurfer license is valid
    if check_valid_fs_license() is not True:
        raise Exception('You need a valid FreeSurfer license to proceed!')

    # Check whether FSL is installed
    if check_fsl_installed() is not True:
        raise Exception('FSL is not installed or sourced')

    # Get all PET files
    if args.participant_label is None:
        args.participant_label = layout.get(suffix='pet', target='subject', return_type='id')

    # Create derivatives directories
    if args.output_dir is None:
        output_dir = os.path.join(args.bids_dir, 'derivatives', 'petprep_hmc')
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Run workflow
    main = init_petprep_hmc_wf()
    main.run(plugin='MultiProc', plugin_args={'n_procs': int(args.n_procs)})

    # Loop through directories and store according to PET-BIDS specification
    mc_files = glob.glob(os.path.join(Path(args.bids_dir), 'petprep_hmc_wf', '*', '*', '*', 'mc.nii.gz'))
    confound_files = glob.glob(os.path.join(Path(args.bids_dir), 'petprep_hmc_wf', '*', '*', '*', 'hmc_confounds.tsv'))
    movement = glob.glob(os.path.join(Path(args.bids_dir), 'petprep_hmc_wf', '*', '*', '*', 'movement.png'))
    rotation = glob.glob(os.path.join(Path(args.bids_dir), 'petprep_hmc_wf', '*', '*', '*', 'rotation.png'))
    translation = glob.glob(os.path.join(Path(args.bids_dir), 'petprep_hmc_wf', '*', '*', '*', 'translation.png'))

    for idx, x in enumerate(mc_files):
        match_sub_id = re.search(r'sub-([A-Za-z0-9]+)_', mc_files[idx])
        sub_id = match_sub_id.group(1)
        
        match_ses_id = re.search(r'ses-([A-Za-z0-9]+)_', mc_files[idx])

        if match_ses_id:
            ses_id = match_ses_id.group(1)
        else:
            ses_id = None

        match_run_id = re.search(r'run-([A-Za-z0-9]+)_', mc_files[idx])

        if match_run_id:
            run_id = match_run_id.group(1)
        else:
            run_id = None

        match_file_prefix = re.search(r'_pet_file_(.*?)_pet', mc_files[idx])
        file_prefix = match_file_prefix.group(1)

        if ses_id is not None:
            sub_out_dir = Path(os.path.join(output_dir, 'sub-' + sub_id, 'ses-' + ses_id))
        else:
            sub_out_dir = Path(os.path.join(output_dir, 'sub-' + sub_id))

        os.makedirs(sub_out_dir, exist_ok=True)
        shutil.copyfile(mc_files[idx], os.path.join(sub_out_dir, f'{file_prefix}_desc-mc_pet.nii.gz'))
        shutil.copyfile(confound_files[idx], os.path.join(sub_out_dir, f'{file_prefix}_desc-confounds_timeseries.tsv'))
        shutil.copyfile(movement[idx], os.path.join(sub_out_dir, f'{file_prefix}_desc-movement.png'))
        shutil.copyfile(rotation[idx], os.path.join(sub_out_dir, f'{file_prefix}_desc-rotation.png'))
        shutil.copyfile(translation[idx], os.path.join(sub_out_dir, f'{file_prefix}_desc-translation.png'))

        if ses_id is not None and run_id is None:
            source_file = layout.get(suffix='pet', subject=sub_id, session=ses_id, extension=['.nii', '.nii.gz'], return_type='filename')[0]
        elif ses_id is not None and run_id is not None:
            source_file = layout.get(suffix='pet', subject=sub_id, session=ses_id, run=run_id, extension=['.nii', '.nii.gz'], return_type='filename')[0]
        elif ses_id is None and run_id is not None:
            source_file = layout.get(suffix='pet', subject=sub_id, run=run_id, extension=['.nii', '.nii.gz'], return_type='filename')[0]
        elif ses_id is None and run_id is None:
            source_file = layout.get(suffix='pet', subject=sub_id, extension=['.nii', '.nii.gz'], return_type='filename')[0]

        # create html report
        

        hmc_json = {
            "Description": "Motion-corrected PET file",
            "Sources": source_file,
            "ReferenceImage": "Robust template using mri_robust_register",
            "CostFunction": "ROB",
            "MCTreshold": f"{args.mc_thresh}",
            "MCFWHM": f"{args.mc_fwhm}",
            "MCStartTime": f"{args.mc_start_time}",
            "QC": "",
            "SoftwareName": "PETPrep HMC workflow",
            "SoftwareVersion": str(__version__),
            "CommandLine": " ".join(sys.argv)
            }

        json_object = json.dumps(hmc_json, indent=4)
        with open(os.path.join(sub_out_dir, f'{file_prefix}_desc-mc_pet.json'), "w") as outfile:
            outfile.write(json_object)

        # Plot with and without motion correction
        plot_mc_dynamic_pet(source_file, mc_files[idx], sub_out_dir, file_prefix)

    # Remove temp outputs
    shutil.rmtree(os.path.join(args.bids_dir, 'petprep_hmc_wf'))

    dataset_description_json = {
        "Name": "PETPrep HMC workflow",
        "DatasetType": "derivative",
        "BIDSVersion": "1.7.0",
        "GeneratedBy": [
            {
                "Name": "petprep_hmc",
                "Version": str(__version__),
            }
        ]
    }

    json_object = json.dumps(dataset_description_json, indent=4)
    with open(os.path.join(args.output_dir, 'dataset_description.json'), "w") as outfile:
        outfile.write(json_object)


def init_petprep_hmc_wf():
    from bids import BIDSLayout

    layout = BIDSLayout(args.bids_dir, validate=False)

    petprep_hmc_wf = Workflow(name='petprep_hmc_wf', base_dir=args.bids_dir)
    petprep_hmc_wf.config['execution']['remove_unnecessary_outputs'] = 'false'

    # Define the subjects to iterate over
    subject_list = layout.get(return_type='id', target='subject', suffix='pet')

    # Set up the main workflow to iterate over subjects
    for subject_id in subject_list:
        # For each subject, create a subject-specific workflow
        subject_wf = init_single_subject_wf(subject_id)
        petprep_hmc_wf.add_nodes([subject_wf])

    return petprep_hmc_wf


def init_single_subject_wf(subject_id):
    from bids import BIDSLayout

    layout = BIDSLayout(args.bids_dir, validate=False)

    # Create a new workflow for this specific subject
    subject_wf = Workflow(name=f'subject_{subject_id}_wf', base_dir='.')
    subject_wf.config['execution']['remove_unnecessary_outputs'] = 'false'

    subject_data = collect_data(layout,
                            participant_label=subject_id)[0]['pet']

    # This function will strip the extension(s) from a filename
    def strip_extensions(filename):
        while os.path.splitext(filename)[1]:
            filename = os.path.splitext(filename)[0]
        return filename

    # Use os.path.basename to get the last part of the path and then remove the extensions
    cleaned_subject_data = [strip_extensions(os.path.basename(path)) for path in subject_data]

    inputs = Node(IdentityInterface(fields=['pet_file']), name='inputs')
    inputs.iterables = ('pet_file', cleaned_subject_data)

    sessions = layout.get_sessions(subject=subject_id)

    templates = {'pet_file': 's*/pet/*{pet_file}.[n]*' if not sessions else 's*/s*/pet/*{pet_file}.[n]*',
                 'json_file': 's*/pet/*{pet_file}.json' if not sessions else 's*/s*/pet/*{pet_file}.json'}

    selectfiles = Node(SelectFiles(templates,
                                   base_directory=args.bids_dir),
                       name="select_files")

    # Define nodes for hmc workflow

    split_pet = Node(interface=fs.MRIConvert(split=True),
                     name="split_pet")

    smooth_frame = MapNode(interface=fsl.Smooth(fwhm=int(args.mc_fwhm)),
                           name="smooth_frame",
                           iterfield=['in_file'])

    thres_frame = MapNode(interface=fsl.maths.Threshold(thresh=int(args.mc_thresh),
                                                        use_robust_range=True),
                          name="thres_frame",
                          iterfield=['in_file'])

    estimate_motion = Node(interface=fs.RobustTemplate(auto_detect_sensitivity=True,
                                                       intensity_scaling=True,
                                                       average_metric='mean',
                                                       args=f'--cras --frobnorm-thresh {args.frobnorm_thresh}'),
                           name="estimate_motion", iterfield=['in_files'])

    correct_motion = MapNode(interface=fs.ApplyVolTransform(),
                             name="correct_motion",
                             iterfield=['source_file', 'reg_file', 'transformed_file'])

    if args.no_resample:
        correct_motion.inputs.no_resample = True

    concat_frames = Node(interface=fs.Concatenate(concatenated_file='mc.nii.gz'),
                         name="concat_frames")

    lta2xform = MapNode(interface=fs.utils.LTAConvert(),
                        name="lta2xform",
                        iterfield=['in_lta', 'out_fsl'])

    est_trans_rot = MapNode(interface=fsl.AvScale(all_param=True),
                            name="est_trans_rot",
                            iterfield=['mat_file'])

    est_min_frame = Node(Function(input_names=['json_file', 'mc_start_time'],
                                  output_names=['min_frame'],
                                  function=get_min_frame),
                         name="est_min_frame")

    est_min_frame.inputs.mc_start_time = int(args.mc_start_time)

    upd_frame_list = Node(Function(input_names=['in_file', 'min_frame'],
                                   output_names=['upd_list_frames'],
                                   function=update_list_frames),
                          name="upd_frame_list")

    upd_transform_list = Node(Function(input_names=['in_file', 'min_frame'],
                                       output_names=['upd_list_transforms'],
                                       function=update_list_transforms),
                              name="upd_transform_list")

    hmc_movement_output = Node(Function(input_names=['translations', 'rot_angles', 'rotation_translation_matrix','in_file'],
                                        output_names=['hmc_confounds'],
                                        function=combine_hmc_outputs),
                               name="hmc_movement_output")

    plot_motion = Node(Function(input_names=['in_file'],
                                function=plot_motion_outputs),
                       name="plot_motion")

    # Connect workflow - init_pet_hmc_wf
    subject_wf.connect([(inputs, selectfiles, [('pet_file', 'pet_file')]),
                        (selectfiles, split_pet, [('pet_file', 'in_file')]),
                        (selectfiles, est_min_frame, [('json_file', 'json_file')]),
                        (split_pet, smooth_frame, [('out_file', 'in_file')]),
                        (smooth_frame, thres_frame, [('smoothed_file', 'in_file')]),
                        (thres_frame, upd_frame_list, [('out_file', 'in_file')]),
                        (est_min_frame, upd_frame_list, [('min_frame', 'min_frame')]),
                        (upd_frame_list, estimate_motion, [('upd_list_frames', 'in_files')]),
                        (thres_frame, upd_transform_list, [('out_file', 'in_file')]),
                        (est_min_frame, upd_transform_list, [('min_frame', 'min_frame')]),
                        (upd_transform_list, estimate_motion, [('upd_list_transforms', 'transform_outputs')]),
                        (split_pet, correct_motion, [('out_file', 'source_file')]),
                        (estimate_motion, correct_motion, [('transform_outputs', 'reg_file')]),
                        (estimate_motion, correct_motion, [('out_file', 'target_file')]),
                        (split_pet, correct_motion, [(('out_file', add_mc_ext), 'transformed_file')]),
                        (correct_motion, concat_frames, [('transformed_file', 'in_files')]),
                        (estimate_motion, lta2xform, [('transform_outputs', 'in_lta')]),
                        (estimate_motion, lta2xform, [(('transform_outputs', lta2mat), 'out_fsl')]),
                        (lta2xform, est_trans_rot, [('out_fsl', 'mat_file')]),
                        (est_trans_rot, hmc_movement_output, [('translations', 'translations'), ('rot_angles', 'rot_angles'), ('rotation_translation_matrix', 'rotation_translation_matrix')]),
                        (upd_frame_list, hmc_movement_output, [('upd_list_frames', 'in_file')]),
                        (hmc_movement_output, plot_motion, [('hmc_confounds', 'in_file')])
                        ])
    return subject_wf


def display_motion_correction_html(file_prefix, sub_out_dir):
    """
    Load and display motion correction figures based on config settings.
    """

    report_config_path = op.join(os.path.dirname(petprep_hmc.__file__), 'docs/reports-spec.yml')
    report_config = load_config(report_config_path)

    # Start the HTML content with the basic HTML structure
    html_content = "<html><head><title>Motion Correction Report</title></head><body>"
    
    for reportlet in report_config['sections'][0]['reportlets']:
        desc = reportlet['bids']['desc']
        extension = reportlet['bids']['extension'][0]  # Assuming only one extension per type
        file_name = f"{file_prefix}_desc-{desc}{extension}"
        file_path = op.join(sub_out_dir, file_name)
        
        if op.exists(file_path):
            # Include title and description if available
            title = f"<h2>{reportlet.get('subtitle', 'Motion Correction Visual')}</h2>"
            caption = f"<p>{reportlet.get('caption', '')}</p>"
            
            # Generate HTML image tag or use plotly for GIFs if needed
            img_tag = f"<img src='{file_path}' style='width:100%; max-width:{reportlet['style']['max-width']}'>"
            
            # Combine elements
            html_content += title + caption + img_tag + "<br>"
        else:
            html_content += f"<p>File not found: {file_path}</p>"

    # Close the HTML tags
    html_content += "</body></html>"

    # Write the complete HTML content to the report file
    report_file_path = op.join(sub_out_dir, f"{file_prefix}_report.html")
    with open(report_file_path, "w") as report_file:
        report_file.write(html_content)


def load_config(filepath):
    """
    Load a YAML configuration file.
    :param filepath: str, path to the YAML file
    :return: dict, parsed YAML data
    """
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)


# HELPER FUNCTIONS
def update_list_frames(in_file, min_frame):
    """
    Function to update the list of frames to be used in the hmc workflow.

    Parameters
    ----------
    in_file : list of frames
    min_frame : minimum frame to use for the analysis (first frame after 2 min)

    Returns
    -------
    new_list : list of updated frames to be used in the hmc workflow

    """

    new_list = [in_file[min_frame]] * min_frame + in_file[min_frame:]
    return new_list


def update_list_transforms(in_file, min_frame):
    """
    Function to update the list of transforms to be used in the hmc workflow.

    Parameters
    ----------
    in_file : list of transforms
    min_frame : minimum frame to use for the analysis (first frame after 2 min)

    Returns
    -------
    lta_list : list of updated transforms to be used in the hmc workflow
    """

    new_list = [in_file[min_frame]] * min_frame + in_file[min_frame:]
    lta_list = [ext.replace('nii.gz', 'lta') for ext in new_list]
    return lta_list


def add_mc_ext(in_file):
    """ 
    Function to add the mc extension to the list of file names.

    Parameters
    ----------
    in_file : file name to be updated

    Returns
    -------
    mc_list : list of updated file names with mc extension
    """

    if len(in_file) > 1:
        mc_list = [ext.replace('.nii.gz', '_mc.nii') for ext in in_file]
    else:
        mc_list = in_file.replace('.nii.gz', '_mc.nii')
    return mc_list


def lta2mat(in_file):
    """
    Function to convert the lta file to the fsl format (.mat).

    Parameters
    ----------
    in_file : list of lta files to be converted

    Returns
    -------
    mat_list : list of mat files
    """

    mat_list = [ext.replace('.lta', '.mat') for ext in in_file]
    return mat_list


def get_min_frame(json_file, mc_start_time):
    """
    Function to extract the frame number after mc_start_time (default=120) seconds of mid frames of dynamic PET data to be used with motion correction

    Parameters
    ----------
    json_file : json file containing the frame length and duration of the dynamic PET data

    Returns
    -------
    min_frame : minimum frame to use for the motion correction (first frame after 2 min)
    """

    import numpy as np
    import json

    with open(json_file, 'r') as f:
        info = json.load(f)
        frames_duration = np.array(info['FrameDuration'], dtype=float)
        frames_start = np.pad(np.cumsum(frames_duration)[:-1], (1, 0))
        mid_frames = frames_start + frames_duration/2

        min_frame = next(x for x, val in enumerate(mid_frames)
                         if val > mc_start_time)
    return min_frame


def combine_hmc_outputs(translations, rot_angles, rotation_translation_matrix, in_file):
    """
    Function to combine the outputs of the hmc workflow.

    Parameters
    ----------
    translations : list of estimated translations across frames
    rot_angles : list of estimated rotational angles across frames
    rotation_translation_matrix : list of estimated rotation translation matrices across frames
    in_file : list of frames to be used in the hmc workflow

    Returns
    -------
    Output path to confounds file for head motion correction
    """

    import os
    import pandas as pd
    import numpy as np
    import nibabel as nib

    new_pth = os.getcwd()

    movement = []
    for idx, trans in enumerate(translations):

        img = nib.load(in_file[idx])
        vox_ind = np.asarray(np.nonzero(img.get_fdata()))
        pos_bef = np.concatenate((vox_ind, np.ones((1, len(vox_ind[0, :])))))
        pos_aft = rotation_translation_matrix[idx] @ pos_bef
        diff_pos = pos_bef-pos_aft

        max_x = abs(max(diff_pos[0, :], key=abs))
        max_y = abs(max(diff_pos[1, :], key=abs))
        max_z = abs(max(diff_pos[2, :], key=abs))
        overall = np.sqrt(diff_pos[0, :] ** 2 + diff_pos[1,:] ** 2 + diff_pos[2, :] ** 2)
        max_tot = np.max(overall)
        median_tot = np.median(overall)

        movement.append(np.concatenate((translations[idx],rot_angles[idx], [max_x, max_y, max_z, max_tot, median_tot])))

    confounds = pd.DataFrame(movement, columns=['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'max_x', 'max_y', 
                                                'max_z', 'max_tot', 'median_tot'])
    confounds.to_csv(os.path.join(new_pth, 'hmc_confounds.tsv'), sep='\t')

    return os.path.join(new_pth, 'hmc_confounds.tsv')


def plot_motion_outputs(in_file):
    """
    Function to plot estimated motion data

    Parameters
    ----------
    in_file : list of estimated motion data

    Returns
    -------
    Plots of estimated motion data
    """

    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    confounds = pd.read_csv(in_file, sep='\t')

    new_pth = os.getcwd()

    n_frames = len(confounds.index)

    plt.figure(figsize=(11,5))
    plt.plot(np.arange(0,n_frames), confounds['trans_x'], "-r", label='trans_x')
    plt.plot(np.arange(0,n_frames), confounds['trans_y'], "-g", label='trans_y')
    plt.plot(np.arange(0,n_frames), confounds['trans_z'], "-b", label='trans_z')
    plt.legend(loc="upper left")
    plt.ylabel('Translation [mm]')
    plt.xlabel('frame #')
    plt.grid(visible=True)
    plt.savefig(os.path.join(new_pth, 'translation.png'), format='png')
    plt.close()

    plt.figure(figsize=(11, 5))
    plt.plot(np.arange(0, n_frames), confounds['rot_x'], "-r", label='rot_x')
    plt.plot(np.arange(0, n_frames), confounds['rot_y'], "-g", label='rot_y')
    plt.plot(np.arange(0, n_frames), confounds['rot_z'], "-b", label='rot_z')
    plt.legend(loc="upper left")
    plt.ylabel('Rotation [degrees]')
    plt.xlabel('frame #')
    plt.grid(visible=True)
    plt.savefig(os.path.join(new_pth, 'rotation.png'), format='png')
    plt.close()

    plt.figure(figsize=(11, 5))
    plt.plot(np.arange(0, n_frames), confounds['max_x'], "--r", label='max_x')
    plt.plot(np.arange(0, n_frames), confounds['max_y'], "--g", label='max_y')
    plt.plot(np.arange(0, n_frames), confounds['max_z'], "--b", label='max_z')
    plt.plot(np.arange(0, n_frames), confounds['max_tot'], "-k", label='max_total')
    plt.plot(np.arange(0, n_frames), confounds['median_tot'], "-m", label='median_tot')
    plt.legend(loc="upper left")
    plt.ylabel('Movement [mm]')
    plt.xlabel('frame #')
    plt.grid(visible=True)
    plt.savefig(os.path.join(new_pth, 'movement.png'), format='png')
    plt.close()


def check_fsl_installed():
    try:
        fsl_home = os.environ['FSLDIR']
        if fsl_home:
            print("FSL is installed at:", fsl_home)
            return True
    except KeyError:
        print("FSL is not installed or FSLDIR environment variable is not set.")
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BIDS App for PETPrep head motion correction workflow')
    parser.add_argument('--bids_dir', required=True,  help='The directory with the input dataset '
                        'formatted according to the BIDS standard.')
    parser.add_argument('--output_dir', required=False, help='The directory where the output files '
                        'should be stored. If you are running group level analysis '
                        'this folder should be prepopulated with the results of the'
                        'participant level analysis.')
    parser.add_argument('--analysis_level', default='participant', help='Level of the analysis that will be performed. '
                        'Multiple participant level analyses can be run independently '
                        '(in parallel) using the same output_dir.',
                        choices=['participant', 'group'])
    parser.add_argument('--participant_label', help='The label(s) of the participant(s) that should be analyzed. The label '
                        'corresponds to sub-<participant_label> from the BIDS spec '
                        '(so it does not include "sub-"). If this parameter is not '
                        'provided all subjects should be analyzed. Multiple '
                        'participants can be specified with a space separated list.',
                        nargs="+", default=None)
    parser.add_argument('--mc_start_time', help='Start time for when to perform motion correction (subsequent frame will be chosen) in seconds', default=120)
    parser.add_argument('--mc_fwhm', help='FWHM for smoothing of frames prior to estimating motion', default=10)
    parser.add_argument('--mc_thresh', help='Threshold below the following percentage (0-100) of framewise ROBUST RANGE prior to estimating motion correction', default=20)
    parser.add_argument('--n_procs', help='Number of processors to use when running the workflow', default=2)
    parser.add_argument('--no_resample', help='Whether or not to resample the motion corrected PET data to lowest x/y/z dim in original data', action='store_true')
    parser.add_argument('--frobnorm_thresh', help='Threshold for the Frobenius norm of the robust registration', default=0.0001)
    parser.add_argument('--skip_bids_validator', help='Whether or not to perform BIDS dataset validation',
                        action='store_true')
    parser.add_argument('-v', '--version', action='version',
                        version='PETPrep BIDS-App version {}'.format(__version__))

    args = parser.parse_args()

    main(args)
