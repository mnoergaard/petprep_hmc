import argparse 
import os
import sys
from bids import BIDSLayout
import pandas as pd
import nipype.interfaces.freesurfer as fs
import nipype.interfaces.fsl as fsl
from nipype.interfaces.utility import IdentityInterface
from nipype.pipeline import Workflow
from nipype.interfaces.io import DataSink
from nipype import Node, Function, MapNode
from nipype.interfaces.io import SelectFiles
from pathlib import Path
import glob
import re
import shutil
import json
from niworkflows.utils.misc import check_valid_fs_license
from petprep_hmc.utils import plot_mc_dynamic_pet

__version__ = open(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'version')).read()

def main(args): 
    
    if os.path.exists(args.bids_dir):
        if not args.skip_bids_validator:
            layout = BIDSLayout(args.bids_dir, validate=True)
        else:
            layout = BIDSLayout(args.bids_dir, validate=False)
    else:
        raise Exception('BIDS directory does not exist')
    
    if check_valid_fs_license() is not True:
        raise Exception('You need a valid FreeSurfer license to proceed!')
    
    if check_fsl_installed() is not True:
        raise Exception('FSL is not installed or sourced')
    
    # Get all PET files
    if args.participant_label is None:
        args.participant_label = layout.get(suffix='pet', target='subject', return_type='id')        
    
     # clean up and create derivatives directories
    if args.output_dir is None:
        output_dir = os.path.join(args.bids_dir,'derivatives','petprep_hmc')
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)

    infosource = Node(IdentityInterface(
                        fields = ['subject_id','session_id']),
                        name = "infosource")
    
    sessions = layout.get_sessions()
    if sessions:
        infosource.iterables = [('subject_id', args.participant_label),
                                ('session_id', sessions)]
    else:
        infosource.iterables = [('subject_id', args.participant_label)]

    templates = {'pet_file': 'sub-{subject_id}/pet/*_pet.[n]*' if not sessions else 'sub-{subject_id}/ses-{session_id}/pet/*_pet.[n]*',
                'json_file': 'sub-{subject_id}/pet/*_pet.json' if not sessions else 'sub-{subject_id}/ses-{session_id}/pet/*_pet.json'}
           
    selectfiles = Node(SelectFiles(templates, 
                               base_directory = args.bids_dir), 
                               name = "select_files")

    datasink = Node(DataSink(base_directory = os.path.join(args.bids_dir, 'derivatives')), 
                         name = "datasink")

    substitutions = [('_subject_id', 'sub'), ('_session_id_', 'ses')]
    subjFolders = [('sub-%s' % (sub), 'sub-%s' % (sub))
               for sub in layout.get_subjects()] if not sessions else [('sub-%s_ses-%s' % (sub, ses), 'sub-%s/ses-%s' % (sub, ses))
               for ses in layout.get_sessions()
               for sub in layout.get_subjects()]

    substitutions.extend(subjFolders)
    datasink.inputs.substitutions = substitutions

    # Define nodes for hmc workflow

    split_pet = Node(interface = fs.MRIConvert(split = True), 
                     name = "split_pet")
    
    smooth_frame = MapNode(interface=fsl.Smooth(fwhm=int(args.mc_fwhm)), 
                           name="smooth_frame", 
                           iterfield=['in_file'])
    
    thres_frame = MapNode(interface = fsl.maths.Threshold(thresh = int(args.mc_thresh), use_robust_range = True),
                          name = "thres_frame", 
                          iterfield = ['in_file'])
    
    estimate_motion = Node(interface = fs.RobustTemplate(auto_detect_sensitivity = True,
                                            intensity_scaling = True,
                                            average_metric = 'mean',
                                            args = '--cras'),
                           name="estimate_motion", iterfield=['in_files'])
    
    correct_motion = MapNode(interface = fs.ApplyVolTransform(), 
                             name = "correct_motion", 
                             iterfield = ['source_file', 'reg_file', 'transformed_file'])
    
    if args.no_resample:
        correct_motion.inputs.no_resample = True
    
    concat_frames = Node(interface = fs.Concatenate(concatenated_file = 'mc.nii.gz'), 
                         name = "concat_frames")
    
    lta2xform = MapNode(interface = fs.utils.LTAConvert(), 
                        name = "lta2xform", 
                        iterfield = ['in_lta', 'out_fsl'])
    
    est_trans_rot = MapNode(interface = fsl.AvScale(all_param = True), 
                            name = "est_trans_rot", 
                            iterfield = ['mat_file'])
    
    est_min_frame = Node(Function(input_names = ['json_file', 'mc_start_time'],
                                  output_names = ['min_frame'],
                                  function = get_min_frame),
                         name = "est_min_frame",)
    
    est_min_frame.inputs.mc_start_time = int(args.mc_start_time)

    upd_frame_list = Node(Function(input_names = ['in_file','min_frame'],
                                   output_names = ['upd_list_frames'],
                                   function = update_list_frames),
                          name = "upd_frame_list")
    
    upd_transform_list = Node(Function(input_names = ['in_file','min_frame'],
                                       output_names = ['upd_list_transforms'],
                                       function = update_list_transforms),
                          name = "upd_transform_list")
    
    hmc_movement_output = Node(Function(input_names = ['translations', 'rot_angles', 'rotation_translation_matrix','in_file'],
                                           output_names = ['hmc_confounds'],
                                           function = combine_hmc_outputs),
                               name = "hmc_movement_output")
    
    plot_motion = Node(Function(input_names = ['in_file'],
                                           function = plot_motion_outputs),
                               name = "plot_motion")

    # Connect workflow - init_pet_hmc_wf
    workflow = Workflow(name = "petprep_hmc_wf", base_dir=args.bids_dir)
    workflow.config['execution']['remove_unnecessary_outputs'] = 'false'
    workflow.connect([(infosource, selectfiles, [('subject_id', 'subject_id'),('session_id', 'session_id')]), 
                         (selectfiles, split_pet, [('pet_file', 'in_file')]),
                         (selectfiles, est_min_frame, [('json_file', 'json_file')]),
                         (split_pet,smooth_frame,[('out_file', 'in_file')]),
                         (smooth_frame,thres_frame,[('smoothed_file', 'in_file')]),
                         (thres_frame,upd_frame_list,[('out_file', 'in_file')]),
                         (est_min_frame,upd_frame_list,[('min_frame', 'min_frame')]),
                         (upd_frame_list,estimate_motion,[('upd_list_frames', 'in_files')]),
                         (thres_frame,upd_transform_list,[('out_file', 'in_file')]),
                         (est_min_frame,upd_transform_list,[('min_frame', 'min_frame')]),
                         (upd_transform_list,estimate_motion,[('upd_list_transforms', 'transform_outputs')]),
                         (split_pet,correct_motion,[('out_file', 'source_file')]),
                         (estimate_motion,correct_motion,[('transform_outputs', 'reg_file')]),
                         (estimate_motion,correct_motion,[('out_file', 'target_file')]),
                         (split_pet,correct_motion,[(('out_file', add_mc_ext), 'transformed_file')]),
                         (correct_motion,concat_frames,[('transformed_file', 'in_files')]),
                         (estimate_motion,lta2xform,[('transform_outputs', 'in_lta')]),
                         (estimate_motion,lta2xform,[(('transform_outputs', lta2mat), 'out_fsl')]),
                         (lta2xform,est_trans_rot,[('out_fsl', 'mat_file')]),
                         (est_trans_rot,hmc_movement_output,[('translations', 'translations'),('rot_angles', 'rot_angles'),('rotation_translation_matrix','rotation_translation_matrix')]),
                         (upd_frame_list,hmc_movement_output,[('upd_list_frames', 'in_file')]),
                         (hmc_movement_output,plot_motion,[('hmc_confounds','in_file')])
                         ])
    wf = workflow.run(plugin='MultiProc', plugin_args={'n_procs' : int(args.n_procs)})
    
    # loop through directories and store according to BIDS
    mc_files = glob.glob(os.path.join(Path(args.bids_dir),'petprep_hmc_wf','*','*','mc.nii.gz'))
    confound_files = glob.glob(os.path.join(Path(args.bids_dir),'petprep_hmc_wf','*','*','hmc_confounds.tsv'))
    movement = glob.glob(os.path.join(Path(args.bids_dir),'petprep_hmc_wf','*','*','movement.png'))
    rotation = glob.glob(os.path.join(Path(args.bids_dir),'petprep_hmc_wf','*','*','rotation.png'))
    translation = glob.glob(os.path.join(Path(args.bids_dir),'petprep_hmc_wf','*','*','translation.png'))
    
    for idx, x in enumerate(mc_files):
        sub_id = re.findall('subject_id_(.*)/concat', mc_files[idx])[0]
        
        if sessions:
            sess_id = re.findall('session_id_(.*)_subject_id', mc_files[idx])[0]
            sub_out_dir = Path(os.path.join(output_dir, 'sub-' + sub_id, 'ses-' + sess_id))
            file_prefix = f'sub-{sub_id}_ses-{sess_id}'
        else:
            sub_out_dir = Path(os.path.join(output_dir, 'sub-' + sub_id))
            file_prefix = f'sub-{sub_id}'
        
        os.makedirs(sub_out_dir, exist_ok=True)
        shutil.copyfile(mc_files[idx], os.path.join(sub_out_dir, f'{file_prefix}_desc-mc_pet.nii.gz'))
        shutil.copyfile(confound_files[idx], os.path.join(sub_out_dir, f'{file_prefix}_desc-confounds_timeseries.tsv'))
        shutil.copyfile(movement[idx], os.path.join(sub_out_dir, f'{file_prefix}_movement.png'))
        shutil.copyfile(rotation[idx], os.path.join(sub_out_dir, f'{file_prefix}_rotation.png'))
        shutil.copyfile(translation[idx], os.path.join(sub_out_dir, f'{file_prefix}_translation.png'))
        
        if sessions:
            source_file = layout.get(suffix='pet', subject=sub_id, session=sess_id, extension=['.nii', '.nii.gz'], return_type='filename')[0]
        else:
            source_file = layout.get(suffix='pet', subject=sub_id, extension=['.nii', '.nii.gz'], return_type='filename')[0]

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
        
     # remove temp outputs
    shutil.rmtree(os.path.join(args.bids_dir, 'petprep_hmc_wf'))

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
    lta_list = [ext.replace('nii.gz','lta') for ext in new_list]  
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
        mc_list = [ext.replace('.nii.gz','_mc.nii') for ext in in_file] # list comphrehension
    else:
        mc_list = in_file.replace('.nii.gz','_mc.nii')
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
    
    mat_list = [ext.replace('.lta','.mat') for ext in in_file]
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
    
    import os
    from os.path import join, isfile
    import numpy as np
    import json

    with open(json_file, 'r') as f:
        info = json.load(f)
        frames_duration = np.array(info['FrameDuration'], dtype=float)
        frames_start =np.pad(np.cumsum(frames_duration)[:-1],(1,0))
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
        pos_bef = np.concatenate((vox_ind,np.ones((1,len(vox_ind[0,:])))))
        pos_aft = rotation_translation_matrix[idx] @ pos_bef
        diff_pos = pos_bef-pos_aft
        
        max_x = abs(max(diff_pos[0,:], key=abs))
        max_y = abs(max(diff_pos[1,:], key=abs))
        max_z = abs(max(diff_pos[2,:], key=abs))
        overall = np.sqrt(diff_pos[0,:] ** 2 + diff_pos[1,:] ** 2 + diff_pos[2,:] ** 2)
        max_tot = np.max(overall)
        median_tot = np.median(overall)
        
        movement.append(np.concatenate((translations[idx],rot_angles[idx], [max_x, max_y, max_z, max_tot, median_tot])))
        
    confounds = pd.DataFrame(movement, columns=['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'max_x', 'max_y', 
                                                'max_z', 'max_tot', 'median_tot'])
    confounds.to_csv(os.path.join(new_pth,'hmc_confounds.tsv'), sep='\t')
    # np.savetxt(os.path.join(new_pth,'hmc_confounds.tsv'), movement, fmt='%10.5f', delimiter='\t', header='trans_x    trans_y    trans_z    rot_x    rot_y    rot_z')
    
    return os.path.join(new_pth,'hmc_confounds.tsv')

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
    import nibabel as nib
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
    plt.savefig(os.path.join(new_pth,'translation.png'), format='png')
    plt.close()
    
    plt.figure(figsize=(11,5))
    plt.plot(np.arange(0,n_frames), confounds['rot_x'], "-r", label='rot_x')
    plt.plot(np.arange(0,n_frames), confounds['rot_y'], "-g", label='rot_y')
    plt.plot(np.arange(0,n_frames), confounds['rot_z'], "-b", label='rot_z')
    plt.legend(loc="upper left")
    plt.ylabel('Rotation [degrees]')
    plt.xlabel('frame #')
    plt.grid(visible=True)
    plt.savefig(os.path.join(new_pth,'rotation.png'), format='png')
    plt.close()
    
    plt.figure(figsize=(11,5))
    plt.plot(np.arange(0,n_frames), confounds['max_x'], "--r", label='max_x')
    plt.plot(np.arange(0,n_frames), confounds['max_y'], "--g", label='max_y')
    plt.plot(np.arange(0,n_frames), confounds['max_z'], "--b", label='max_z')
    plt.plot(np.arange(0,n_frames), confounds['max_tot'], "-k", label='max_total')
    plt.plot(np.arange(0,n_frames), confounds['median_tot'], "-m", label='median_tot')
    plt.legend(loc="upper left")
    plt.ylabel('Movement [mm]')
    plt.xlabel('frame #')
    plt.grid(visible=True)
    plt.savefig(os.path.join(new_pth,'movement.png'), format='png')
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
    parser.add_argument('--skip_bids_validator', help='Whether or not to perform BIDS dataset validation',
                   action='store_true')
    parser.add_argument('-v', '--version', action='version',
                    version='PETPrep BIDS-App version {}'.format(__version__))
    
    args = parser.parse_args() 
    
    main(args)
