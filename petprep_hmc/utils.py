from nilearn.plotting import plot_epi
import nibabel as nib
from nilearn import image
from nilearn.plotting.find_cuts import find_xyz_cut_coords
import numpy as np
from niworkflows.interfaces.bids import BIDSInfo
import imageio
import os

def plot_mc_dynamic_pet(pet_orig, pet_mc, sub_out_dir, file_prefix):
    nib_orig = nib.load(pet_orig)
    nib_mc = nib.load(pet_mc)
    shape_orig = nib_orig.shape
    shape_mc = nib_mc.shape
    t_orig = shape_orig[-1]
    t_mc = shape_mc[-1]

    mid_orig = image.index_img(pet_orig, t_orig//2)
    mid_mc = image.index_img(pet_mc, t_mc//2)

    vmax_orig = np.percentile(mid_orig.get_fdata().flatten(), 99.9) 
    vmin_orig = np.percentile(mid_orig.get_fdata().flatten(), 80)
    vmax_mc = np.percentile(mid_mc.get_fdata().flatten(), 99.9)
    vmin_mc = np.percentile(mid_mc.get_fdata().flatten(), 80)

    x_orig, y_orig, z_orig = find_xyz_cut_coords(mid_orig)
    x_mc, y_mc, z_mc = find_xyz_cut_coords(mid_mc)

    orig_images = []
    for idx, img in enumerate(image.iter_img(pet_orig)):
        # img is now an in-memory 3D img
        plot_epi(
            img, colorbar=True, display_mode='ortho', title=f"Frame #{idx}", cut_coords=(x_orig,y_orig,z_orig), vmin=vmin_orig, vmax=vmax_orig, output_file=f"orig_{idx}.png"
        )
        orig_images.append(imageio.imread(f"orig_{idx}.png"))
        os.remove(f"orig_{idx}.png")
    
    # Write the images to a GIF file
    imageio.mimsave(os.path.join(sub_out_dir, f'{file_prefix}_desc-without_motion_correction.gif'), orig_images, duration=1, loop=0)

    mc_images = []
    for idx, img in enumerate(image.iter_img(pet_mc)):
        # img is now an in-memory 3D img
        plot_epi(
            img, colorbar=True, display_mode='ortho', title=f"Frame #{idx}", cut_coords=(x_mc,y_mc,z_mc), vmin=vmin_mc, vmax=vmax_mc, output_file=f"mc_{idx}.png"
        )
        mc_images.append(imageio.imread(f"mc_{idx}.png"))
        os.remove(f"mc_{idx}.png")
    
    # Write the images to a GIF file
    imageio.mimsave(os.path.join(sub_out_dir, f'{file_prefix}_desc-with_motion_correction.gif'), mc_images, duration=1, loop=0)

    