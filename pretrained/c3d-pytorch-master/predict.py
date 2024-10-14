""" How to use C3D network. """
import numpy as np

import torch
from torch.autograd import Variable

from os.path import join
from glob import glob

import skimage.io as io
from skimage.transform import resize

from C3D_model import C3D


def get_sport_clip(clip_name, verbose=True):
    """
    Loads a clip to be fed to C3D for classification.
    
    Parameters
    ----------
    clip_name: str
        The name of the clip (subfolder in 'data').
    verbose: bool
        If True, shows the unrolled clip (default is True).

    Returns
    -------
    Tensor
        A PyTorch batch (n, ch, fr, h, w).
    """

    # Load image file paths
    clip_paths = sorted(glob(join('data', clip_name, '*.png')))

    # Load and process each image
    clip = [resize(io.imread(frame), output_shape=(112, 200), preserve_range=True) for frame in clip_paths]
    
    # Convert list of images into a 4D numpy array (fr, h, w, ch)
    clip = np.array(clip)
    
    # Crop centrally (assuming the images have more width than height)
    clip = clip[:, :, 44:44+112, :]  # Now clip is (fr, h, w, ch)

    if verbose:
        # Display the clip
        clip_img = np.reshape(clip.transpose(1, 0, 2, 3), (112, 16 * 112, 3))
        io.imshow(clip_img.astype(np.uint8))
        io.show()

    # Rearrange dimensions and add a batch axis (batch, ch, fr, h, w)
    clip = clip.transpose(3, 0, 1, 2)  # ch, fr, h, w
    clip = np.expand_dims(clip, axis=0)  # batch axis
    clip = np.float32(clip)

    return torch.from_numpy(clip)
