import torch
import numpy as np
from skimage.filters import gaussian


# def post_process_output(q_img, cos_img, sin_img, width_img):
#     """
#     Post-process the raw output of the network, convert to numpy arrays, apply filtering.
#     :param q_img: Q output of network (as torch Tensors)
#     :param cos_img: cos output of network
#     :param sin_img: sin output of network
#     :param width_img: Width output of network
#     :return: Filtered Q output, Filtered Angle output, Filtered Width output
#     """
#     q_img = q_img.cpu().numpy().squeeze()
#     ang_img = (torch.atan2(sin_img, cos_img) / 2.0).cpu().numpy().squeeze()
#     width_img = width_img.cpu().numpy().squeeze() * 336.0

#     q_img = gaussian(q_img, 2.0, preserve_range=True)
#     ang_img = gaussian(ang_img, 2.0, preserve_range=True)
#     width_img = gaussian(width_img, 1.0, preserve_range=True)

#     return q_img, ang_img, width_img

def post_process_output(q_img, cos_img, sin_img, width_img):
    """
    Post-process the raw output of the network, convert to numpy arrays, apply filtering.
    :param q_img: Q output of network (as torch Tensors)
    :param cos_img: cos output of network
    :param sin_img: sin output of network
    :param width_img: Width output of network
    :return: Filtered Q output, Filtered Angle output, Filtered Width output
    """
    q_img = q_img.cpu().numpy().squeeze()
    sin_img = sin_img.cpu().numpy().squeeze()
    cos_img = cos_img.cpu().numpy().squeeze()
    width_img = width_img.cpu().numpy().squeeze() * 336.0

    q_img = gaussian(q_img, 2.0, preserve_range=True)
    ang_img = np.arctan2(sin_img, cos_img) / 2.0
    ang_img_180 = np.where(ang_img<0, ang_img+np.pi, ang_img)
    ang_img_180 = gaussian(ang_img_180, 2.0, preserve_range=True)
    ang_img_90 = np.where(np.abs(ang_img)>1.5, ang_img_180-np.pi, ang_img)
    ang_img = ang_img_90
    width_img = gaussian(width_img, 1.0, preserve_range=True)

    return q_img, ang_img, width_img