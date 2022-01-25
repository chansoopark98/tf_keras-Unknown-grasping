import os
import matplotlib.pyplot as plt
import numpy as np
import random
from torch import le
from skimage.filters import gaussian
import tensorflow as tf

from utils.data_generator_test import CornellDataset, JacquardDataset    
# from utils.dataset_processing import evaluation
import utils.utils.dataset_processing.evaluation as evaluation

def post_processing(q_img, cos_img, sin_img, width_img):
    q_img = np.squeeze(q_img)
    ang_img = np.squeeze(tf.math.atan2(sin_img, cos_img) / 2.0)
    width_img = np.squeeze(width_img) * 150.0

    # tfa.image.gaussian_filter2d()
    q_img = gaussian(q_img, 2.0, preserve_range=True)
    # q_img = tfa.image.gaussian_filter2d(image=q_img, sigma=2.0)

    ang_img = gaussian(ang_img, 2.0, preserve_range=True)
    # ang_img = tfa.image.gaussian_filter2d(image=ang_img, sigma=2.0)

    width_img = gaussian(width_img, 1.0, preserve_range=True)
    # width_img = tfa.image.gaussian_filter2d(image=width_img, sigma=1.0)

    return q_img, ang_img, width_img

# mode = 'jacquard'
mode = 'cornell'


cornell_path = './datasets/Cornell/'
    

jacquard_path = './datasets/Samples/'
    

output_size = 224
rows=3
cols=4

# dataset = CornellDataset(file_path=cornell_path, output_size=output_size)
dataset = JacquardDataset(file_path=jacquard_path, output_size=output_size)
for i in range(dataset.length):
    rotations = [0, np.pi / 2, 2 * np.pi / 2, 3 * np.pi / 2]
    rot = random.choice(rotations)
    zoom_factor = np.random.uniform(0.5, 1.0)

    # get bbox
    bbs = dataset.get_gtbb(idx=i, rot=rot, zoom=zoom_factor)
    # get depth
    depth_img = dataset.get_depth(idx=i, rot=rot, zoom=zoom_factor)
    # get img
    rgb_img = dataset.get_rgb(idx=i, rot=rot, zoom=zoom_factor)
    
    pos_img, ang_img, width_img = bbs.draw((output_size, output_size))
    width_img = np.clip(width_img, 0.0, output_size / 2) / (output_size / 2)

    cos = np.cos(2 * ang_img)
    sin = np.sin(2 * ang_img)


    fig = plt.figure()

    ax0 = fig.add_subplot(rows, cols, 1)
    ax0.imshow(pos_img)
    ax0.set_title('pos_img')
    ax0.axis("off")

    ax1 = fig.add_subplot(rows, cols, 2)
    ax1.imshow(cos)
    ax1.set_title('cos')
    ax1.axis("off")

    ax2 = fig.add_subplot(rows, cols, 3)
    ax2.imshow(sin)
    ax2.set_title('sin')
    ax2.axis("off")

    ax3 = fig.add_subplot(rows, cols, 4)
    ax3.imshow(width_img)
    ax3.set_title('width_img')
    ax3.axis("off")

    ax3 = fig.add_subplot(rows, cols, 5)
    ax3.imshow(rgb_img)
    ax3.set_title('rgb_img')
    ax3.axis("off")

    ax3 = fig.add_subplot(rows, cols, 6)
    ax3.imshow(depth_img, cmap='gray')
    ax3.set_title('depth_img')
    ax3.axis("off")

    q_img, ang_img, width_img = post_processing(q_img=pos_img,
                                            cos_img=cos,
                                            sin_img=sin,
                                            width_img=width_img)


    ax3 = fig.add_subplot(rows, cols, 7)
    ax3.imshow(q_img)
    ax3.set_title('q_img')
    ax3.axis("off")

    ax3 = fig.add_subplot(rows, cols, 8)
    ax3.imshow(ang_img)
    ax3.set_title('ang_img')
    ax3.axis("off")

    ax3 = fig.add_subplot(rows, cols, 9)
    ax3.imshow(width_img)
    ax3.set_title('width_img')
    ax3.axis("off")

    s = evaluation.calculate_iou_match(grasp_q = q_img,
                            grasp_angle = ang_img,
                            ground_truth_bbs = bbs,
                            no_grasps = 1,
                            grasp_width = width_img,
                            threshold=0.25)

    print('iou results', s)

    plt.show()
