import random

import numpy as np
import torch
import torch.utils.data

import cv2

class GraspDatasetBase(torch.utils.data.Dataset):
    """
    An abstract dataset for training networks in a common format.
    """

    def __init__(self, output_size=224, include_depth=True, include_rgb=False, random_rotate=False,
                 random_zoom=False, input_only=False, dataset_id=0, augmentation=False):
        """
        :param output_size: Image output size in pixels (square)
        :param include_depth: Whether depth image is included
        :param include_rgb: Whether RGB image is included
        :param random_rotate: Whether random rotations are applied
        :param random_zoom: Whether random zooms are applied
        :param input_only: Whether to return only the network input (no labels)
        """
        self.output_size = output_size
        self.random_rotate = random_rotate
        self.random_zoom = random_zoom
        self.input_only = input_only
        self.include_depth = include_depth
        self.include_rgb = include_rgb

        self.grasp_files = []

        self.dataset_id = dataset_id
        self.augmentation = augmentation

        if include_depth is False and include_rgb is False:
            raise ValueError('At least one of Depth or RGB must be specified.')

    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_depth(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_rgb(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def __getitem__(self, idx):
        if self.random_rotate:
            # rotations = [0, np.pi / 2, 2 * np.pi / 2, 3 * np.pi / 2]
            if self.__module__== 'utils.data.plaif_data':
                rotations = list(np.concatenate((np.arange(-np.pi/9, np.pi/9, 0.01),
                    np.arange(8*np.pi/9, 10*np.pi/9, 0.01))))
            else:
                rotations = list(np.arange(0, 2*np.pi, 0.01))
            rot = random.choice(rotations)
        else:
            rot = 0.0

        if self.random_zoom:
            if self.__module__== 'utils.data.plaif_data':
                zoom_factor = np.random.uniform(0.95, 1.0)
            else:
                zoom_factor = np.random.uniform(0.5, 1.0)
        else:
            zoom_factor = 1.0

        # Load the depth image
        if self.include_depth:
            depth_img = self.get_depth(idx, rot, zoom_factor)

        # Load the RGB image
        if self.include_rgb:
            rgb_img = self.get_rgb(idx, rot, zoom_factor, augment=self.augmentation)

        # Load the grasps
        bbs = self.get_gtbb(idx, rot, zoom_factor)

        # color = rgb_img.transpose((1,2,0))
        # color = color * 255
        # color = color - color.min()
        # color = color.astype(np.uint8)
        # for gr in bbs.grs:
        #     igr = gr.points.astype(np.int)
        #     cv2.line(color, (igr[0,1],igr[0,0]), (igr[1,1],igr[1,0]), (255,0,0),lineType=cv2.LINE_AA)
        #     cv2.line(color, (igr[1,1],igr[1,0]), (igr[2,1],igr[2,0]), (0,0,255),lineType=cv2.LINE_AA)
        #     cv2.line(color, (igr[2,1],igr[2,0]), (igr[3,1],igr[3,0]), (255,0,0),lineType=cv2.LINE_AA)
        #     cv2.line(color, (igr[3,1],igr[3,0]), (igr[0,1],igr[0,0]), (0,0,255),lineType=cv2.LINE_AA)

        # cv2.imwrite("/root/workspace/robotic-grasping/utils/data/gt.png", color)
        pos_img, ang_img, width_img = bbs.draw((self.output_size, self.output_size))
        width_img = np.clip(width_img, 0.0, self.output_size / 2) / (self.output_size / 2)

        if self.include_depth and self.include_rgb:
            x = self.numpy_to_torch(
                np.concatenate(
                    (np.expand_dims(depth_img, 0),
                     rgb_img),
                    0
                )
            )
        elif self.include_depth:
            x = self.numpy_to_torch(depth_img)
        elif self.include_rgb:
            x = self.numpy_to_torch(rgb_img)

        pos = self.numpy_to_torch(pos_img)
        cos = self.numpy_to_torch(np.cos(2 * ang_img))
        sin = self.numpy_to_torch(np.sin(2 * ang_img))
        width = self.numpy_to_torch(width_img)

        return x, (pos, cos, sin, width), idx, rot, zoom_factor, self.dataset_id

    def __len__(self):
        return len(self.grasp_files)