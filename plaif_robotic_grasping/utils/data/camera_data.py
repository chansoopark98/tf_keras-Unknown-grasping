import numpy as np
import torch

import os
import sys
workspace_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
sys.path.append(workspace_dir)

from dataset_processing import image


class CameraData:
    """
    Dataset wrapper for the camera data.
    """
    def __init__(self,
                 width=1280,
                 height=720,
                 output_size=672,
                 include_depth=True,
                 include_rgb=True
                 ):
        """
        :param output_size: Image output size in pixels (square)
        :param include_depth: Whether depth image is included
        :param include_rgb: Whether RGB image is included
        """
        self.output_size = output_size
        self.include_depth = include_depth
        self.include_rgb = include_rgb

        if include_depth is False and include_rgb is False:
            raise ValueError('At least one of Depth or RGB must be specified.')

        left = (width - output_size) // 2
        top = (height - output_size) // 2
        right = (width + output_size) // 2
        bottom = (height + output_size) // 2

        self.bottom_right = (bottom, right)
        self.top_left = (top, left)
        self.zoom = 1.0

    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

    def set_crop_attrs(self, src_width, src_height, roi_xs, roi_ys, roi_xe, roi_ye):
        height = roi_ye - roi_ys
        width = roi_xe - roi_xs
        center = ((roi_ye + roi_ys) // 2, (roi_xe + roi_xs) // 2)
        size = max(height, width)
        half = size // 2

        left = center[1] - half
        right = center[1] + half
        top = center[0] - half
        bottom = center[0] + half

        if left < 0:
            offset = -left
            right += offset
            left = 0
        if right > src_width - 1:
            offset =  src_width - 1 - right
            left += offset
            right = src_width - 1
        if top < 0:
            offset = -top
            bottom += offset
            top = 0
        if bottom > src_height - 1:
            offset = src_height - 1 - bottom
            top += offset
            bottom = src_height - 1
        
        self.bottom_right = (bottom, right)
        self.top_left = (top, left)
        self.zoom = (bottom - top) / self.output_size
        
    def get_depth(self, img):
        depth_img = image.DepthImage(img)
        depth_img.crop(bottom_right=self.bottom_right, top_left=self.top_left)
        depth_img.normalise()
        if self.zoom != 1.0:
            depth_img.resize((self.output_size, self.output_size))
        depth_img.img = depth_img.img.transpose((2, 0, 1))
        return depth_img.img

    def get_rgb(self, img, norm=True):
        rgb_img = image.Image(img)
        rgb_img.crop(bottom_right=self.bottom_right, top_left=self.top_left)
        if self.zoom != 1.0:
            rgb_img.resize((self.output_size, self.output_size))
        if norm:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img

    def get_data(self, rgb=None, depth=None):
        depth_img = None
        rgb_img = None
        # Load the depth image
        if self.include_depth:
            depth_img = self.get_depth(img=depth)

        # Load the RGB image
        if self.include_rgb:
            rgb_img = self.get_rgb(img=rgb)

        if self.include_depth and self.include_rgb:
            x = self.numpy_to_torch(
                    np.concatenate(
                        (np.expand_dims(depth_img, 0),
                         np.expand_dims(rgb_img, 0)),
                        1
                    )
                )
        elif self.include_depth:
            x = self.numpy_to_torch(depth_img)
        elif self.include_rgb:
            x = self.numpy_to_torch(np.expand_dims(rgb_img, 0))

        return x, depth_img, rgb_img
    
        
    def get_mask(self, img):
        mask_img = image.MaskImage(img)
        mask_img.resize((mask_img.shape[0], self.output_size, self.output_size))
        return mask_img.img