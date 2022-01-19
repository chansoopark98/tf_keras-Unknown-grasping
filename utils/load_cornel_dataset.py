import glob
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from imageio import imread
import numpy as np


class CornellDataset:
    def __init__(self, file_path, ds_rotate=0, **kwargs):
        """
        :param file_path: Cornell Dataset directory.
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """

        self.grasp_files = glob.glob(os.path.join(file_path, '*', 'pcd*cpos.txt'))
        self.grasp_files.sort()
        self.length = len(self.grasp_files)

        self.depth_files = [f.replace('cpos.txt', 'd.tiff') for f in self.grasp_files] 
        self.rgb_files = [f.replace('d.tiff', 'r.png') for f in self.depth_files]

dataset = CornellDataset(file_path='./datasets/')
len = dataset.length
print(len)
depth_test = imread(dataset.depth_files[0])


