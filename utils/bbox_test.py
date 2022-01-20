import glob
import os
from imageio import imread, imsave, imwrite
from tqdm import tqdm
import shutil
import numpy as np
import matplotlib.pyplot as plt
from utils.dataset_processing import grasp, image
# from utils.dataset_processing import grasp, image

class CornellDataset:
    def __init__(self, file_path):
        """
        :param file_path: Cornell Dataset directory
        """
        self.grasp_files = glob.glob(os.path.join(file_path, '*', 'pcd*cpos.txt'))
        self.grasp_files.sort()
        self.length = len(self.grasp_files)

        self.depth_files = [f.replace('cpos.txt', 'd.tiff') for f in self.grasp_files] 
        self.rgb_files = [f.replace('d.tiff', 'r.png') for f in self.depth_files]
        
output_path = './cornell_output/'
rgb_path = output_path + 'rgb/'
depth_path = output_path + 'depth/'
box_path = output_path + 'box/'
os.makedirs(output_path, exist_ok=True)
os.makedirs(rgb_path, exist_ok=True)
os.makedirs(depth_path, exist_ok=True)
os.makedirs(box_path, exist_ok=True)

output_size = 224
rot = 0
dataset = CornellDataset(file_path='./datasets/')
pbar = tqdm(range(dataset.length))
for i in pbar:
    bbox = dataset.grasp_files[i]
    gtbbs = grasp.GraspRectangles.load_from_cornell_file(bbox)
    center = gtbbs.center
    left = max(0, min(center[1] - output_size // 2, 640 - output_size))
    top = max(0, min(center[0] - output_size // 2, 480 - output_size))
    #return center, left, top
    gtbbs.rotate(rot, center)
    gtbbs.offset((-top, -left))
    # gtbbs.zoom(zoom, (output_size // 2, output_size // 2))
    a, b, c = gtbbs.draw((output_size, output_size))
    
    plt.imshow(a)
    plt.show()
    
    plt.imshow(b)
    plt.show()

    plt.imshow(c)
    plt.show()
    