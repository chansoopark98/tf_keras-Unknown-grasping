import glob
import os
from imageio import imread, imsave
from tqdm import tqdm


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
os.makedirs(output_path, exist_ok=True)
os.makedirs(rgb_path, exist_ok=True)
os.makedirs(depth_path, exist_ok=True)

dataset = CornellDataset(file_path='./datasets/')

pbar = tqdm(range(dataset.length))

for i in pbar:
    rgb = imread(dataset.rgb_files[i])
    imsave(rgb_path + str(i) + '_rgb.jpeg', rgb)
    depth = imread(dataset.depth_files[i])
    imsave(depth_path + str(i) + '_depth.tiff', depth)



