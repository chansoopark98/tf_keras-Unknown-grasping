import glob
import os
from imageio import imread, imsave, imwrite
from tqdm import tqdm
import shutil
from utils.dataset_processing import grasp, image

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

class JacquardDataset:
    def __init__(self, file_path):

        
        self.grasp_files = glob.glob(os.path.join(file_path+'/*/', '*', '*_grasps.txt'))
        self.grasp_files.sort()
        self.length = len(self.grasp_files)

        if self.length == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))

        self.depth_files = [f.replace('grasps.txt', 'perfect_depth.tiff') for f in self.grasp_files]
        self.rgb_files = [f.replace('perfect_depth.tiff', 'RGB.png') for f in self.depth_files]
        
# output_path = './cornell_output/'
output_path = './jacquard_output/'
rgb_path = output_path + 'rgb/'
depth_path = output_path + 'depth/'
box_path = output_path + 'box/'
os.makedirs(output_path, exist_ok=True)
os.makedirs(rgb_path, exist_ok=True)
os.makedirs(depth_path, exist_ok=True)
os.makedirs(box_path, exist_ok=True)

# dataset = CornellDataset(file_path='./datasets/')
dataset = JacquardDataset(file_path='./datasets/Jacquard/')
pbar = tqdm(range(dataset.length))
for i in pbar:
    rgb = imread(dataset.rgb_files[i])
    imsave(rgb_path + str(i) + '_rgb.png', rgb)
    depth = imread(dataset.depth_files[i])
    # imsave(depth_path + str(i) + '_depth.png', depth)
    imwrite(depth_path + str(i) + '_depth.tif', depth)
    bbox = dataset.grasp_files[i]
    shutil.copy(bbox, box_path + str(i) + '_box.txt')

print("Data is being compressed.....")
shutil.make_archive('data', 'zip', output_path)
print("Data compression complete")

