import glob
import os
from imageio import imread, imsave, imwrite
from tqdm import tqdm
import shutil
import numpy as np
import matplotlib.pyplot as plt
from utils.dataset_processing import grasp, image
import tensorflow as tf
# from utils.dataset_processing import grasp, image

class CornellDataset:
    def __init__(self, file_path):
        """-
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
rows=2
cols=4
dataset = CornellDataset(file_path='./datasets/')
pbar = tqdm(range(dataset.length))
for i in pbar:
    # bbox
    bbox = dataset.grasp_files[i]
    gtbbs = grasp.GraspRectangles.load_from_cornell_file(bbox)
    a = gtbbs.to_array()
    a = np.array(a)
    a = tf.convert_to_tensor(a)
    gtbbs = grasp.GraspRectangles.load_from_tensor(a)


    # GET center position
    center = gtbbs.center
    left = max(0, min(center[1] - output_size // 2, 640 - output_size))
    top = max(0, min(center[0] - output_size // 2, 480 - output_size))
    # get bbox
    gtbbs.rotate(rot, center)
    gtbbs.offset((-top, -left))
    # gtbbs.zoom(zoom, (output_size // 2, output_size // 2)) TODO
    pos_img, ang_img, width_img = gtbbs.draw((output_size, output_size))
    width_img = np.clip(width_img, 0.0, output_size /2 ) / (output_size / 2)
    cos = np.cos(2 * ang_img)
    sin = np.sin(2 * ang_img)


    # RGB
    rgb = dataset.rgb_files[i]
    img = image.Image.from_file(rgb)
    img = tf.convert_to_tensor(img)
    img = image.Image.from_tensor(img)
    img.rotate(rot, center)
    img.crop((top, left), (min(480, top + output_size), min(640, left + output_size)))
    img.zoom(1.0)
    img.resize((output_size, output_size))
    # img.rotate(rot, center)
    # img.normalise()
    

    # Depth
    depth = image.DepthImage.from_tiff(dataset.depth_files[i])
    depth_img = tf.convert_to_tensor(depth)
    depth_img = image.DepthImage.from_tensor(depth_img)
    depth_img.rotate(rot, center)
    depth_img.crop((top, left), (min(480, top + output_size), min(640, left + output_size)))
    depth_img.normalise()
    depth_img.zoom(1.0)
    depth_img.resize((output_size, output_size))

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
    ax3.imshow(img)
    ax3.set_title('rgb_img')
    ax3.axis("off")

    ax3 = fig.add_subplot(rows, cols, 6)
    ax3.imshow(depth_img)
    ax3.set_title('depth_img')
    ax3.axis("off")
    
    rgb = imread(rgb)
    ax3 = fig.add_subplot(rows, cols, 7)
    ax3.imshow(rgb)
    ax3.set_title('original_rgb')
    ax3.axis("off")

    ax3 = fig.add_subplot(rows, cols, 8)
    ax3.imshow(depth)
    ax3.set_title('original_depth')
    ax3.axis("off")




    plt.show()
    



