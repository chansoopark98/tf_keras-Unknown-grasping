import glob
import os
from imageio import imread, imsave, imwrite
from tqdm import tqdm
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
from utils.dataset_processing import grasp, image
import tensorflow as tf
from skimage.filters import gaussian
from utils.dataset_processing import evaluation
import tensorflow_datasets as tfds
# from utils.dataset_processing import grasp, image

class CornellDataset:
    def __init__(self, file_path, mode):
        """-
        :param file_path: Cornell Dataset directory
        """
        self.mode = mode
        if self.mode == 'cornell':
            self.grasp_files = glob.glob(os.path.join(file_path, '*', 'pcd*cpos.txt'))
            self.pcds =  glob.glob(os.path.join(file_path, '*', 'pcd*[0-9].txt'))
            self.pcds.sort()
            self.grasp_files.sort()
            self.length = len(self.grasp_files)

            self.depth_files = [f.replace('cpos.txt', 'd.tiff') for f in self.grasp_files] 
            self.rgb_files = [f.replace('d.tiff', 'r.png') for f in self.depth_files]
        
        else:
            # self.grasp_files = glob.glob(os.path.join(file_path+'/*/', '*', '*_grasps.txt'))
            self.grasp_files = glob.glob(os.path.join(file_path+'/Jacquard_Dataset_0/', '*', '*_grasps.txt'))
            
            self.grasp_files.sort()
            self.length = len(self.grasp_files)
            print(self.length)
            if self.length == 0:
                raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))

            self.depth_files = [f.replace('grasps.txt', 'perfect_depth.tiff') for f in self.grasp_files]
            self.depth_files.sort()
            self.rgb_files = [f.replace('perfect_depth.tiff', 'RGB.png') for f in self.depth_files]
            self.rgb_files.sort()

def write_log(callback, names, logs, batch_no):
    for name in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = logs
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

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

mode = 'jacquard'
# mode = 'cornell'

if mode == 'cornell':
    path = './datasets/Cornell/'
    output_path = './cornell_output/'
else:
    path = './datasets/Jacquard/'
    output_path = './jacquard_output/'
    
    
rgb_path = output_path + 'rgb/'
depth_path = output_path + 'depth/'
box_path = output_path + 'box/'
os.makedirs(output_path, exist_ok=True)
os.makedirs(rgb_path, exist_ok=True)
os.makedirs(depth_path, exist_ok=True)
os.makedirs(box_path, exist_ok=True)

output_size = 300
rows=4
cols=4
dataset = CornellDataset(file_path=path, mode=mode)
pbar = tqdm(range(dataset.length))

dataset_path = './tfds/'

train_data, meta = tfds.load('Jacquard', split='train', with_info=True, shuffle_files=False)

BATCH_SIZE = 1
number_train = meta.splits['train'].num_examples
print(number_train)

# number_train = train_data.reduce(0, lambda x, _: x + 1).numpy()
# print("?????? ????????? ??????", number_train)
steps_per_epoch = number_train // BATCH_SIZE

rotations = [0, np.pi / 2, 2 * np.pi / 2, 3 * np.pi / 2]
rot = random.choice(rotations)
zoom_factor = np.random.uniform(0.5, 1.0)
tfds_rgb = None
tfds_depth = None
tfds_box = None
for data in train_data:
    tfds_rgb = data['rgb']
    tfds_depth = data['depth']
    tfds_box = data['box']
    break


for i in pbar:
    if dataset.mode == 'cornell':
        # bbox
        rotations = [0, np.pi / 2, 2 * np.pi / 2, 3 * np.pi / 2]
        rot = random.choice(rotations)
        # zoom_factor = np.random.uniform(0.5, 1.0)
        zoom_factor = np.random.uniform(0.5, 1.0)
        
        bbox = dataset.grasp_files[i]
        
        gtbbs = grasp.GraspRectangles.load_from_cornell_file(bbox)
        
        center = gtbbs.center
        # center = (output_size // 2, output_size // 2)
        
        rgb = dataset.rgb_files[i]
        img = image.Image.from_file(rgb)
        original_img = img.copy()
        left = max(0, min(center[1] - output_size // 2, img.shape[1] - output_size))
        top = max(0, min(center[0] - output_size // 2, img.shape[0] - output_size))

        gtbbs.rotate(rot, center)
        gtbbs.offset((-top, -left))
        gtbbs.zoom(zoom_factor, (output_size // 2, output_size // 2))
        
        pos_img, ang_img, width_img = gtbbs.draw((output_size, output_size))
        width_img = np.clip(width_img, 0.0, output_size /2 ) / (output_size / 2)
        cos = np.cos(2 * ang_img)
        sin = np.sin(2 * ang_img)

        img.rotate(rot, center)
        img.crop((top, left), (min(img.shape[0], top + output_size), min(img.shape[1], left + output_size)))
        img.zoom(zoom_factor)
        img.resize((output_size, output_size))
        before_norm_img = img.copy()
        img.normalise()
        
        # Depth
        depth_img = image.DepthImage.from_tiff(dataset.depth_files[i])
        inpaint_depth = image.DepthImage.from_pcd(dataset.pcds[i], (480, 640))
        inpaint_depth.inpaint()
        inpaint_depth.rotate(rot, center)
        inpaint_depth.crop((top, left), (min(480, top + output_size), min(640, left + output_size)))
        inpaint_depth.normalise()
        inpaint_depth.zoom(zoom_factor)
        inpaint_depth.resize((output_size, output_size))
        depth_input = depth_img.copy()
        # depth_img.inpaint()        
        # depth_img.crop((top, left), (min(imgprint(dataset.grasp_files[i])
        depth_img.rotate(rot, center)
        depth_img.crop((top, left), (min(480, top + output_size), min(640, left + output_size)))
        depth_img.normalise()
        depth_img.zoom(zoom_factor)
        depth_img.resize((output_size, output_size))


    else:
        bbox = dataset.grasp_files[i]
        print(dataset.grasp_files[i])
        gtbbs = grasp.GraspRectangles.load_from_jacquard_file(bbox, output_size/1024.)
        get_shape = gtbbs.to_array()
        
        print(get_shape.shape)
        c = output_size // 2
        gtbbs.rotate(rot, (c, c))
        gtbbs.zoom(zoom_factor, (c, c))
        
        pos_img, ang_img, width_img = gtbbs.draw((output_size, output_size))
        width_img = np.clip(width_img, 0.0, output_size /2 ) / (output_size / 2)
        cos = np.cos(2 * ang_img)
        sin = np.sin(2 * ang_img)

        rgb = dataset.rgb_files[i]
        img = image.Image.from_file(rgb)

        
        rgb_img = image.Image.from_file(dataset.rgb_files[i])
        rgb_img.rotate(rot)
        rgb_img.zoom(zoom_factor)
        rgb_img.resize((output_size, output_size))
        before_norm_img = rgb_img.copy()
        rgb_img.normalise()

        # Depth
        depth_input = imread(dataset.depth_files[i])
        depth_img = image.DepthImage.from_tiff(dataset.depth_files[i])
        inpaint_depth = depth_img.copy()
        inpaint_depth.inpaint()
        depth_img.rotate(rot)
        depth_img.normalise()
        depth_img.zoom(zoom_factor)
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


    max_v = np.max(img)
    min_v = np.min(img)
    ax3 = fig.add_subplot(rows, cols, 5)
    ax3.imshow(before_norm_img)
    ax3.set_title('rgb_img')
    ax3.axis("off")

    ax3 = fig.add_subplot(rows, cols, 6)
    ax3.imshow(depth_img)
    ax3.set_title('depth_img')
    ax3.axis("off")
    
    rgb = imread(rgb)
    ax3 = fig.add_subplot(rows, cols, 7)
    ax3.imshow(img)
    ax3.set_title('original_rgb')
    ax3.axis("off")

    ax3 = fig.add_subplot(rows, cols, 8)
    ax3.imshow(depth_input)
    ax3.set_title('original_depth')
    ax3.axis("off")

    q_img, ang_img, width_img = post_processing(q_img=pos_img,
                                            cos_img=cos,
                                            sin_img=sin,
                                            width_img=width_img)


    ax3 = fig.add_subplot(rows, cols, 9)
    ax3.imshow(q_img)
    ax3.set_title('q_img')
    ax3.axis("off")

    ax3 = fig.add_subplot(rows, cols, 10)
    ax3.imshow(ang_img)
    ax3.set_title('ang_img')
    ax3.axis("off")

    ax3 = fig.add_subplot(rows, cols, 11)
    ax3.imshow(width_img)
    ax3.set_title('width_img')
    ax3.axis("off")

    ax3 = fig.add_subplot(rows, cols, 12)
    ax3.imshow(inpaint_depth)
    ax3.set_title('from_pcd_inpaint')
    ax3.axis("off")
    s = evaluation.calculate_iou_match(grasp_q = q_img,
                            grasp_angle = ang_img,
                            ground_truth_bbs = gtbbs,
                            no_grasps = 3,
                            grasp_width = width_img,
                            threshold=0.25)

    print('iou results', s)



    ax3 = fig.add_subplot(rows, cols, 13)
    ax3.imshow(tfds_rgb)
    ax3.set_title('tfds_rgb')
    ax3.axis("off")

    ax3 = fig.add_subplot(rows, cols, 14)
    ax3.imshow(tfds_depth)
    ax3.set_title('tfds_depth')
    ax3.axis("off")
    

    plt.show()
