import numpy as np
import random
import tensorflow as tf
import tensorflow_datasets as tfds
from utils.utils.dataset_processing import grasp, image

AUTO = tf.data.experimental.AUTOTUNE

class DatasetGenerate:
    def __init__(self, dataset_name, data_dir, output_size, batch_size, input_type):
        """
             Args:
                 dataset_name: the type of dataset to load (ex. Cornell_dataset)
                 data_dir: Dataset relative path
                 image_size: size of image resolution according to DL model
                 batch_size: batch size
                 input_type: (rgb, depth, rgb_depth)
        """
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.output_size = output_size
        self.batch_size = batch_size
        self.input_type = input_type
        self.train_data, self.number_train = self._load_dataset()
    
    
    def _load_dataset(self):
        train_data, meta_data = tfds.load('CornellGrasp', data_dir=self.data_dir, split='train', with_info=True)
        number_train = meta_data.splits['train'].num_examples
        return train_data, number_train



def preprocess(sample):
    rgb = sample['rgb']
    depth = sample['depth']
    box = sample['box']

    return (rgb, depth, box)

def body(gtbbs):
    return [gtbbs.points]

def _get_crop_attrs(gtbbs, output_size):
    # center = gtbbs.center
    
    """
    Compute mean center of all GraspRectangles
    :return: float, mean centre of all GraspRectangles
    """
    # points = [gr.points for gr in gtbbs]
    # center = gtbbs.center
    
    # vstack = tf.experimental.numpy.vstack(points)
    # center = tf.reduce_mean(vstack, axis=0)
    # center = tf.reduce_mean(
            # np.vstack(points),axis=0, keepdims=False)
    
    center = gtbbs.center
    left = tf.math.maximum(0, tf.math.minimum(center[1] - output_size // 2, 640 - output_size))
    top = tf.math.maximum(0, tf.math.minimum(center[0] - output_size // 2, 480 - output_size))
    return center, left, top


def augment(rgb, depth, box, output_size):
    # get random value
    rotations = [0, np.pi / 2, 2 * np.pi / 2, 3 * np.pi / 2]
    rot = random.choice(rotations)
    zoom_factor = np.random.uniform(0.5, 1.0)

    # get rgb, depth, box
    rgb_img = image.Image.from_tensor(rgb)
    depth_img = image.DepthImage.from_tensor(depth)
    # depth_img.inpaint()
    gtbbs = grasp.GraspRectangles.load_from_tensor(box)
    
    center, left, top = _get_crop_attrs(gtbbs=gtbbs, output_size=output_size)

    # augment rgb
    rgb_img.rotate(rot, center)
    rgb_img.crop((top, left), (tf.math.minimum(480, top + output_size), tf.math.minimum(640, left + output_size)))
    # rgb_img.zoom(zoom_factor)
    rgb_img.resize((output_size, output_size))
    rgb_img.normalise()
    
    # augment depth
    depth_img.rotate(rot, center)
    depth_img.crop((top, left), (tf.math.minimum(480, top + output_size), tf.math.minimum(640, left + output_size)))
    depth_img.normalise()
    # depth_img.zoom(zoom_factor)
    depth_img.resize((output_size, output_size))

    # augment gtbbs
    gtbbs.rotate(rot, center)
    gtbbs.offset((-top, -left))
    # gtbbs.zoom(zoom_factor, (output_size // 2, output_size // 2))

    pos_img, ang_img, width_img = gtbbs.draw((output_size, output_size))
    width_img = tf.clip_by_value(width_img, 0.0, output_size /2 ) / (output_size / 2)
    cos = tf.math.cos(2 * ang_img)
    sin = tf.math.sin(2 * ang_img)

    rgb_img = tf.convert_to_tensor(rgb_img, tf.float32)
    depth_img = tf.convert_to_tensor(depth_img, tf.float32)
    # depth_img = tf.expand_dims(depth_img, axis=-1)
    img = tf.concat([rgb_img, depth_img], axis=-1)

    pos = tf.convert_to_tensor(pos_img, tf.float32)
    pos = tf.expand_dims(pos, axis=-1)

    cos = tf.cast(cos, tf.float32)
    cos = tf.convert_to_tensor(cos, tf.float32)
    cos = tf.expand_dims(cos, axis=-1)

    sin = tf.cast(sin, tf.float32)
    sin = tf.convert_to_tensor(sin, tf.float32)
    sin = tf.expand_dims(sin, axis=-1)

    width = tf.cast(width_img, tf.float32)
    width = tf.convert_to_tensor(width, tf.float32)
    width = tf.expand_dims(width, axis=-1)
        
    label = tf.concat([pos, cos, sin, width], axis=-1)

    return img, label, gtbbs
