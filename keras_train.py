import argparse
import tensorflow_datasets as tfds
import time
import os
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from torch import le
from tqdm import tqdm
from model.model_builder import model_builder
from model.loss import Loss
import matplotlib.pyplot as plt
from utils.utils.dataset_processing import grasp, image
from utils.data_generator import augment
from tensorflow.keras import backend as K
from skimage.filters import gaussian
from utils.utils.dataset_processing import evaluation, grasp
from utils.utils.visualisation.plot import save_results
from utils.data_generator_test import CornellDataset, JacquardDataset
import random 
import tensorflow_addons as tfa
tf.keras.backend.clear_session()
# LD_PRELOAD="/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4" python train.py

def preprocess(sample):
    tfds_rgb = sample['rgb']
    tfds_depth = sample['depth']
    tfds_box = sample['box']
    
    return (tfds_rgb, tfds_depth, tfds_box)

def augment(tfds_rgb, tfds_depth, tfds_box):

    # get center
    c = output_size // 2

    # rotate box
    rotations = [0, np.pi / 2, 2 * np.pi / 2, 3 * np.pi / 2]
    rot = random.choice(rotations)
    zoom_factor = np.random.uniform(0.5, 1.0)

    # zoom box

    tfds_box = grasp.GraspRectangles.load_from_tensor(tfds_box)
    tfds_box.to_array()
    tfds_box.rotate(rot, (c, c))
    tfds_box.zoom(zoom_factor, (c, c))

    pos_img, ang_img, width_img  = tfds_box.draw((output_size, output_size))

    width_img = np.clip(width_img, 0.0, output_size /2 ) / (output_size / 2)
    cos = np.cos(2 * ang_img)
    sin = np.sin(2 * ang_img)

    pos_img = tf.expand_dims(pos_img, axis=-1)
    cos = tf.expand_dims(cos, axis=-1)
    sin = tf.expand_dims(sin, axis=-1)
    width_img = tf.expand_dims(width_img, axis=-1)

    output = tf.concat([pos_img, cos, sin, width_img], axis=-1)



    # input data
    rgb_img = image.Image.from_tensor(tfds_rgb)
    rgb_img.rotate(rot)
    rgb_img.zoom(zoom_factor)
    rgb_img.resize((output_size, output_size))
    rgb_img.normalise()

    # Depth
    depth_img = image.DepthImage.from_tensor(tfds_depth)
    depth_img.rotate(rot)
    depth_img.normalise()
    depth_img.zoom(zoom_factor)
    depth_img.resize((output_size, output_size))

    input = tf.concat([rgb_img, depth_img], axis=-1)
    input = tf.cast(input, tf.float64)
    return (input, output)



parser = argparse.ArgumentParser()
parser.add_argument("--batch_size",     type=int,   help="배치 사이즈값 설정", default=8)
parser.add_argument("--epoch",          type=int,   help="에폭 설정", default=300)
parser.add_argument("--lr",             type=float, help="Learning rate 설정", default=0.001)
parser.add_argument("--weight_decay",   type=float, help="Weight Decay 설정", default=0.0005)
parser.add_argument("--optimizer",     type=str,   help="Optimizer", default='adam')
parser.add_argument("--model_name",     type=str,   help="저장될 모델 이름",
                    default=str(time.strftime('%m%d', time.localtime(time.time()))))
parser.add_argument("--dataset_dir",    type=str,   help="데이터셋 다운로드 디렉토리 설정", default='./datasets/')
parser.add_argument("--checkpoint_dir", type=str,   help="모델 저장 디렉토리 설정", default='./checkpoints/')
parser.add_argument("--tensorboard_dir",  type=str,   help="텐서보드 저장 경로", default='tensorboard')
parser.add_argument("--save_weight",  type=str,   help="가중치 저장 경로", default='./checkpoints')
parser.add_argument("--use_weightDecay",  type=bool,  help="weightDecay 사용 유무", default=False)
parser.add_argument("--load_weight",  type=bool,  help="가중치 로드", default=False)
parser.add_argument("--mixed_precision",  type=bool,  help="mixed_precision 사용", default=False)
parser.add_argument("--distribution_mode",  type=bool,  help="분산 학습 모드 설정", default=True)

args = parser.parse_args()
WEIGHT_DECAY = args.weight_decay
OPTIMIZER_TYPE = args.optimizer
BATCH_SIZE = args.batch_size
EPOCHS = args.epoch
base_lr = args.lr
SAVE_MODEL_NAME = args.model_name
DATASET_DIR = args.dataset_dir
CHECKPOINT_DIR = args.checkpoint_dir
TENSORBOARD_DIR = args.tensorboard_dir
SAVE_WEIGHTS_DIR = args.save_weight
IMAGE_SIZE = (300, 300, 4)
USE_WEIGHT_DECAY = args.use_weightDecay
LOAD_WEIGHT = args.load_weight
MIXED_PRECISION = args.mixed_precision
DISTRIBUTION_MODE = args.distribution_mode


os.makedirs(SAVE_WEIGHTS_DIR, exist_ok=True)

output_size = IMAGE_SIZE[0]

dataset_path = './tfds/'

train_data, meta = tfds.load('Jacquard', split='train', with_info=True, shuffle_files=False)

number_train = meta.splits['train'].num_examples
steps_per_epoch = number_train // BATCH_SIZE

AUTO = tf.data.experimental.AUTOTUNE
train_data = train_data.map(preprocess)
train_data = train_data.map(lambda tfds_rgb, tfds_depth, tfds_box: tf.py_function(augment, [tfds_rgb, tfds_depth, tfds_box], [tf.float64]))
train_data = train_data.batch(BATCH_SIZE)
train_data = train_data.prefetch(AUTO)
train_data = train_data.repeat()

model_input, model_output = model_builder(input_shape=IMAGE_SIZE)
model = tf.keras.Model(model_input, model_output)

loss = Loss(use_aux=False)

optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr)
# optimizer = tfa.optimizers.RectifiedAdam(learning_rate =base_lr, weight_decay=0.00001)
if MIXED_PRECISION:
    policy = mixed_precision.Policy('mixed_float16', loss_scale=1024)
    mixed_precision.set_policy(policy)
    optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')  # tf2.4.1 이전

model.compile(optimizer=optimizer, loss=loss.loss)
model.summary()

history = model.fit(train_data,
                    steps_per_epoch=steps_per_epoch,
                    epochs=EPOCHS)

