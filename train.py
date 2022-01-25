import argparse
import time
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import numpy as np
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tqdm import tqdm
from model.model_builder import model_builder
from model.loss import Loss
import matplotlib.pyplot as plt
from utils.utils.dataset_processing import grasp, image
from utils.data_generator import augment
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
from skimage.filters import gaussian
from utils.utils.dataset_processing import evaluation


# LD_PRELOAD="/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4" python train.py

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


def calculate_iou_match(grasp_q, grasp_angle, ground_truth_bbs, no_grasps=1, grasp_width=None, threshold=0.25):
    # grasp_q = grasp_q.numpy()
    # grasp_angle = grasp_angle.numpy()
    # grasp_width = grasp_width.numpy()

    if not isinstance(ground_truth_bbs, grasp.GraspRectangles):
        gt_bbs = grasp.GraspRectangles.load_from_array(ground_truth_bbs.numpy())
    else:
        gt_bbs = ground_truth_bbs
    gs = grasp.detect_grasps(grasp_q, grasp_angle, width_img=grasp_width, no_grasps=no_grasps)
    for g in gs:
        if g.max_iou(gt_bbs) > threshold:
            return True
    else:
        return False

def poly_decay(lr=3e-4, max_epochs=100, warmup=False):
    """
    poly decay.
    :param lr: initial lr
    :param max_epochs: max epochs
    :param warmup: warm up or not
    :return: current lr
    """
    max_epochs = max_epochs - 5 if warmup else max_epochs

def decay(current_lr, current_epochs, epochs):
    # lrate = current_lr * (1 - np.power(current_epochs / epochs, 0.9))
    if current_epochs <= 30:
        lrate = 0.001
    elif current_epochs <= 100:
        lrate = 0.0001
    elif current_epochs <= 200:
        lrate = 0.00001
    return lrate


tf.keras.backend.clear_session()

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size",     type=int,   help="배치 사이즈값 설정", default=8)
parser.add_argument("--epoch",          type=int,   help="에폭 설정", default=200)
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
IMAGE_SIZE = (224, 224, 4)
USE_WEIGHT_DECAY = args.use_weightDecay
LOAD_WEIGHT = args.load_weight
MIXED_PRECISION = args.mixed_precision
DISTRIBUTION_MODE = args.distribution_mode
if MIXED_PRECISION:
    policy = mixed_precision.Policy('mixed_float16', loss_scale=1024)
    mixed_precision.set_policy(policy)

os.makedirs(SAVE_WEIGHTS_DIR, exist_ok=True)
train_data, meta_data = tfds.load('CornellGrasp', data_dir='./tfds/', split='train', with_info=True)
number_train = meta_data.splits['train'].num_examples
steps_per_epoch = number_train // BATCH_SIZE
train_data = train_data.shuffle(1024)
train_data = train_data.padded_batch(BATCH_SIZE)
# train_data = train_data.repeat()
train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)

test_data, test_meta_data = tfds.load('CornellGrasp', data_dir='./tfds/', split='train[95%:]', with_info=True)
number_test = test_meta_data.splits['train'].num_examples
test_steps_per_epoch = number_test // BATCH_SIZE
test_data = test_data.padded_batch(BATCH_SIZE)
# train_data = train_data.repeat()
# test_data = test_data.prefetch(tf.data.experimental.AUTOTUNE)

model_input, model_output = model_builder(input_shape=IMAGE_SIZE)
model = tf.keras.Model(model_input, model_output)

loss = Loss(use_aux=False)

optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr)

model.compile(optimizer=optimizer, loss=loss.test_loss)
model.summary()
# model.load_weights(SAVE_WEIGHTS_DIR+'/'+'23.h5')

loss_name = ['total', 'pos', 'cos', 'sin', 'width']
rows = 3
cols = 4
for epoch in range(EPOCHS):
    pbar = tqdm(train_data, total=steps_per_epoch, desc='Batch', leave=True, disable=False)
    batch_counter = 0
    toggle = True
    dis_res = 0
    index = 0
    results = {
        'correct': 0,
        'failed': 0}
    lr = decay(base_lr, current_epochs=epoch, epochs=EPOCHS)
    K.set_value(model.optimizer.learning_rate, lr)
    for sample in pbar:
        batch_counter += 1
        # ---------------------
        #  Train
        # ---------------------
        rgb = sample['rgb']
        depth = sample['depth']
        box = sample['box']

        batch_input = []
        batch_label = []
        pos_stack =[]
        cos_stack =[]
        sin_stack =[]
        width_stack =[]
        for i in range(len(rgb)):
            # input_stack, label_stack, _ = augment(rgb=rgb[i], depth=depth[i], box=box[i], output_size=IMAGE_SIZE[0])
            input_stack, pos, cos, sin, width, _ = augment(rgb=rgb[i], depth=depth[i], box=box[i], output_size=IMAGE_SIZE[0])
            batch_input.append(input_stack)
            pos_stack.append(pos)
            cos_stack.append(cos)
            sin_stack.append(sin)
            width_stack.append(width)
            # batch_label.append(label_stack)
            
            
            
        batch_input = tf.convert_to_tensor(batch_input, dtype=tf.float32)
        # batch_label = tf.convert_to_tensor(batch_label, dtype=tf.float32)
        pos_stack = tf.convert_to_tensor(pos_stack, dtype=tf.float32)
        cos_stack = tf.convert_to_tensor(cos_stack, dtype=tf.float32)
        sin_stack = tf.convert_to_tensor(sin_stack, dtype=tf.float32)
        width_stack = tf.convert_to_tensor(width_stack, dtype=tf.float32)

        batch_loss = model.train_on_batch(batch_input, [pos_stack, cos_stack, sin_stack, width_stack])
        # batch_loss = tf.reduce_mean(batch_loss)
        pbar.set_description("Epoch : %d lr: %f pos loss: %f cos loss: %f sin loss: %f width loss: %f" % (epoch, lr, batch_loss[0], batch_loss[1], batch_loss[2], batch_loss[3]))
        # pbar.set_description("Epoch : %d Total loss: %f pos loss: %f cos loss: %f sin loss: %f width loss: %f" % (epoch, 
                                        # batch_loss[0], batch_loss[1], batch_loss[2], batch_loss[3], batch_loss[4]))


    model.save_weights(SAVE_WEIGHTS_DIR +'/' + str(epoch) + '.h5', overwrite=True)
    TEST_SAVE_EPCOHS_DIR = SAVE_WEIGHTS_DIR +'/'+ str(epoch) + '/'
    os.makedirs(TEST_SAVE_EPCOHS_DIR, exist_ok=True)
    # validation

    if epoch % 3 == 0:
        for sample in test_data:
            rgb = sample['rgb']
            depth = sample['depth']
            box = sample['box']

            batch_input = []
            batch_label = []
            pos_stack =[]
            cos_stack =[]
            sin_stack =[]
            width_stack =[]
            batch_gtbbs = [] 
            for i in range(len(rgb)):
                input_stack, pos, cos, sin, width, gtbbs = augment(rgb=rgb[i], depth=depth[i], box=box[i], output_size=IMAGE_SIZE[0])
                batch_input.append(input_stack)
                pos_stack.append(pos)
                cos_stack.append(cos)
                sin_stack.append(sin)
                width_stack.append(width)
                batch_gtbbs.append(gtbbs)
            batch_input = tf.convert_to_tensor(batch_input)
            # batch_label = tf.convert_to_tensor(batch_label)
            pos_stack = tf.convert_to_tensor(pos_stack, dtype=tf.float32)
            cos_stack = tf.convert_to_tensor(cos_stack, dtype=tf.float32)
            sin_stack = tf.convert_to_tensor(sin_stack, dtype=tf.float32)
            width_stack = tf.convert_to_tensor(width_stack, dtype=tf.float32)
            
            preds = model.predict(batch_input)          
            
            fig = plt.figure()
            for i in range(len(batch_input)):
                ax0 = fig.add_subplot(rows, cols, 1)
                ax0.imshow(preds[0][i, :, :])
                ax0.set_title('pred_pos')
                ax0.axis("off")

                ax0 = fig.add_subplot(rows, cols, 2)
                ax0.imshow(preds[1][i, :, :])
                ax0.set_title('pred_cos')
                ax0.axis("off")

                ax0 = fig.add_subplot(rows, cols, 3)
                ax0.imshow(preds[2][i, :, :])
                ax0.set_title('pred_sin')
                ax0.axis("off")

                ax0 = fig.add_subplot(rows, cols, 4)
                ax0.imshow(preds[3][i, :, :])
                ax0.set_title('pred_width')
                ax0.axis("off")

                ax0 = fig.add_subplot(rows, cols, 5)
                ax0.imshow(pos_stack[i])
                ax0.set_title('gt_pos')
                ax0.axis("off")

                ax0 = fig.add_subplot(rows, cols, 6)
                ax0.imshow(cos_stack[i])
                ax0.set_title('gt_cos')
                ax0.axis("off")

                ax0 = fig.add_subplot(rows, cols, 7)
                ax0.imshow(sin_stack[i])
                ax0.set_title('gt_sin')
                ax0.axis("off")

                ax0 = fig.add_subplot(rows, cols, 8)
                ax0.imshow(width_stack[i])
                ax0.set_title('gt_width')
                ax0.axis("off")
                
                q_img, ang_img, width_img = post_processing(q_img=preds[0][i, :, :],
                                                            cos_img=preds[1][i, :, :],
                                                            sin_img=preds[2][i, :, :],
                                                            width_img=preds[3][i, :, :])
                s = evaluation.calculate_iou_match(grasp_q = q_img,
                                        grasp_angle = ang_img,
                                        ground_truth_bbs = batch_gtbbs[i],
                                        no_grasps = 1,
                                        grasp_width = width_img,
                                        threshold=0.25)
                if s:
                    results['correct'] += 1
                else:
                    results['failed'] += 1

                ax0 = fig.add_subplot(rows, cols, 9)
                ax0.imshow(q_img)
                ax0.set_title('q_img')
                ax0.axis("off")

                ax0 = fig.add_subplot(rows, cols, 10)
                ax0.imshow(ang_img)
                ax0.set_title('ang_img')
                ax0.axis("off")

                ax0 = fig.add_subplot(rows, cols, 11)
                ax0.imshow(width_img)
                ax0.set_title('width_img')
                ax0.axis("off")

                img = batch_input[i, :, :, :3]
                ax0 = fig.add_subplot(rows, cols, 12)
                ax0.imshow(tf.clip_by_value(tf.cast(img+0.5, tf.float32), 0., 1.))
                ax0.set_title('input')
                ax0.axis("off")

                plt.savefig(SAVE_WEIGHTS_DIR +'/'+ str(epoch) + '/'+ str(index)+'.png', dpi=200)
                index +=1
        
        print("IoU", results['correct'] / (results['correct'] + results['failed']))