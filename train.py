import argparse
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

# LD_PRELOAD="/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4" python train.py

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

def decay(current_lr, current_epochs, epochs):
    lrate = current_lr * (1 - np.power(current_epochs / epochs, 0.9))
    return lrate


tf.keras.backend.clear_session()

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
mode = 'cornell'
cornell_path = './datasets/Cornell/'
jacquard_path = './datasets/Samples/'
dataset = CornellDataset(file_path=cornell_path, output_size=output_size)
jacquard = JacquardDataset(file_path=jacquard_path, output_size=output_size)


steps_per_epoch = dataset.length // BATCH_SIZE


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
# model.load_weights(SAVE_WEIGHTS_DIR+'/'+'23.h5')

loss_name = ['total', 'pos', 'cos', 'sin', 'width']
rows = 3
cols = 4
validation_length = 16
validation_freq = 3
np.random.seed(123)
# create tensorboard graph data for the model
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_DIR +'/' + SAVE_MODEL_NAME + '_' + str(IMAGE_SIZE[0]), 
                                    histogram_freq=0, 
                                    batch_size=BATCH_SIZE, 
                                    write_graph=True, 
                                    write_grads=False)
tensorboard.set_model(model)


for epoch in range(EPOCHS):
    batch_counter = 0
    toggle = True
    dis_res = 0
    index = 0
    results = {
        'correct': 0,
        'failed': 0}
    lr = decay(base_lr, current_epochs=epoch, epochs=EPOCHS)
    K.set_value(model.optimizer.learning_rate, lr)
    
    batch_idx = list(range(dataset.length))
    # batch_idx = batch_idx[validation_length:]
    # validation_idx = batch_idx[:validation_length]
    validation_idx = list(range(jacquard.length))

    pbar = tqdm(range(len(batch_idx)//BATCH_SIZE), total=len(batch_idx)//BATCH_SIZE, desc='Batch', leave=True, disable=False)    
    for j in pbar:
        batch_counter += 1
        # ---------------------
        #  Train
        # ---------------------
        

        batch_input = []
        batch_label = []
        pos_stack =[]
        cos_stack =[]
        sin_stack =[]
        width_stack =[]
        

        for i in range(BATCH_SIZE):
            rotations = [0, np.pi / 2, 2 * np.pi / 2, 3 * np.pi / 2]
            rot = random.choice(rotations)
            zoom_factor = np.random.uniform(0.5, 1.0)
            
            rnd_range = random.choice(batch_idx)
            iter_idx = rnd_range
            batch_idx.remove(rnd_range)
            
            
            # get bbox
            bbs = dataset.get_gtbb(idx=iter_idx, rot=rot, zoom=zoom_factor)
            # get depth
            depth_img = dataset.get_depth(idx=iter_idx, rot=rot, zoom=zoom_factor)
            # get img
            rgb_img = dataset.get_rgb(idx=iter_idx, rot=rot, zoom=zoom_factor)
            
            pos_img, ang_img, width_img = bbs.draw((output_size, output_size))
            width_img = np.clip(width_img, 0.0, output_size / 2) / (output_size / 2)

            cos = np.cos(2 * ang_img)
            sin = np.sin(2 * ang_img)

            depth_img = tf.expand_dims(depth_img, axis=-1)
            rgbd = tf.concat([rgb_img, depth_img], axis=-1)

            batch_input.append(rgbd)
            pos_stack.append(pos_img)
            cos_stack.append(cos)
            sin_stack.append(sin)
            width_stack.append(width_img)
            
            
        batch_input = tf.convert_to_tensor(batch_input, dtype=tf.float32)
        # batch_label = tf.convert_to_tensor(batch_label, dtype=tf.float32)
        pos_stack = tf.convert_to_tensor(pos_stack, dtype=tf.float32)
        pos_stack = tf.expand_dims(pos_stack, axis=-1)
        cos_stack = tf.convert_to_tensor(cos_stack, dtype=tf.float32)
        cos_stack = tf.expand_dims(cos_stack, axis=-1)
        sin_stack = tf.convert_to_tensor(sin_stack, dtype=tf.float32)
        sin_stack = tf.expand_dims(sin_stack, axis=-1)
        width_stack = tf.convert_to_tensor(width_stack, dtype=tf.float32)
        width_stack = tf.expand_dims(width_stack, axis=-1)
    
    
    


        batch_label = tf.concat([pos_stack, cos_stack, sin_stack, width_stack], axis=-1)
        batch_loss = model.train_on_batch(batch_input, batch_label)

        tensorboard.on_epoch_end(batch_counter, {'train_loss': batch_loss})
        
        # pbar.set_description("Epoch : %d | lr: %f | Total loss: %f | pos loss: %f | cos loss: %f | sin loss: %f | width loss: %f" 
                            # % (epoch, lr, total_loss, batch_loss[0], batch_loss[1], batch_loss[2], batch_loss[3]))

        pbar.set_description("Epoch : %d | lr: %f | Total loss: %f" 
                            % (epoch, lr, batch_loss))

    
    
    # validation
    if epoch % validation_freq == 0:
        # Save training weight
        model.save_weights(SAVE_WEIGHTS_DIR +'/' + str(epoch) + '.h5', overwrite=True)

        # Create validation results (per validation frequency)
        TEST_SAVE_EPCOHS_DIR = SAVE_WEIGHTS_DIR +'/'+ str(epoch) + '/'
        os.makedirs(TEST_SAVE_EPCOHS_DIR, exist_ok=True)
        
        for sample in range(len(validation_idx)//BATCH_SIZE):
            batch_input = []
            batch_label = []
            batch_gtbbs = []
            batch_before_norm = []
            pos_stack =[]
            cos_stack =[]
            sin_stack =[]
            width_stack =[]

            for i in range(BATCH_SIZE):
                rot = 0
                zoom_factor = 1.0
                
                rnd_range = random.choice(validation_idx)
                iter_idx = rnd_range
                validation_idx.remove(rnd_range)
                
                # get bbox
                bbs = jacquard.get_gtbb(idx=iter_idx, rot=rot, zoom=zoom_factor)
                # get depth
                depth_img = jacquard.get_depth(idx=iter_idx, rot=rot, zoom=zoom_factor)
                # get img
                rgb_img, rgb_ori = jacquard.get_rgb(idx=iter_idx, rot=rot, zoom=zoom_factor)
                
                pos_img, ang_img, width_img = bbs.draw((output_size, output_size))
                width_img = np.clip(width_img, 0.0, output_size / 2) / (output_size / 2)

                cos = np.cos(2 * ang_img)
                sin = np.sin(2 * ang_img)

                
                depth_img = tf.expand_dims(depth_img, axis=-1)
                rgbd = tf.concat([rgb_img, depth_img], axis=-1)

                batch_input.append(rgbd)
                batch_before_norm.append(rgb_ori)
                pos_stack.append(pos_img)
                cos_stack.append(cos)
                sin_stack.append(sin)
                width_stack.append(width_img)
                batch_gtbbs.append(bbs)


            batch_input = tf.convert_to_tensor(batch_input)
            pos_stack = tf.convert_to_tensor(pos_stack, dtype=tf.float32)
            cos_stack = tf.convert_to_tensor(cos_stack, dtype=tf.float32)
            sin_stack = tf.convert_to_tensor(sin_stack, dtype=tf.float32)
            width_stack = tf.convert_to_tensor(width_stack, dtype=tf.float32)
            
            preds = model.predict(batch_input)          
            
            # Draw validation results
            fig = plt.figure()
            for i in range(len(batch_input)):
                ax0 = fig.add_subplot(rows, cols, 1)
                ax0.imshow(preds[i, :, :, 0])
                ax0.set_title('pred_pos')
                ax0.axis("off")

                ax0 = fig.add_subplot(rows, cols, 2)
                ax0.imshow(preds[i, :, :, 1])
                ax0.set_title('pred_cos')
                ax0.axis("off")

                ax0 = fig.add_subplot(rows, cols, 3)
                ax0.imshow(preds[i, :, :, 2])
                ax0.set_title('pred_sin')
                ax0.axis("off")

                ax0 = fig.add_subplot(rows, cols, 4)
                ax0.imshow(preds[i, :, :, 3])
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
                
                q_img, ang_img, width_img = post_processing(q_img=preds[i, :, :, 0],
                                                            cos_img=preds[i, :, :, 1],
                                                            sin_img=preds[i, :, :, 2],
                                                            width_img=preds[i, :, :, 3])

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
                ax0.imshow(tf.cast(batch_before_norm[i], tf.uint8))
                ax0.set_title('input')
                ax0.axis("off")

                plt.savefig(SAVE_WEIGHTS_DIR +'/'+ str(epoch) + '/'+ str(index)+'.png', dpi=200)
                index +=1

                # grasps = grasp.detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=1)
                # with open(jo_fn, 'a') as f:
                    # for g in grasps:
                        # f.write(test_data.dataset.get_jname(didx) + '\n')
                        # f.write(g.to_jacquard(scale=1024 / 300) + '\n')
                if i == 0:
                    plot_rgb, _ = jacquard.get_rgb(i, rot, zoom_factor, normalise=False)
                    plot_depth = jacquard.get_depth(i, rot, zoom=zoom_factor)
                    save_results(rgb_img=plot_rgb,
                    depth_img=plot_depth,
                    grasp_q_img=q_img,
                    grasp_angle_img=ang_img,
                    no_grasps=1,
                    grasp_width_img=width_img,
                    epoch=epoch)

        iou = results['correct'] / (results['correct'] + results['failed'])
        print("IoU", iou)
        tensorboard.on_epoch_end(epoch, {'IoU': iou})