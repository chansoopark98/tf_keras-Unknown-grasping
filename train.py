import argparse
import time
import tensorflow as tf
import tensorflow_datasets as tfds
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


# LD_PRELOAD="/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4" python train.py

def write_log(callback, names, logs, batch_no):
    for name in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = logs
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()


tf.keras.backend.clear_session()

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size",     type=int,   help="배치 사이즈값 설정", default=8)
parser.add_argument("--epoch",          type=int,   help="에폭 설정", default=100)
parser.add_argument("--lr",             type=float, help="Learning rate 설정", default=0.01)
parser.add_argument("--weight_decay",   type=float, help="Weight Decay 설정", default=0.0005)
parser.add_argument("--optimizer",     type=str,   help="Optimizer", default='adam')
parser.add_argument("--model_name",     type=str,   help="저장될 모델 이름",
                    default=str(time.strftime('%m%d', time.localtime(time.time()))))
parser.add_argument("--dataset_dir",    type=str,   help="데이터셋 다운로드 디렉토리 설정", default='./datasets/')
parser.add_argument("--checkpoint_dir", type=str,   help="모델 저장 디렉토리 설정", default='./checkpoints/')
parser.add_argument("--tensorboard_dir",  type=str,   help="텐서보드 저장 경로", default='tensorboard')
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
IMAGE_SIZE = (224, 224, 4)
USE_WEIGHT_DECAY = args.use_weightDecay
LOAD_WEIGHT = args.load_weight
MIXED_PRECISION = args.mixed_precision
DISTRIBUTION_MODE = args.distribution_mode
if MIXED_PRECISION:
    policy = mixed_precision.Policy('mixed_float16', loss_scale=1024)
    mixed_precision.set_policy(policy)


train_data, meta_data = tfds.load('CornellGrasp', data_dir='./tfds/', split='train', with_info=True)
number_train = meta_data.splits['train'].num_examples
steps_per_epoch = number_train // BATCH_SIZE
train_data = train_data.shuffle(1024)
train_data = train_data.padded_batch(BATCH_SIZE)
# train_data = train_data.repeat()
train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)



model_input, model_output = model_builder(input_shape=IMAGE_SIZE)
model = tf.keras.Model(model_input, model_output)

loss = Loss(use_aux=False)

optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr)
if MIXED_PRECISION:
    optimizer = mixed_precision.aschedules.PolynomialDecay(initial_learning_rate=base_lr,
                                                          decay_steps=EPOCHS,
                                                          end_learning_rate=base_lr*0.1, power=0.9)

polyDecay = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=base_lr,
                                                          decay_steps=EPOCHS,
                                                          end_learning_rate=0.0001, power=0.9)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(polyDecay,verbose=1)

checkpoint_val_loss = ModelCheckpoint(CHECKPOINT_DIR + '_' + SAVE_MODEL_NAME + '_best_loss.h5',
                                      monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_DIR, write_graph=True, write_images=True)

callback = [checkpoint_val_loss,  tensorboard, lr_scheduler]

model.compile(
    optimizer=optimizer,
    loss=loss.loss
)

model.summary()
loss_name = ['total', 'pos', 'cos', 'sin', 'width']
for epoch in range(EPOCHS):
    pbar = tqdm(train_data, total=steps_per_epoch, desc='Batch', leave=True, disable=False)
    batch_counter = 0
    toggle = True
    dis_res = 0
    index = 0
    for sample in pbar:
        batch_counter += 1
        # ---------------------
        #  Train Discriminator
        # ---------------------
        # img = tf.cast(features['image']def write_log(callback, names, logs, batch_no):
        rgb = sample['rgb']
        depth = sample['depth']
        box = sample['box']

        batch_input = []
        batch_label = []
        for i in range(len(rgb)):
            input_stack, label_stack = augment(rgb=rgb[i], depth=depth[i], box=box[i])

            batch_input.append(input_stack)
            batch_label.append(label_stack)
            
            # batch_input = tf.concat([batch_input, input_stack], axis=0)
            # batch_label = tf.concat([batch_label, label_stack], axis=0)
        batch_input = tf.convert_to_tensor(batch_input)
        batch_label = tf.convert_to_tensor(batch_label)
        batch_loss = model.train_on_batch(batch_input, batch_label)
        # batch_loss = tf.reduce_mean(batch_loss)
        pbar.set_description("Epoch : %d Total loss: %f" % (epoch, batch_loss))
        # pbar.set_description("Epoch : %d Total loss: %f pos loss: %f cos loss: %f sin loss: %f width loss: %f" % (epoch, 
                                        # batch_loss[0], batch_loss[1], batch_loss[2], batch_loss[3], batch_loss[4]))