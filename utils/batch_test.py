import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
from utils.dataset_processing import grasp, image
import numpy as np

dataset_path = './tfds/'

train_data, meta = tfds.load('CornellGrasp', split='train', with_info=True)

BATCH_SIZE = 2


number_train = meta.splits['train'].num_examples
print(number_train)

# number_train = train_data.reduce(0, lambda x, _: x + 1).numpy()
# print("학습 데이터 개수", number_train)
steps_per_epoch = number_train // BATCH_SIZE
train_data = train_data.shuffle(1024)
train_data = train_data.padded_batch(BATCH_SIZE)
train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)

rows = 1
cols = 5

output_size = 224
rot = 0

for batch in train_data:
    rgb = batch['rgb']
    depth = batch['depth']
    box = batch['box']

    batch_x = []
    batch_pos = []
    batch_cos = []
    batch_sin = []
    batch_width = []


    for i in range(BATCH_SIZE):
        gtbbs = grasp.GraspRectangles.load_from_tensor(box[i])

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
        img = image.Image.from_tensor(rgb[i])
        img.rotate(rot, center)
        img.crop((top, left), (min(480, top + output_size), min(640, left + output_size)))
        img.zoom(1.0)
        img.resize((output_size, output_size))
        # img.rotate(rot, center)
        img.normalise()
        

        # Depth
        depth_img = image.DepthImage.from_tensor(depth[i])
        depth_img.rotate(rot, center)
        depth_img.crop((top, left), (min(480, top + output_size), min(640, left + output_size)))
        depth_img.normalise()
        depth_img.zoom(1.0)
        depth_img.resize((output_size, output_size))

        batch_x.append(img)
        batch_pos.append(pos_img)
        batch_cos.append(cos)
        batch_sin.append(sin)
        batch_width.append(width_img)
    
    batch_x = tf.stack(batch_x, axis=0)
    batch_pos = tf.stack(batch_pos, axis=0)
    batch_cos = tf.stack(batch_cos, axis=0)
    batch_sin = tf.stack(batch_sin, axis=0)
    batch_width = tf.stack(batch_width, axis=0)



    for i in range(BATCH_SIZE):
        fig = plt.figure()
        ax0 = fig.add_subplot(rows, cols, 1)
        ax0.imshow(tf.cast((batch_x[i]+1.)*127.5, tf.uint8))
        ax0.set_title('input_image')
        ax0.axis("off")

        ax1 = fig.add_subplot(rows, cols, 2)
        ax1.imshow(batch_pos[i])
        ax1.set_title('batch_pos')
        ax1.axis("off")

        ax2 = fig.add_subplot(rows, cols, 3)
        ax2.imshow(batch_cos[i])
        ax2.set_title('batch_cos')
        ax2.axis("off")

        ax3 = fig.add_subplot(rows, cols, 4)
        ax3.imshow(batch_sin[i])
        ax3.set_title('batch_sin')
        ax3.axis("off")

        ax4 = fig.add_subplot(rows, cols, 5)
        ax4.imshow(batch_width[i])
        ax4.set_title('batch_width')
        ax4.axis("off")

        plt.show()