import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
from utils.dataset_processing import grasp, image
import numpy as np

dataset_path = './tfds/'

train_data, meta = tfds.load('CornellGrasp', split='train', with_info=True)

BATCH_SIZE = 1


number_train = meta.splits['train'].num_examples
print(number_train)

# number_train = train_data.reduce(0, lambda x, _: x + 1).numpy()
# print("학습 데이터 개수", number_train)
steps_per_epoch = number_train // BATCH_SIZE
train_data = train_data.shuffle(1024)
train_data = train_data.padded_batch(BATCH_SIZE)
train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)

rows = 1
cols = 6

output_size = 224
rot = 0

for batch in train_data:
    rgb = batch['rgb']
    depth = batch['depth']
    box = batch['box']

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
        # img.normalise()
        

        # Depth
        depth_img = image.DepthImage.from_tensor(depth[i])
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
            
        plt.show()