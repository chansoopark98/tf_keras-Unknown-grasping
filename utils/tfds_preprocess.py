import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import random
from utils.dataset_processing import grasp, image
import matplotlib.pyplot as plt
dataset_path = './tfds/'

train_data, meta = tfds.load('Jacquard', split='train', with_info=True, shuffle_files=False)

BATCH_SIZE = 1
number_train = meta.splits['train'].num_examples

output_size = 300

def preprocess(sample):
    tfds_rgb = sample['rgb']
    tfds_depth = sample['depth']
    tfds_box = sample['box']
    
    return (tfds_box)

def augment(tfds_box):
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
    return output
    
train_data = train_data.map(preprocess)
# train_data = train_data.map(augment)
train_data = train_data.map(lambda train_data: tf.py_function(augment, [train_data], [tf.float64]))

rows=1
cols=4
train_data = train_data.take(100)

for output in train_data:
    # pos_img = label[0]
    # cos = label[1]
    # sin = label[2]
    # width_img = label[3] 

    fig = plt.figure()
    
    ax0 = fig.add_subplot(rows, cols, 1)
    ax0.imshow(output[0][:, :, 0])
    ax0.set_title('pos_img')
    ax0.axis("off")

    ax1 = fig.add_subplot(rows, cols, 2)
    ax1.imshow(output[0][:, :, 1])
    ax1.set_title('cos')
    ax1.axis("off")

    ax1 = fig.add_subplot(rows, cols, 3)
    ax1.imshow(output[0][:, :, 2])
    ax1.set_title('sin')
    ax1.axis("off")

    ax1 = fig.add_subplot(rows, cols, 4)
    ax1.imshow(output[0][:, :, 3])
    ax1.set_title('width')
    ax1.axis("off")



    # ax2 = fig.add_subplot(rows, cols, 3)
    # ax2.imshow(sin)
    # ax2.set_title('sin')
    # ax2.axis("off")

    # ax3 = fig.add_subplot(rows, cols, 4)
    # ax3.imshow(width_img)
    # ax3.set_title('width_img')
    # ax3.axis("off")

    # q_img, ang_img, width_img = post_processing(q_img=pos_img,
    #                                         cos_img=cos,
    #                                         sin_img=sin,
    #                                         width_img=width_img)


    # ax3 = fig.add_subplot(rows, cols, 9)
    # ax3.imshow(q_img)
    # ax3.set_title('q_img')
    # ax3.axis("off")

    # ax3 = fig.add_subplot(rows, cols, 10)
    # ax3.imshow(ang_img)
    # ax3.set_title('ang_img')
    # ax3.axis("off")

    # ax3 = fig.add_subplot(rows, cols, 11)
    # ax3.imshow(width_img)
    # ax3.set_title('width_img')
    # ax3.axis("off")

    # ax3 = fig.add_subplot(rows, cols, 12)
    # ax3.imshow(inpaint_depth)
    # ax3.set_title('from_pcd_inpaint')
    # ax3.axis("off")
    # s = evaluation.calculate_iou_match(grasp_q = q_img,
    #                         grasp_angle = ang_img,
    #                         ground_truth_bbs = gtbbs,
    #                         no_grasps = 3,
    #                         grasp_width = width_img,
    #                         threshold=0.25)

    # print('iou results', s)


    plt.show()



    