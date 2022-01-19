import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt

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

for batch in train_data:
    rgb = batch['rgb']
    depth = batch['depth']
    plt.imshow(depth[0])
    plt.show()
    