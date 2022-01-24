import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy, Huber

class Loss:
    def __init__(self, use_aux=False):
        self.use_aux = use_aux

    def loss(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pos = y_true[:, :, :, 0]
        y_cos = y_true[:, :, :, 1]
        y_sin = y_true[:, :, :, 2]
        y_width = y_true[:, :, :, 3]

        pred_pos = y_pred[:, :, :, 0]
        pred_cos = y_pred[:, :, :, 1]
        pred_sin = y_pred[:, :, :, 2]
        pred_width = y_pred[:, :, :, 3]

        # pos_loss = BinaryCrossentropy(from_logits=False, name='pos_bce')(y_true=y_pos, y_pred=pred_pos)
        pos_loss = Huber(name='pose_smoothl1')(y_true=y_pos, y_pred=pred_pos)
        cos_loss = Huber(name='cos_smoothl1')(y_true=y_cos, y_pred=pred_cos)
        sin_loss = Huber(name='sin_smoothl1')(y_true=y_sin, y_pred=pred_sin)
        width_loss = Huber(name='width_smoothl1')(y_true=y_width, y_pred=pred_width)

        return pos_loss + cos_loss + sin_loss + width_loss
