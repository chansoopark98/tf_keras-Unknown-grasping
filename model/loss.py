import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy, Huber

class Loss:
    def __init__(self, use_aux=False):
        self.use_aux = use_aux

    def loss(self, y_true, y_pred):
        y_pos, y_cos, y_sin, y_width = tf.unstack(y_true)
        pred_pos, pred_cos, pred_sin, pred_width = tf.unstack(y_pred)

        pos_loss = BinaryCrossentropy(from_logits=False, name='pos_bce')(y_true=y_pos, y_pred=pred_pos)
        cos_loss = Huber(name='cos_smoothl1')(y_true=y_cos, y_pred=pred_cos)
        sin_loss = Huber(name='sin_smoothl1')(y_true=y_sin, y_pred=pred_sin)
        width_loss = Huber(name='width_smoothl1')(y_true=y_width, y_pred=pred_width)

        return pos_loss + cos_loss + sin_loss + width_loss
