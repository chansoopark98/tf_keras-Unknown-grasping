import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy, Huber

class Loss:
    def __init__(self, use_aux=False):
        self.use_aux = use_aux

    def binary_ce(self, y_true, y_pred):
        bce = BinaryCrossentropy(from_logits=False)(y_true=y_true, y_pred=y_pred)
        return bce
    
    def smoothl1(self, y_true, y_pred):
        smoothl1 = Huber()(y_true=y_true, y_pred=y_pred)
        return smoothl1