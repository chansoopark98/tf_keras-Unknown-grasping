from lzma import MODE_FAST

import tensorflow as tf
from model.model import grconvnet

def model_builder(input_shape):
    model_input, model_output = grconvnet(input_shape=input_shape, channel_size=32, output_channels=1)

    model = tf.keras.Model(model_input, model_output)

    return model