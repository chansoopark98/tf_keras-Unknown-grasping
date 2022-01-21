from lzma import MODE_FAST

import tensorflow as tf
from model.model import grconvnet

def model_builder(input_shape):
    return grconvnet(input_shape=input_shape, channel_size=32, output_channels=1)