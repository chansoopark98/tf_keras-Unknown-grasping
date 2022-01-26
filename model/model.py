from pickle import FALSE
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    UpSampling2D, Activation, BatchNormalization, Conv2D,  Concatenate, LeakyReLU, MaxPooling2D, Input, Flatten, Dense, Dropout, concatenate,
    DepthwiseConv2D,  ZeroPadding2D, Conv2DTranspose, GlobalAveragePooling2D, Add)
from tensorflow.keras.activations import tanh, relu
import tensorflow as tf
from tensorflow.keras.models import Model
from torch import conv2d
from .efficientnetv2 import efficientnet_v2
from model.utils import sepconv, aspp


def conv3x3(x, filters, kernel_size, strides, padding,
            bn_momentum=0.99, bn_epsilon=0.001, activation='relu', prefix='name'):
    x = Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size), strides=(strides, strides), kernel_initializer='he_normal', padding=padding, name=prefix+'_conv2d')(x)
    x = BatchNormalization(momentum=bn_momentum, epsilon=bn_epsilon, name=prefix+'_bn')(x)
    x = Activation(activation, name=prefix+'_activation')(x)
    return x

def deconv3x3(x, filters, kernel_size, strides, padding, use_bn=True,
            bn_momentum=0.99, bn_epsilon=0.001, activation='relu', prefix='name'):
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, kernel_size), strides=(strides, strides), kernel_initializer='he_normal', padding=padding, name=prefix+'_deconv2d')(x)
    if use_bn:
        x = BatchNormalization(momentum=bn_momentum, epsilon=bn_epsilon, name=prefix+'_bn')(x)
    x = Activation(activation, name=prefix+'_activation')(x)
    return x

def res_block(x_input, filters, kernel_size, strides, padding,
            bn_momentum=0.99, bn_epsilon=0.001, activation='relu', prefix='name'):
    x = Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size), strides=(strides, strides), kernel_initializer='he_normal', padding=padding, name=prefix+'_res_conv2d_1')(x_input)
    x = BatchNormalization(momentum=bn_momentum, epsilon=bn_epsilon, name=prefix+'_res_bn_1')(x)
    x = Activation(activation, name=prefix+'_activation_1')(x)

    x = Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size), strides=(strides, strides), kernel_initializer='he_normal', padding=padding, name=prefix+'_res_conv2d_2')(x)
    x = BatchNormalization(momentum=bn_momentum, epsilon=bn_epsilon, name=prefix+'_res_bn_2')(x)

    return Add(name=prefix+'_res_add')([x, x_input])

def classifier(x, output_filters, kernel_size, activation, use_dropout, prefix):
    x = Dropout(use_dropout)(x)
    x = Conv2D(filters=output_filters, kernel_size=(kernel_size, kernel_size), strides=(1, 1),
                activation=activation, kernel_initializer='he_normal', padding='same', name=prefix+'_classifier')(x)
    return x


def grconvnet(input_shape, channel_size=32 , output_channels=1):
    inputs = Input(shape=(input_shape[0], input_shape[1], input_shape[2]), name='model_input')

    x = conv3x3(inputs, filters=channel_size, kernel_size=9, strides=1, padding='same', prefix='conv1')
    x = conv3x3(x, filters=channel_size * 2, kernel_size=4, strides=2, padding='same', prefix='conv2')
    x = conv3x3(x, filters=channel_size * 4, kernel_size=4, strides=2, padding='same', prefix='conv3')
    # x = conv3x3(x, filters=channel_size * 8, kernel_size=3, strides=2, padding='same', prefix='conv4')

    x = res_block(x_input=x, filters=channel_size * 4, kernel_size=3, strides=1, padding='same', prefix='res_1')
    x = res_block(x_input=x, filters=channel_size * 4, kernel_size=3, strides=1, padding='same', prefix='res_2')
    x = res_block(x_input=x, filters=channel_size * 4, kernel_size=3, strides=1, padding='same', prefix='res_3')
    x = res_block(x_input=x, filters=channel_size * 4, kernel_size=3, strides=1, padding='same', prefix='res_4')
    x = res_block(x_input=x, filters=channel_size * 4, kernel_size=3, strides=1, padding='same', prefix='res_5')
    
    x = deconv3x3(x, filters=channel_size *4, kernel_size=4, strides=2, padding='same', use_bn=True, prefix='deconv4')
    x = deconv3x3(x, filters=channel_size *2, kernel_size=4, strides=2, padding='same', use_bn=True, prefix='deconv3')
    x = deconv3x3(x, filters=channel_size, kernel_size=9, strides=1, padding='same', use_bn=True,prefix='deconv2')


    pos = classifier(x, output_filters=output_channels, kernel_size=3, activation=None, use_dropout=0.0, prefix='pos')
    cos = classifier(x, output_filters=output_channels, kernel_size=3, activation=None, use_dropout=0.0, prefix='cos')
    sin = classifier(x, output_filters=output_channels, kernel_size=3, activation=None, use_dropout=0.0, prefix='sin')
    width = classifier(x, output_filters=output_channels, kernel_size=3, activation=None, use_dropout=0.0, prefix='width')

    # outputs = tf.concat([pos, cos, sin, width], axis=-1)
    return inputs, [pos, cos, sin, width]


def CustomModel(input_shape, output_channels=1):
    base = efficientnet_v2.EfficientNetV2S(input_shape=input_shape, pretrained=None)
    inputs = base.input

    c1 = base.get_layer('add_1').output # 112x112 
    c1_size = tf.keras.backend.int_shape(c1)

    c2 = base.get_layer('add_4').output # 56x56 
    c2_size = tf.keras.backend.int_shape(c2)

    c3 = base.get_layer('add_7').output # 28x28 48
    c3_size = tf.keras.backend.int_shape(c3)

    c4 = base.get_layer('add_20').output # 14x14 16
    c4_size = tf.keras.backend.int_shape(c4)

    c5 = base.get_layer('add_34').output # 7x7 256

    x = aspp(x=c5, activation='swish')

    # C4 : Output stride = 16
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = Conv2D(filters=c4_size[3], kernel_size=(1 ,1), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
    x = Concatenate()([x, c4])
    x = sepconv(x=x, filters=c4_size[3], prefix='conv4_decoder1', stride=1, kernel_size=3, rate=1, depth_activation=True)
    x = sepconv(x=x, filters=c4_size[3], prefix='conv4_decoder2', stride=1, kernel_size=3, rate=1, depth_activation=True)

    # C3 : Output stride = 8
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = Conv2D(filters=c3_size[3], kernel_size=(1 ,1), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
    x = Concatenate()([x, c3])
    x = sepconv(x=x, filters=c3_size[3], prefix='conv3_decoder1', stride=1, kernel_size=3, rate=1, depth_activation=True)
    x = sepconv(x=x, filters=c3_size[3], prefix='conv3_decoder2', stride=1, kernel_size=3, rate=1, depth_activation=True)

    # C2 : Output stride = 4
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = Conv2D(filters=c2_size[3], kernel_size=(1 ,1), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
    x = Concatenate()([x, c2])
    x = sepconv(x=x, filters=c2_size[3], prefix='conv2_decoder1', stride=1, kernel_size=3, rate=1, depth_activation=True)
    x = sepconv(x=x, filters=c2_size[3], prefix='conv2_decoder2', stride=1, kernel_size=3, rate=1, depth_activation=True)

    # C1 : Output stride = 2
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = Conv2D(filters=c1_size[3], kernel_size=(1 ,1), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
    x = Concatenate()([x, c1])
    x = sepconv(x=x, filters=c1_size[3], prefix='conv1_decoder1', stride=1, kernel_size=3, rate=1, depth_activation=True)
    x = sepconv(x=x, filters=c1_size[3], prefix='conv1_decoder2', stride=1, kernel_size=3, rate=1, depth_activation=True)

    # Classifier
    x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    pos = classifier(x, output_filters=output_channels, kernel_size=3, activation=None, use_dropout=0.0, prefix='pos')
    cos = classifier(x, output_filters=output_channels, kernel_size=3, activation=None, use_dropout=0.0, prefix='cos')
    sin = classifier(x, output_filters=output_channels, kernel_size=3, activation=None, use_dropout=0.0, prefix='sin')
    width = classifier(x, output_filters=output_channels, kernel_size=3, activation=None, use_dropout=0.0, prefix='width')

    
    return inputs, [pos, cos, sin, width]
