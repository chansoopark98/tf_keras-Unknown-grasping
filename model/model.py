from pickle import FALSE
from cv2 import multiply
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    UpSampling2D, Activation, BatchNormalization, Conv2D,  Concatenate, LeakyReLU, MaxPooling2D, Input, Flatten, Dense, Dropout, concatenate, Reshape,
    DepthwiseConv2D,  ZeroPadding2D, Conv2DTranspose, GlobalAveragePooling2D, Add)
from tensorflow.keras.activations import tanh, relu
import tensorflow as tf
from tensorflow.keras.models import Model
from torch import conv2d
from .efficientnetv2 import efficientnet_v2
from model.utils import sepconv, aspp, conv3x3, res_block, deconv3x3, classifier
from tensorflow.keras.initializers import HeNormal

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

class RFNet:
    def __init__(self, input_shape, output_channels):
        self.input_shape = input_shape
        self.output_channels = output_channels
        self.kernel_init = HeNormal()

    def stem_block(self, feature):
        x = Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding='same', kernel_initializer=self.kernel_init)(feature)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

        return x

    def res_block(self, feature, output_channel, stride):
        x = Conv2D(filters=output_channel, kernel_size=(3, 3), strides=(stride, stride), padding='same', use_bias=False, kernel_initializer=self.kernel_init)(feature)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=output_channel, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=self.kernel_init)(x)
        x = BatchNormalization()(x)
        skip = x

        # 2
        x = Conv2D(filters=output_channel, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=self.kernel_init)(skip)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=output_channel, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=self.kernel_init)(x)
        x = BatchNormalization()(x)
        output = x + skip
        
        x = Activation('relu')(output)
        return x, output

    def attention_block(self, feature, output_channel):
        x = GlobalAveragePooling2D()(feature)

        x_shape = tf.keras.backend.int_shape(x)
        # from (b_size, channels)->(b_size, 1, 1, channels)
        x = Reshape((1, 1, x_shape[1]))(x)
        x = Conv2D(output_channel, (1, 1), padding='same', kernel_initializer=self.kernel_init)(x)
        x = Activation('sigmoid')(x)
        x *= feature

        return x

    def upsample_block(self, x, skip, output_channel):
        # 1x1 convolution skip
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=output_channel, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=self.kernel_init)(x)

        skip_size = tf.keras.backend.int_shape(skip)
        x = tf.keras.layers.experimental.preprocessing.Resizing(
            *skip_size[1:3], interpolation="bilinear"
        )(x)

        x += skip

        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=output_channel, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=self.kernel_init)(x)

        return x


    def build(self):
        inputs = Input(shape=(self.input_shape[0], self.input_shape[1], self.input_shape[2]), name='model_input')
        
        rgb_input = inputs[:, :, :, :3]
        depth_input = inputs[:, :, :, 3:]
        
        # stem os=1/4
        rgb = self.stem_block(feature=rgb_input)
        depth = self.stem_block(feature=depth_input)

        # conv1 os=1/4
        rgb, skip_1 = self.res_block(feature=rgb, output_channel=64, stride=1)
        depth, _ = self.res_block(feature=depth, output_channel=64, stride=1)
        rgb = self.attention_block(feature=rgb, output_channel=64)
        depth = self.attention_block(feature=depth, output_channel=64)
        rgb += depth

        # conv2 os=1/8
        rgb, skip_2 = self.res_block(feature=rgb, output_channel=128, stride=2)
        depth, _ = self.res_block(feature=depth, output_channel=128, stride=2)
        rgb = self.attention_block(feature=rgb, output_channel=128)
        depth = self.attention_block(feature=depth, output_channel=128)
        rgb += depth

        # conv3 os=1/16
        rgb, skip_3 = self.res_block(feature=rgb, output_channel=256, stride=2)
        depth, _ = self.res_block(feature=depth, output_channel=256, stride=2)
        rgb = self.attention_block(feature=rgb, output_channel=256)
        depth = self.attention_block(feature=depth, output_channel=256)
        rgb += depth

        # conv4 os=1/32
        rgb, _ = self.res_block(feature=rgb, output_channel=512, stride=2)
        depth, _ = self.res_block(feature=depth, output_channel=512, stride=2)
        rgb = self.attention_block(feature=rgb, output_channel=512)
        depth = self.attention_block(feature=depth, output_channel=512)
        rgb += depth

        # ASPP
        rgb = aspp(x=rgb, activation='relu', kernel_init=self.kernel_init)

        # upsample
        output = self.upsample_block(x=rgb, skip=skip_3, output_channel=256) # os32 to os16
        output = self.upsample_block(x=output, skip=skip_2, output_channel=128) # os16 to os8
        output = self.upsample_block(x=output, skip=skip_1, output_channel=64) # os8 to os4
        
        

        skip_size = tf.keras.backend.int_shape(inputs)
        output = tf.keras.layers.experimental.preprocessing.Resizing(
            *skip_size[1:3], interpolation="bilinear"
        )(output)

        output = conv3x3(x=output, filters=32, kernel_size=3, strides=1, padding='same', kernel_init=self.kernel_init, activation='relu', prefix='output_conv')

        pos = classifier(output, output_filters=1, kernel_size=3, activation=None, use_dropout=0.0, prefix='pos', kernel_init=self.kernel_init)
        cos = classifier(output, output_filters=1, kernel_size=3, activation=None, use_dropout=0.0, prefix='cos', kernel_init=self.kernel_init)
        sin = classifier(output, output_filters=1, kernel_size=3, activation=None, use_dropout=0.0, prefix='sin', kernel_init=self.kernel_init)
        width = classifier(output, output_filters=1, kernel_size=3, activation=None, use_dropout=0.0, prefix='width', kernel_init=self.kernel_init)

        outputs = tf.concat([pos, cos, sin, width], axis=-1)
        # [pos, cos, sin, width]
        return inputs, outputs



        


        

        
        

    
    


        



    
