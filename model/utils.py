import tensorflow as tf
from tensorflow.keras.layers import ZeroPadding2D, Activation, DepthwiseConv2D, BatchNormalization, Conv2D, Concatenate, GlobalAveragePooling2D, Reshape, Dropout, Conv2DTranspose, Add



def conv3x3(x, filters, kernel_size, strides, padding,
            bn_momentum=0.99, bn_epsilon=0.001, activation='relu', prefix='name', rate=1, kernel_init='he_normal'):
    x = Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size), strides=(strides, strides), kernel_initializer=kernel_init, padding=padding, name=prefix+'_conv2d')(x)
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

def classifier(x, output_filters, kernel_size, activation, use_dropout, prefix, kernel_init):
    if use_dropout != 0.0:
        x = Dropout(use_dropout)(x)
    x = Conv2D(filters=output_filters, kernel_size=(kernel_size, kernel_size), strides=(1, 1),
                activation=activation, kernel_initializer=kernel_init, padding='same', name=prefix+'_classifier')(x)
    return x


def sepconv(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    activation = 'swish'
    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = Activation(activation)(x)
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, kernel_initializer='he_normal', name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation(activation)(x)
    x = Conv2D(filters, (1, 1), padding='same',
               use_bias=False, name=prefix + '_pointwise', kernel_initializer='he_normal')(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation(activation)(x)

    return x

def aspp(x, activation='relu', kernel_init='he_normal'):
    # Image Feature branch
    shape_before = tf.shape(x)
    b4 = GlobalAveragePooling2D()(x)
    b4_shape = tf.keras.backend.int_shape(b4)
    # from (b_size, channels)->(b_size, 1, 1, channels)
    b4 = Reshape((1, 1, b4_shape[1]))(b4)
    b4 = Conv2D(256, (1, 1), padding='same',
                kernel_initializer=kernel_init,
                use_bias=False, name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling_BN')(b4)
    b4 = Activation(activation)(b4)
    # upsample. have to use compat because of the option align_corners
    size_before = tf.keras.backend.int_shape(x)
    b4 = tf.keras.layers.experimental.preprocessing.Resizing(
            *size_before[1:3], interpolation="bilinear"
        )(b4)

    # b4 = UpSampling2D(size=(32, 64), interpolation="bilinear")(b4)
    # simple 1x1
    b0 = Conv2D(256, (1, 1), padding='same',
                kernel_initializer=kernel_init,
                use_bias=False, name='aspp0')(x)
    # b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = BatchNormalization(name='aspp0_BN')(b0)
    b0 = Activation(activation, name='aspp0_activation')(b0)

    b1 = conv3x3(x=x, filters=256, kernel_size=3, strides=1, padding='same', prefix='aspp_b1', rate=6, kernel_init=kernel_init)
    # rate = 12 (24)
    b2 = conv3x3(x=x, filters=256, kernel_size=3, strides=1, padding='same', prefix='aspp_b2', rate=12, kernel_init=kernel_init)
    # rate = 18 (36)
    b3 = conv3x3(x=x, filters=256, kernel_size=3, strides=1, padding='same', prefix='aspp_b3', rate=18, kernel_init=kernel_init)

    # concatenate ASPP branches & project
    x = Concatenate()([b4, b0, b1, b2, b3])

    x = Conv2D(256, (1, 1), padding='same',
               kernel_initializer=kernel_init,
               use_bias=False, name='concat_projection')(x)
    # x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = BatchNormalization(name='concat_projection_BN')(x)
    x = Activation(activation)(x)

    return x