import tensorflow as tf
from tensorflow.keras.layers import ZeroPadding2D, Activation, DepthwiseConv2D, BatchNormalization, Conv2D, Concatenate, GlobalAveragePooling2D, Reshape, Dropout

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

def aspp(x, activation='swish', MOMENTUM=0.99, EPSILON=0.001):
    # Image Feature branch
    shape_before = tf.shape(x)
    b4 = GlobalAveragePooling2D()(x)
    b4_shape = tf.keras.backend.int_shape(b4)
    # from (b_size, channels)->(b_size, 1, 1, channels)
    b4 = Reshape((1, 1, b4_shape[1]))(b4)
    b4 = Conv2D(256, (1, 1), padding='same',
                kernel_initializer='he_normal',
                use_bias=False, name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling_BN', epsilon=EPSILON, momentum=MOMENTUM)(b4)
    b4 = Activation(activation)(b4)
    # upsample. have to use compat because of the option align_corners
    size_before = tf.keras.backend.int_shape(x)
    b4 = tf.keras.layers.experimental.preprocessing.Resizing(
            *size_before[1:3], interpolation="bilinear"
        )(b4)

    # b4 = UpSampling2D(size=(32, 64), interpolation="bilinear")(b4)
    # simple 1x1
    b0 = Conv2D(256, (1, 1), padding='same',
                kernel_initializer='he_normal',
                use_bias=False, name='aspp0')(x)
    # b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = BatchNormalization(name='aspp0_BN', epsilon=EPSILON, momentum=MOMENTUM)(b0)
    b0 = Activation(activation, name='aspp0_activation')(b0)

    b1 = sepconv(x, 256, 'aspp1',
                    rate=6, depth_activation=True, epsilon=EPSILON)
    # rate = 12 (24)
    b2 = sepconv(x, 256, 'aspp2',
                    rate=12, depth_activation=True, epsilon=EPSILON)
    # rate = 18 (36)
    b3 = sepconv(x, 256, 'aspp3',
                    rate=18, depth_activation=True, epsilon=EPSILON)

    # concatenate ASPP branches & project
    x = Concatenate()([b4, b0, b1, b2, b3])

    x = Conv2D(256, (1, 1), padding='same',
               kernel_initializer='he_normal',
               use_bias=False, name='concat_projection')(x)
    # x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = BatchNormalization(name='concat_projection_BN', epsilon=EPSILON, momentum=MOMENTUM)(x)
    x = Activation(activation)(x)

    x = Dropout(0.5)(x)

    return x