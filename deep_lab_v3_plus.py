from keras.models import Model
from keras.layers import Input, add, Lambda, Concatenate, UpSampling2D
from keras.layers import Conv2D, SeparableConv2D, GlobalAveragePooling2D
from keras.layers import BatchNormalization, ReLU, Dropout, Activation
import tensorflow as tf
import keras.backend as K


def create_network(input_resolution, n_classes=34):
    # input_layer = Input(shape=(*input_resolution, 3))
    input_layer = Input(shape=(None, None, 3))

    # entry flow
    x = _Conv2D(input_layer, filters=32, kernel_size=3, stride=2,
                name='ef_conv32')

    x = _Conv2D(x, filters=64, kernel_size=3, stride=1,
                name='ef_conv64')

    x = xception_block(x, filter=128, last_stride=2, last_rate=1,
                       name='ef_x_block_1', residual_type='conv')

    x, skip = xception_block(x, filter=256, last_stride=2, last_rate=1,
                             name='ef_x_block_2', residual_type='conv', return_skip=True)

    x = xception_block(x, filter=728, last_stride=2, last_rate=1,
                       name='ef_x_block_3', residual_type='conv')

    # middle flow
    for i in range(16):
        x = xception_block(x, filter=728, last_stride=1, last_rate=1,
                           name='mf_x_block_{}'.format(i + 1), residual_type='add')

    # exit
    x = xception_block(x, [728, 1024, 1024], last_stride=1, last_rate=1,
                       name='xf_x_block_1', residual_type='conv')

    x = xception_block(x, [1536, 1536, 2048], last_stride=1, last_rate=1,
                       name='xf_x_block_2', residual_type='none')

    # atrous spatial pyramid pooling
    global_image_pooling_upsampling_factor = tuple(i / 16 for i in input_resolution)
    x = atrous_spatial_pyramid_pooling(x, global_image_pooling_upsampling_factor=global_image_pooling_upsampling_factor)

    # 1x1 conv after aspp
    x = _Conv2D(x, filters=256, kernel_size=1, name='1x1conv_after_aspp')
    x = Dropout(0.1)(x)

    # upsampling by 4
    x = UpSampling2D(size=4, interpolation='bilinear')(x)

    # reducing hypercolumn channels
    skip = _Conv2D(skip, filters=48, kernel_size=1, name='hypercolumn_conv48')

    # concat hypercolumn and high-level features
    x = Concatenate()([skip, x])

    # 2x sepConv 3x3
    x = SeparableConv2D(filters=256, kernel_size=3,
                        padding='same',
                        use_bias=False,
                        name='decoder_sepconv_1')(x)
    x = BatchNormalization(name='decoder_sepconv_1_bn', epsilon=1e-5)(x)
    x = ReLU()(x)

    x = SeparableConv2D(filters=256, kernel_size=3,
                        padding='same',
                        use_bias=False,
                        name='decoder_sepconv_2')(x)
    x = BatchNormalization(name='decoder_sepconv_2_bn', epsilon=1e-5)(x)
    x = ReLU()(x)

    # 1x1 conv reducing channels to class number
    x = Conv2D(filters=n_classes, kernel_size=1, padding='same', name='reduce_channels')(x)

    # upsampling by 4
    x = UpSampling2D(size=4, interpolation='bilinear')(x)

    # activation
    x = Activation('sigmoid')(x)

    model = Model(inputs=input_layer, outputs=x)

    return model


def _Conv2D(input_layer, filters, kernel_size, stride=1,
            name='', bn_epsilon=1e-3):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, padding='same',
               name=name, use_bias=False)(input_layer)
    x = BatchNormalization(name=name + '_bn', epsilon=bn_epsilon)(x)
    x = ReLU()(x)

    return x


def xception_block(input_layer, filter, last_stride, last_rate, name,
                   residual_type='conv', return_skip=False):
    if type(filter) is int:
        filters = [filter, filter, filter]
    else:
        filters = filter

    x = input_layer

    x = SeparableConv2D(filters[0], kernel_size=3,
                        padding='same',
                        use_bias=False,
                        name=name + '_sepconv_1')(x)
    x = BatchNormalization(name=name + '_sepconv_1_bn')(x)
    x = ReLU()(x)

    x = SeparableConv2D(filters[1], kernel_size=3,
                        padding='same',
                        use_bias=False,
                        name=name + '_sepconv_2')(x)
    x = BatchNormalization(name=name + '_sepconv_2_bn')(x)
    x = ReLU()(x)
    skip = x

    x = SeparableConv2D(filters[2], kernel_size=3,
                        strides=last_stride,
                        dilation_rate=last_rate,
                        padding='same',
                        use_bias=False,
                        name=name + '_atrous_sepconv')(x)
    x = BatchNormalization(name=name + '_atrous_sepconv_bn')(x)
    x = ReLU()(x)

    if residual_type == 'conv':
        res = Conv2D(filters=filters[2], kernel_size=1,
                     strides=last_stride,
                     padding='same',
                     use_bias=False,
                     name=name + "_residual")(input_layer)
        res = BatchNormalization(name=name + "_residual_bn")(res)
        # res = ReLU()(res)

        x = add([x, res])

    elif residual_type == 'add':
        x = add([x, input_layer])

    if return_skip:
        return x, skip

    return x


def atrous_spatial_pyramid_pooling(input_layer, global_image_pooling_upsampling_factor):
    # branch: 1x1 conv
    b_aspp_0 = _Conv2D(input_layer, filters=256, kernel_size=1,
                       name='aspp_0_conv', bn_epsilon=1e-5)

    # branch: 3x3 conv, rate 6
    b_aspp_1 = SeparableConv2D(filters=256, kernel_size=3,
                               padding='same',
                               dilation_rate=6,
                               use_bias=False,
                               name='aspp_1_sepconv')(input_layer)
    b_aspp_1 = BatchNormalization(name='aspp_1_sepconv_bn', epsilon=1e-5)(b_aspp_1)
    b_aspp_1 = ReLU()(b_aspp_1)

    # branch: 3x3 conv, rate 12
    b_aspp_2 = SeparableConv2D(filters=256, kernel_size=3,
                               padding='same',
                               dilation_rate=12,
                               use_bias=False,
                               name='aspp_2_sepconv')(input_layer)
    b_aspp_2 = BatchNormalization(name='aspp_2_sepconv_bn', epsilon=1e-5)(b_aspp_2)
    b_aspp_2 = ReLU()(b_aspp_2)

    # branch: 3x3 conv, rate 18
    b_aspp_3 = SeparableConv2D(filters=256, kernel_size=3,
                               padding='same',
                               dilation_rate=18,
                               use_bias=False,
                               name='pyramid_3x3sepconv')(input_layer)
    b_aspp_3 = BatchNormalization(name='pyramid_3x3sepconv_bn', epsilon=1e-5)(b_aspp_3)
    b_aspp_3 = ReLU()(b_aspp_3)

    # branch: image pooling
    # b_image_pooling = GlobalAveragePooling2D(name='pyramid_img_pool')(input_layer)
    # b_image_pooling = Lambda(lambda x: K.expand_dims(K.expand_dims(x, 1), 1))(
    #     b_image_pooling)  # (batch size x channels)->(batch size x 1 x 1 x channels)
    # b_image_pooling = Conv2D(filters=256, kernel_size=1, padding='same',
    #                          use_bias=False, name='pyramid_img_pool_conv')(b_image_pooling)
    # b_image_pooling = BatchNormalization(name='pyramid_img_pool_conv_bn')(b_image_pooling)
    # b_image_pooling = ReLU()(b_image_pooling)
    # b_image_pooling = UpSampling2D(global_image_pooling_upsampling_factor, interpolation='bilinear')(b_image_pooling)

    output_layer = Concatenate()([b_aspp_0, b_aspp_1, b_aspp_2, b_aspp_3])
    # output_layer = Concatenate()([b_aspp_0, b_aspp_1, b_aspp_2, b_aspp_3, b_image_pooling])

    return output_layer
