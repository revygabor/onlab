# from functools import partial
# from typing import List
#
# import tensorflow as tf
# from keras import Model
# from keras.layers import Input, Conv2D, Add, BatchNormalization, ReLU, UpSampling2D, Concatenate, Softmax
# from keras.utils.vis_utils import model_to_dot
#
# BN = partial(BatchNormalization, momentum=0.01, epsilon=1e-05)
# Conv2D = partial(Conv2D, use_bias=False, padding='same')
# UpSampling2D = partial(UpSampling2D, interpolation='bilinear')
#
#
# def tensor_out_channels(x: tf.Tensor) -> int:
#     """
#     Returns the number of channels of the tensor (last dimension value)
#     :param x: tensor
#     :return: number of channels
#     """
#     return x.shape[-1].value
#
#
# def bottleneck(input_layer: tf.Tensor, filters: List[int] = [64, 64, 256]) -> tf.Tensor:
#     """
#     Creates the bottleneck block in HRNet
#     :param input_layer: input of the bottleneck
#     :param filters: number of convolution filters in the bottleneck Conv2D layers
#     :return: output layer of the bottleneck
#     """
#     if tensor_out_channels(input_layer) != 256:
#         residual = Conv2D(filters=256, kernel_size=1)(input_layer)
#         residual = BN()(residual)
#     else:
#         residual = input_layer
#
#     x = input_layer
#     x = Conv2D(filters=filters[0], kernel_size=1)(x)
#     x = BN()(x)
#     x = ReLU()(x)
#     x = Conv2D(filters=filters[1], kernel_size=3)(x)
#     x = BN()(x)
#     x = ReLU()(x)
#     x = Conv2D(filters=filters[2], kernel_size=1)(x)
#     x = BN()(x)
#
#     x = Add()([residual, x])
#     x = ReLU()(x)
#
#     return x
#
#
# def stack_bottleneck_blocks(x: tf.Tensor, repeats: int = 4, filters: List[int] = [64, 64, 256]):
#     """
#     Repeats bottleneck blocks 'repeat' times for the input x
#     :param x: input of the bottleneck stack
#     :param repeats: the bottleneck block must be repeated this as many times
#     :param filters: number of filters in the
#     :return: output of the stacked bottleneck blocks
#     """
#     for i in range(repeats):
#         x = bottleneck(x, filters=filters)
#     return x
#
#
# def transition(input_layers: List[tf.Tensor], n_filter_min: int):
#     if len(input_layers) == 1:
#         branch1 = Conv2D(filters=n_filter_min, kernel_size=3, strides=1)(input_layers[0])
#         branch1 = BN()(branch1)
#         branch1 = ReLU()(branch1)
#
#         branch2 = Conv2D(filters=2 * n_filter_min, kernel_size=3, strides=2)(input_layers[0])
#         branch2 = BN()(branch2)
#         branch2 = ReLU()(branch2)
#
#         return [branch1, branch2]
#
#     else:
#         n_branches = len(input_layers) + 1
#         new_branch = Conv2D(filters=n_filter_min * pow(2, n_branches - 1), kernel_size=3, strides=2)(input_layers[-1])
#         new_branch = BN()(new_branch)
#         new_branch = ReLU()(new_branch)
#
#         input_layers.append(new_branch)
#
#     return input_layers
#
#
# def basic_block(input_layer: tf.Tensor):
#     """
#     Creates the basic block for HRNet (Conv, BN, ReLu, Conv, BN, Add(residual), ReLU)
#     :param input_layer: input layer of the basic block
#     :return: output layer of basic block
#     """
#     n_filters = tensor_out_channels(input_layer)
#     x = Conv2D(filters=n_filters, kernel_size=3)(input_layer)
#     x = BN()(x)
#     x = ReLU()(x)
#     x = Conv2D(filters=n_filters, kernel_size=3)(input_layer)
#     x = BN()(x)
#     x = ReLU()(x)
#
#     x = Add()([input_layer, x])
#     x = ReLU()(x)
#
#     return x
#
#
# def stack_basic_blocks(input_layer: tf.Tensor, n_basic_blocks: int):
#     """
#     Creates a branch by stacking basic blocks on top of each other
#     :param input_layer: input layer of the branch
#     :param n_basic_blocks: the number of blocks in the branch
#     :return: output layer of the branch
#     """
#     x = input_layer
#     for i in range(n_basic_blocks):
#         x = basic_block(x)
#
#     return x
#
#
# def fuse_layers(input_layers: List[tf.Tensor]) -> List[tf.Tensor]:
#     """
#     Fuses info from every branches to all branches
#     :param input_layers: all branches to fuse
#     :return: fused branches with same shape as provided in input_layers
#     """
#     fusion = []
#
#     n_layers = len(input_layers)
#     for current_layer_idx in range(n_layers):
#         current_fusion = []
#
#         for layer_to_fuse_idx in range(n_layers):
#             n_filters_current_layer = tensor_out_channels(input_layers[current_layer_idx])
#             n_filters_layer_to_fuse = tensor_out_channels(input_layers[layer_to_fuse_idx])
#
#             # the layer to fuse has bigger resolution -> stride=2 conv2,
#
#             if current_layer_idx > layer_to_fuse_idx:
#
#                 x = input_layers[layer_to_fuse_idx]
#
#                 for i in range(current_layer_idx - layer_to_fuse_idx)[::-1]:
#                     # change filter number at the last conv:
#                     filters = n_filters_layer_to_fuse if i != 0 else n_filters_current_layer
#
#                     x = Conv2D(filters, kernel_size=3, strides=2)(x)
#                     x = BN()(x)
#                     if i != 0:
#                         x = ReLU()(x)  # no need for ReLU before Add[]
#
#                 current_fusion.append(x)
#
#             # current layer is fused (added) without modification
#             elif current_layer_idx == layer_to_fuse_idx:
#                 current_fusion.append(input_layers[layer_to_fuse_idx])
#
#             # the layer to fuse has smaller resolution -> upscale bilinear, conv to change filters
#             else:  # current_layer_idx < layer_to_fuse_idx
#                 x = input_layers[layer_to_fuse_idx]
#                 x = UpSampling2D(pow(2, layer_to_fuse_idx - current_layer_idx)
#                                  # name='UpS_{}_to_{}'.format(layer_to_fuse_idx, current_layer_idx))(x)
#                                  )(x)
#                 x = Conv2D(n_filters_current_layer, kernel_size=1)(x)
#                 x = BN()(x)
#                 current_fusion.append(x)
#
#         # x = Add(name='fuse_to_{}'.format(current_layer_idx))(current_fusion)
#         x = Add()(current_fusion)
#         x = ReLU()(x)
#
#         fusion.append(x)
#
#     return fusion
#
#
# def concat_branches(input_layers: List[tf.Tensor]):
#     """
#     Concatenates different resolution branches by upscaling them
#     :param input_layers: input layers to concatenate
#     :return: concatenated layers with same resolution
#     """
#     same_resolution_branches = []
#     for i in range(len(input_layers)):
#         same_resolution_branches.append(
#             input_layers[i] if i == 0
#             else UpSampling2D(pow(2, i))(input_layers[i]))
#
#     x = Concatenate()(same_resolution_branches)
#
#     return x
#
#
# def create_classificator(input_layers: List[tf.Tensor], n_classes: int):
#     x = [stack_basic_blocks(l, n_basic_blocks=4) for l in input_layers]
#     x = fuse_layers(x)
#     x = [stack_basic_blocks(l, n_basic_blocks=4) for l in x]
#     x = fuse_layers(x)
#     x = concat_branches(x)
#
#     x = Conv2D(filters=tensor_out_channels(x), kernel_size=1)(x)
#     x = BN()(x)
#     x = ReLU()(x)
#     x = Conv2D(filters=n_classes, kernel_size=1)(x)
#
#     return x
#
#
# def create_network(input_resolution=(None, None, 3), n_classes=34):
#     input = Input(shape=(*input_resolution, 3))
#     # input = Input(shape=(None, None, 3))
#
#     x = Conv2D(filters=64, kernel_size=3, strides=2)(input)
#     x = BN()(x)
#     x = Conv2D(filters=64, kernel_size=3, strides=2)(x)
#     x = BN()(x)
#     x = ReLU()(x)
#
#     stage1 = stack_bottleneck_blocks(x)
#
#     n_filter_min = 48
#     stage2 = transition([stage1], n_filter_min=n_filter_min)
#     stage2 = [stack_basic_blocks(layer, n_basic_blocks=4) for layer in stage2]
#     stage2 = fuse_layers(stage2)
#
#     stage3 = transition(stage2, n_filter_min=n_filter_min)
#     stage3 = [stack_basic_blocks(layer, n_basic_blocks=4) for layer in stage3]
#     stage3 = fuse_layers(stage3)
#
#     stage4 = transition(stage3, n_filter_min=n_filter_min)
#     stage4 = [stack_basic_blocks(layer, n_basic_blocks=4) for layer in stage4]
#     stage4 = fuse_layers(stage4)
#
#     classificator = create_classificator(stage4, n_classes)
#     x = UpSampling2D(size=4)(classificator)
#     output = Softmax()(x)
#
#     model = Model(inputs=input, outputs=output)
#     return model
#
#
# if __name__ == '__main__':
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#     session = tf.Session(config=config)
#
#     model = create_network(input_resolution=(None, None), n_classes=20)
#     model.summary()
#
#     # import os
#     # print(os.environ['PATH'])
#     # os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'
#     # print('\n\n\n')
#     # print(os.environ['PATH'])
#
#     # model_to_dot(model, show_shapes=True).write_svg('model.svg')
#
#
#


from typing import List

import tensorflow as tf


def leaky_relu(input, name=None):
    return tf.keras.layers.LeakyReLU(alpha=0.1, name=None if name is None else name + '_relu')(input)


def conv_2d(inputs, filters, kernel_size=3, strides=1, batch_normalization=True,
            activation=None, name=None, padding='same',
            kernel_initializer=tf.random_normal_initializer(stddev=0.01)):
    output = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                    strides=strides, padding=padding, name=None if name is None else name + '_conv',
                                    kernel_initializer=kernel_initializer,
                                    # use_bias=False
                                    )(inputs)
    name = None if name is None else name + '_conv'

    if batch_normalization:
        output = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9,
                                                    name=None if name is None else name + '_bn')(output)

    if activation:
        output = activation(output, name=name if name is not None else None)

    return output


def bottleneck(input_layer, filters: List[int] = [64, 64, 256], name='bottleneck') -> tf.Tensor:
    """
    Creates the bottleneck block in HRNet
    :param name: name of the block
    :param input_layer: input of the bottleneck
    :param filters: number of convolution filters in the bottleneck Conv2D layers
    :return: output layer of the bottleneck
    """
    _, _, _, c = input_layer.get_shape()
    if c != 256:
        residual = conv_2d(inputs=input_layer, filters=256, kernel_size=1, name=name + '_residual')
    else:
        residual = input_layer

    x = input_layer
    x = conv_2d(inputs=x, filters=filters[0], kernel_size=1, activation=leaky_relu, name=name + '_conv_1x1_0')

    x = conv_2d(inputs=x, filters=filters[1], kernel_size=3, activation=leaky_relu, name=name + '_conv_3x3')

    x = conv_2d(inputs=x, filters=filters[2], kernel_size=1, name=name + '_conv_1x1_1')

    x = tf.keras.layers.Add(name=name + '_add')([residual, x])
    x = leaky_relu(x, name + '_leakyReLU')

    return x


def stack_bottleneck_blocks(x: tf.Tensor, repeats: int = 4, filters: List[int] = [64, 64, 256], name='bottleneck'):
    """
    Repeats bottleneck blocks 'repeat' times for the input x
    :param name: name of the layers
    :param x: input of the bottleneck stack
    :param repeats: the bottleneck block must be repeated this as many times
    :param filters: number of filters in the
    :return: output of the stacked bottleneck blocks
    """
    for i in range(repeats):
        x = bottleneck(x, filters=filters, name=name+str(i))
    return x


def transition(input_layers: List[tf.Tensor], n_filter_min: int):
    if len(input_layers) == 1:
        branch1 = conv_2d(inputs=input_layers[0], filters=n_filter_min, kernel_size=3,
                          strides=1, activation=leaky_relu, name='transition0_0')

        branch2 = conv_2d(inputs=input_layers[0], filters=2 * n_filter_min, kernel_size=3,
                          strides=2, activation=leaky_relu, name='transition0_1')

        return [branch1, branch2]

    else:
        n_branches = len(input_layers) + 1
        new_branch = conv_2d(inputs=input_layers[-1], filters=n_filter_min * pow(2, n_branches - 1),
                             kernel_size=3, strides=2, activation=leaky_relu,
                             name='transition{}'.format(len(input_layers)-1))

        input_layers.append(new_branch)

    return input_layers


def basic_block(input_layer: tf.Tensor, name):
    """
    Creates the basic block for HRNet (Conv, BN, ReLu, Conv, BN, Add(residual), ReLU)
    :param name: name of the layers in the block
    :param input_layer: input layer of the basic block
    :return: output layer of basic block
    """
    _, _, _, n_filters = input_layer.get_shape()
    x = conv_2d(inputs=input_layer, filters=n_filters, kernel_size=3, activation=leaky_relu, name=name+'_0')
    x = conv_2d(inputs=x, filters=n_filters, kernel_size=3, name=name+'_1')

    x = tf.keras.layers.Add(name=name+'_add')([input_layer, x])
    x = leaky_relu(x, name=name+'leaky_ReLU')

    return x


def stack_basic_blocks(input_layer: tf.Tensor, n_basic_blocks: int, name):
    """
    Creates a branch by stacking basic blocks on top of each other
    :param input_layer: input layer of the branch
    :param n_basic_blocks: the number of blocks in the branch
    :return: output layer of the branch
    """
    x = input_layer
    for i in range(n_basic_blocks):
        x = basic_block(x, name=None if name is None else name+'_{}'.format(i))

    return x


def fuse_layers(input_layers: List[tf.Tensor]) -> List[tf.Tensor]:
    """
    Fuses info from every branches to all branches
    :param input_layers: all branches to fuse
    :return: fused branches with same shape as provided in input_layers
    """
    fusion = []

    n_layers = len(input_layers)
    for current_layer_idx in range(n_layers):
        current_fusion = []

        for layer_to_fuse_idx in range(n_layers):
            _, _, _, n_filters_current_layer = input_layers[current_layer_idx].get_shape()
            _, _, _, n_filters_layer_to_fuse = input_layers[layer_to_fuse_idx].get_shape()

            # the layer to fuse has bigger resolution -> stride=2 conv2,

            if current_layer_idx > layer_to_fuse_idx:

                x = input_layers[layer_to_fuse_idx]

                for i in range(current_layer_idx - layer_to_fuse_idx)[::-1]:
                    # change filter number at the last conv:
                    filters = n_filters_layer_to_fuse if i != 0 else n_filters_current_layer

                    x = conv_2d(inputs=x, filters=filters, kernel_size=3, strides=2,
                                activation=leaky_relu if i != 0 else None)  # no need for ReLU before Add[]

                current_fusion.append(x)

            # current layer is fused (added) without modification
            elif current_layer_idx == layer_to_fuse_idx:
                current_fusion.append(input_layers[layer_to_fuse_idx])

            # the layer to fuse has smaller resolution -> upscale bilinear, conv to change filters
            else:  # current_layer_idx < layer_to_fuse_idx
                x = input_layers[layer_to_fuse_idx]
                x = tf.keras.layers.UpSampling2D(pow(2, layer_to_fuse_idx - current_layer_idx),
                                                 interpolation='bilinear'
                                                 # name='UpS_{}_to_{}'.format(layer_to_fuse_idx, current_layer_idx))(x)
                                                 )(x)
                x = conv_2d(inputs=x, filters=n_filters_current_layer, kernel_size=1)
                current_fusion.append(x)

        # x = Add(name='fuse_to_{}'.format(current_layer_idx))(current_fusion)
        x = tf.keras.layers.Add()(current_fusion)
        x = leaky_relu(input=x)

        fusion.append(x)

    return fusion


def concat_branches(input_layers: List[tf.Tensor]):
    """
    Concatenates different resolution branches by upscaling them
    :param input_layers: input layers to concatenate
    :return: concatenated layers with same resolution
    """
    same_resolution_branches = []
    for i in range(len(input_layers)):
        same_resolution_branches.append(
            input_layers[i] if i == 0
            else tf.keras.layers.UpSampling2D(pow(2, i), interpolation='bilinear')(input_layers[i]))

    x = tf.keras.layers.Concatenate()(same_resolution_branches)

    return x


def create_classificator(input_layers: List[tf.Tensor], n_classes: int):
    x = [stack_basic_blocks(l, n_basic_blocks=4, name='class_bblock_0_{}'.format(i)) for i, l in enumerate(input_layers)]
    x = fuse_layers(x)
    x = [stack_basic_blocks(l, n_basic_blocks=4, name='class_bblock_1_{}'.format(i)) for i, l in enumerate(x)]
    x = fuse_layers(x)
    x = concat_branches(x)

    _, _, _, c = x.get_shape()

    x = conv_2d(inputs=x, filters=c, kernel_size=1)
    x = conv_2d(inputs=x, filters=n_classes, kernel_size=1)

    return x


def create_network(input_resolution=(None, None), n_classes=34):
    input = tf.keras.layers.Input(shape=(*input_resolution, 3))

    x = conv_2d(inputs=input, filters=64, kernel_size=3, strides=2, name='conv1')
    x = conv_2d(inputs=x, filters=64, kernel_size=3, strides=2, activation=leaky_relu, name='conv2')

    stage1 = stack_bottleneck_blocks(x, name='stage_1_bneck')

    n_filter_min = 48
    stage2 = transition([stage1], n_filter_min=n_filter_min)
    stage2 = [stack_basic_blocks(layer, n_basic_blocks=4, name='bblock_0_{}'.format(i)) for i, layer in enumerate(stage2)]
    stage2 = fuse_layers(stage2)

    stage3 = transition(stage2, n_filter_min=n_filter_min)
    stage3 = [stack_basic_blocks(layer, n_basic_blocks=4, name='bblock_1_{}'.format(i)) for i, layer in enumerate(stage3)]
    stage3 = fuse_layers(stage3)

    stage4 = transition(stage3, n_filter_min=n_filter_min)
    stage4 = [stack_basic_blocks(layer, n_basic_blocks=4, name='bblock_2_{}'.format(i)) for i, layer in enumerate(stage4)]
    stage4 = fuse_layers(stage4)

    classificator = create_classificator(stage4, n_classes)
    x = tf.keras.layers.UpSampling2D(size=4, interpolation='bilinear')(classificator)
    output = tf.keras.layers.Softmax()(x)

    model = tf.keras.models.Model(inputs=input, outputs=output)
    return model


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    model = create_network(input_resolution=(None, None), n_classes=20)
    model.summary()
