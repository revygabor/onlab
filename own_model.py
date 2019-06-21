from keras.models import Model
from keras.layers import Input
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization


def create_network(n_classes):
    input_layer = Input((None, None, 3))

    encoded_layer = create_encoding_layers(input_layer)
    decoded_layer = create_decoding_layers(encoded_layer)

    final_layer = Conv2D(n_classes, 1, padding='same')(decoded_layer)
    final_layer = Activation('softmax')(final_layer)

    semseg_model = Model(inputs=input_layer, outputs=final_layer)

    return semseg_model


def create_encoding_layers(input_layer):
    kernel = 3
    filter_size = 32
    pool_size = 2

    x = Conv2D(filter_size, kernel, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(pool_size, pool_size))(x)

    x = Conv2D(64, kernel, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(pool_size, pool_size))(x)

    x = Conv2D(128, kernel, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(pool_size, pool_size))(x)

    x = Conv2D(256, kernel, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


def create_decoding_layers(input_layer):
    kernel = 3
    filter_size = 32
    pool_size = 2

    x = Conv2D(256, kernel, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = UpSampling2D(size=(pool_size, pool_size))(x)
    x = Conv2D(128, kernel, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = UpSampling2D(size=(pool_size, pool_size))(x)
    x = Conv2D(64, kernel, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = UpSampling2D(size=(pool_size, pool_size))(x)
    x = Conv2D(filter_size, kernel, padding='same')(x)
    x = BatchNormalization()(x)

    return x
