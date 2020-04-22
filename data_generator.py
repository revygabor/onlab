import tensorflow as tf

from typing import List, Union, Tuple
from functools import partial
from glob import glob

from config import *


@tf.function
def read_image(path: str):
    input_image = tf.io.read_file(path)
    input_image = tf.io.decode_image(input_image, expand_animations=False)

    return input_image


@tf.function
def normalize_image(input_image: tf.Tensor):
    return tf.cast(input_image, tf.float32) / 255.0


def squeeze_label(label: tf.Tensor):
    # removes last (unnecessary) axis
    return tf.squeeze(label, axis=-1)


def resize_input_image(input_image: tf.Tensor, size: Tuple[int, int]):
    return tf.image.resize(
        images=input_image,
        size=size,
        method=tf.image.ResizeMethod.BILINEAR
    )


def resize_label(label_image: tf.Tensor, size: Tuple[int, int]):
    label_image = tf.image.resize(
        images=label_image,
        size=size,
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )

    return label_image


# @tf.function
# def augment_images(input_image: tf.Tensor, label_image: tf.Tensor):
#     uniform_random = tf.random.uniform([], 0, 1.0)
#     flip_cond = tf.less(uniform_random, .5)
#     input_image = tf.cond(flip_cond, lambda: tf.reverse(input_image, axis=[1]), lambda: input_image)
#     label_image = tf.cond(flip_cond, lambda: tf.reverse(label_image, axis=[1]), lambda: label_image)
#
#     # TODO
#
#     return input_image, label_image


def to_one_hot(label: tf.Tensor, n_classes: int):
    return tf.one_hot(label, depth=n_classes)


def create_training_data_lists(inputs_file_pattern, labels_file_pattern, frames, shift, stride):
    input_filenames = glob(inputs_file_pattern)
    label_filenames = glob(labels_file_pattern)

    input_filenames.sort()
    label_filenames.sort()

    n_items_to_cut_from_end = (len(input_filenames) - (((frames - 1) * stride) + 1)) % stride
    if n_items_to_cut_from_end > 0:
        input_filenames = input_filenames[:-n_items_to_cut_from_end]

    first_label_index = (frames - 1) * stride
    label_filenames = label_filenames[first_label_index::shift]

    return input_filenames, label_filenames


def make_window_dataset(ds, window_size=5, shift=1, stride=1):
    windows = ds.window(window_size, shift=shift, stride=stride)

    def sub_to_batch(sub):
        return sub.batch(window_size, drop_remainder=True)

    windows = windows.flat_map(sub_to_batch)
    return windows


def create_dataset(inputs_file_pattern: str,
                   labels_file_pattern: str,
                   n_classes: int,
                   frames: int, shift: int = None, stride: int = 1,
                   size: Tuple[int, int] = None,
                   cache: bool = True):
    input_filenames, label_filenames = \
        create_training_data_lists(inputs_file_pattern, labels_file_pattern,
                                   frames, shift, stride)

    input_filenames_ds = tf.data.Dataset.from_tensor_slices(input_filenames)
    label_filenames_ds = tf.data.Dataset.from_tensor_slices(label_filenames)

    input_dataset = input_filenames_ds.map(read_image, tf.data.experimental.AUTOTUNE)
    label_dataset = label_filenames_ds.map(read_image, tf.data.experimental.AUTOTUNE)

    input_dataset = input_dataset.map(normalize_image, tf.data.experimental.AUTOTUNE)

    if size is not None:
        input_resize_function = partial(resize_input_image, size=size)
        input_dataset = input_dataset.map(tf.function(input_resize_function), tf.data.experimental.AUTOTUNE)

        label_resize_function = partial(resize_label, size=size)
        label_dataset = label_dataset.map(tf.function(label_resize_function), tf.data.experimental.AUTOTUNE)

    label_dataset = label_dataset.map(squeeze_label, tf.data.experimental.AUTOTUNE)

    one_hot_function = partial(to_one_hot, n_classes=n_classes)
    label_dataset = label_dataset.map(tf.function(one_hot_function), tf.data.experimental.AUTOTUNE)

    # if cache:
    #     input_dataset = input_dataset.cache()
    #     label_dataset = label_dataset.cache()

    input_dataset = make_window_dataset(input_dataset, frames, shift, stride)

    dataset = tf.data.Dataset.zip((input_dataset, label_dataset))

    return dataset


if __name__ == '__main__':
    # x = create_dataset(inputs_file_pattern=[BDD100K_TRAIN_IMAGES, CITYSCAPES_TRAIN_IMAGES],
    #                    labels_file_pattern=[BDD100K_TRAIN_LABELS, CITYSCAPES_TRAIN_FINE_LABELS],
    #                    n_classes=19, frames=5, size=(50, 100))
    train_config = ApolloScapeConfig
    train_images_dirs = glob(train_config.train_image_dirs)
    train_images_dirs.sort()
    train_images = os.path.join(train_images_dirs[0], '*.jpg')

    train_labels_dirs = glob(train_config.train_label_dirs)
    train_labels_dirs.sort()
    train_labels = os.path.join(train_labels_dirs[0], '*.png')

    train_dataset = create_dataset(inputs_file_pattern=train_images,
                                   labels_file_pattern=train_labels,
                                   n_classes=16, size=(240, 320),
                                   batch_size=1, frames=5, shift=1, stride=1)

    from matplotlib import pyplot as plt
    import numpy as np

    for frames_b, labels_b in train_dataset.take(20):
        for frames in frames_b:
            for x in frames:
                plt.imshow(x)
                plt.show()
        for label in labels_b:
            plt.imshow(np.argmax(label, axis=-1))
            plt.show()
