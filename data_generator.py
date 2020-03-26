# import os
# import glob
# import random
# from typing import List, Tuple
#
# import cv2
# import numpy as np
# from PIL import Image
# import imgaug as ia
# import imgaug.augmenters as iaa
# from matplotlib import pyplot as plt
# from keras.preprocessing import utils as keras_utils
#
#
# def one_hot_tensor_to_label(output_label):
#     """
#     Converts output tensor to labels format
#     :param output_label: output tensor from NN
#     :return: tensor with class numbers
#     """
#     return np.argmax(output_label, axis=-1)
#

# class DataSet:
#     def __init__(self, images_path: str, labels_path: str, n_classes: int, batch_size: int):
#         """
#         Initialize the DataSet class
#         ----------------------------
#         :param images_path: dir path to the images e.g. "leftImg8bit/train"; use os.path.join
#         :param labels_path: dir path to the labels e.g. "gtCoarse/train"; use os.path.join
#         :param n_classes: number of classes in the dataset
#         :param batch_size: batch size
#         """
#         self.cache = {}
#         self.images_dir = images_path
#         self.labels_dir = labels_path
#         self.n_classes = n_classes
#
#         images_list = glob.glob(images_path)
#         images_list.sort()
#         self.images_list = images_list
#
#         labels_list = glob.glob(labels_path)
#         labels_list.sort()
#         self.labels_list = labels_list
#
#         assert len(images_list) == len(labels_list), '#images and #labels must be equal'
#         self.data_size = len(images_list)
#         print("{} images found in directory {}".format(self.data_size, self.images_dir))
#         self.n_batches = self.data_size // batch_size
#         self.batch_size = batch_size
#
#     def generate_data(self, image_size: Tuple[int, int] = (600, 300), shuffle: bool = True,
#                       enable_caching: bool = False, enable_augmentation: bool = True):
#         """
#         Generator for the training, generates (x, y) pair
#         -------------------------------------------------
#         :param enable_augmentation: whether or not you want to enable input image augmentation
#         :param enable_caching: whether or not you want to enable to cache the input+output images
#         :param image_size: size to resize the image to
#         :param shuffle: whether to shuffle dataset after one epoch
#         :return: (x, y) where x: batch of input images, y: batch of one hot encoded pixel-level labels
#         """
#
#         if enable_augmentation:
#             ia.seed(5)
#             seq = iaa.Sequential([
#                 iaa.Fliplr(.5),
#                 iaa.Add((-25, 25)),
#                 iaa.AdditivePoissonNoise((0, 16)),
#                 iaa.Crop(percent=(0, 0.1))], random_order=True)
#
#         while True:
#             if shuffle:
#                 shuffle_indices = list(range(self.data_size))
#                 random.shuffle(shuffle_indices)
#                 self.images_list = [self.images_list[i] for i in shuffle_indices]
#                 self.labels_list = [self.labels_list[i] for i in shuffle_indices]
#
#             images_batch = []
#             labels_batch = []
#
#             for img_path, label_path in zip(self.images_list, self.labels_list):
#
#                 # --- images  ---
#
#                 if img_path in self.cache:
#                     img = self.cache[img_path]
#                 else:
#                     img = cv2.resize(np.asarray(Image.open(img_path)), image_size)
#                     if enable_caching:
#                         self.cache[img_path] = img
#
#                 images_batch.append(img)
#
#                 # --- labels ---
#
#                 if label_path in self.cache:
#                     label = self.cache[label_path]
#                 else:
#                     label = cv2.resize((plt.imread(label_path) * 255).astype(int), # by multiplying by 255 we get the id
#                                        image_size, interpolation=cv2.INTER_NEAREST)[..., None] # add one dimension for imgaug not to cry
#                     if enable_caching:
#                         self.cache[label_path] = label
#
#                 labels_batch.append(label)
#
#                 # --- yield batch ---
#
#                 if len(images_batch) == self.batch_size:
#                     if enable_augmentation:
#                         images_batch, labels_batch = seq(images=images_batch, segmentation_maps=labels_batch)
#
#                     labels_tensor_batch = [keras_utils.to_categorical(label, self.n_classes) for label in labels_batch]
#
#                     images_batch = np.array(images_batch) / 255.
#                     labels_tensor_batch = np.array(labels_tensor_batch)
#
#                     yield images_batch, labels_tensor_batch
#
#                     images_batch = []
#                     labels_batch = []
#
#     def show_random_samples(self, n_samples: int):
#         """
#         Shows random samples from the dataset.
#         --------------------------------------
#         :param n_samples: number of samples to show.
#         """
#         indices = random.sample(range(self.data_size), n_samples)
#
#         images = [np.asarray(Image.open(self.images_list[i])) / 255. for i in indices]
#         labels = [plt.imread(self.labels_list[i]) * 255. for i in indices]
#
#         show_grid_images([images, labels_to_images(images, labels, self.n_classes)])
#
#
# if __name__ == '__main__':
#     # images_path = os.path.join("leftImg8bit", "train", "*", "*_leftImg8bit.png")
#     # labels_path = os.path.join("converted", "gtCoarse", "train", "*", "*_labelIds.png")
#
#     from config import *
#
#     labels_path = CITYSCAPES_TRAIN_FINE_LABELS
#     images_path = CITYSCAPES_TRAIN_IMAGES
#     n_classes = 16
#
#     dataset = DataSet(images_path=images_path, labels_path=labels_path, n_classes=n_classes, batch_size=3)
#     train_generator = dataset.generate_data(image_size=(640, 320), shuffle=True, enable_caching=True)
#
#     # dataset.show_random_samples(2)
#     #
#     x, y = next(train_generator)
#     labeled_image = labels_to_images(x, one_hot_tensor_to_label(y), n_classes)
#     # plt.imshow(labeled_image)
#     show_grid_images([x, labeled_image])
#     plt.show()
#     # plt.savefig('hy.png')
#     print(x.shape, x[0])
#     print(y.shape, y[0])
#
#     # import time
#     #
#     # for _ in range(3):
#     #     epoch_start = time.time()
#     #
#     #     for i in range(dataset.n_batches):
#     #         next(train_generator)
#     #
#     #     epoch_end = time.time()
#     #     print('epoch time: {}'.format(epoch_end - epoch_start))


import tensorflow as tf

from typing import List, Union, Tuple
from functools import partial
from glob import glob

from config import *


def glob_files(file_pattern: Union[str, List[str]]):
    paths_list = []

    if isinstance(file_pattern, list):
        for p in file_pattern:
            paths_list.extend(glob(p))

    else:
        paths_list = glob(file_pattern)

    paths_list.sort()

    return paths_list


@tf.function
def read_images_tuple(input_path: str, label_path: str):
    input_image = tf.io.read_file(input_path)
    input_image = tf.io.decode_image(input_image, expand_animations=False)

    label_image = tf.io.read_file(label_path)
    label_image = tf.io.decode_image(label_image, expand_animations=False)

    return input_image, label_image


@tf.function
def normalize_images(input_image: tf.Tensor, label_image: tf.Tensor):
    input_image = tf.cast(input_image, tf.float32) / 255.0

    return input_image, label_image


def resize_images(input_image: tf.Tensor, label_image: tf.Tensor, size: Tuple[int, int]):
    input_image = tf.image.resize(
        images=input_image,
        size=size,
        method=tf.image.ResizeMethod.BILINEAR
    )

    label_image = tf.image.resize(
        images=label_image,
        size=size,
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )

    label_image = tf.squeeze(label_image, axis=-1)

    return input_image, label_image


@tf.function
def augment_images(input_image: tf.Tensor, label_image: tf.Tensor):
    uniform_random = tf.random.uniform([], 0, 1.0)
    flip_cond = tf.less(uniform_random, .5)
    input_image = tf.cond(flip_cond, lambda: tf.reverse(input_image, axis=[1]), lambda: input_image)
    label_image = tf.cond(flip_cond, lambda: tf.reverse(label_image, axis=[1]), lambda: label_image)

    # TODO

    return input_image, label_image


def to_one_hot(input_image: tf.Tensor, label_image: tf.Tensor, n_classes: int):
    label_image = tf.one_hot(label_image, depth=n_classes)

    return input_image, label_image


def create_dataset(inputs_file_pattern: Union[str, List[str]],
                   labels_file_pattern: Union[str, List[str]],
                   n_classes: int, batch_size: int,
                   size: Tuple[int, int] = None,
                   cache: bool = True, shuffle: bool = True, augment: bool = True):

    input_filenames = glob_files(inputs_file_pattern)
    label_filenames = glob_files(labels_file_pattern)

    input_filenames_ds = tf.data.Dataset.from_tensor_slices(input_filenames)
    label_filenames_ds = tf.data.Dataset.from_tensor_slices(label_filenames)

    dataset = tf.data.Dataset.zip((input_filenames_ds, label_filenames_ds))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(input_filenames))

    dataset = dataset.map(read_images_tuple, tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(normalize_images, tf.data.experimental.AUTOTUNE)

    if size is not None:
        resize_function = partial(resize_images, size=size)
        dataset = dataset.map(tf.function(resize_function), tf.data.experimental.AUTOTUNE)

    if cache:
        dataset = dataset.cache()

    dataset = dataset.map(augment_images, tf.data.experimental.AUTOTUNE)

    one_hot_function = partial(to_one_hot, n_classes=n_classes)
    dataset = dataset.map(tf.function(one_hot_function), tf.data.experimental.AUTOTUNE)

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(1)

    return dataset


if __name__ == '__main__':
    x = create_dataset(inputs_file_pattern=[BDD100K_TRAIN_IMAGES, CITYSCAPES_TRAIN_IMAGES],
                       labels_file_pattern=[BDD100K_TRAIN_LABELS, CITYSCAPES_TRAIN_FINE_LABELS],
                       n_classes=19, batch_size=5, size=(50, 100))

    from matplotlib import pyplot as plt
    import numpy as np

    for y1, z1 in x.take(2):
        for y in y1:
            plt.imshow(y)
            plt.show()
        for z in z1:
            plt.imshow(np.argmax(z, axis=-1))
            plt.show()


        x.take(100)
