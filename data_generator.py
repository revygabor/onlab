import os
import glob
import random
from typing import List, Tuple

import cv2
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from keras.preprocessing import utils as keras_utils


def one_hot_tensor_to_label(output_label):
    """
    Converts output tensor to labels format
    :param output_label: output tensor from NN
    :return: tensor with class numbers
    """
    return np.argmax(output_label, axis=-1)


def labels_to_images(images: List[np.ndarray], labels: List[np.ndarray], n_labels: int):
    """
    Creates an image from the input and the label
    :param images: list of input images
    :param labels: list of labels
    :param n_labels: number of label types
    :return: list of created images
    """
    assert len(images) == len(labels), 'Length of image and label array must be equal!'
    cm = plt.get_cmap('gist_ncar')
    return [images[i] * .6 +
            cm(labels[i] / float(n_labels))[..., :3] * .4
            for i in range(len(images))]


def show_grid_images(images: List[List[np.ndarray]]):
    """
    Plots the images and the images with their labels.
    :param images: matrix of images (list of list of images)
    :return fig: current figure
    """
    n_cols = len(images)
    n_samples = len(images[0])

    fig, axes = plt.subplots(n_samples, n_cols, figsize=(20, 10))
    for i in range(n_samples):
        for j in range(n_cols):
            axes[i, j].imshow(images[j][i])
    fig = plt.gcf()
    # plt.show()
    return fig


class DataSet:
    def __init__(self, images_path: str, labels_path: str, n_classes: int, batch_size: int):
        """
        Initialize the DataSet class
        ----------------------------
        :param images_path: dir path to the images e.g. "leftImg8bit/train"; use os.path.join
        :param labels_path: dir path to the labels e.g. "gtCoarse/train"; use os.path.join
        :param n_classes: number of classes in the dataset
        :param batch_size: batch size
        """
        self.cache = {}
        self.images_dir = images_path
        self.labels_dir = labels_path
        self.n_classes = n_classes

        images_list = glob.glob(images_path)
        images_list.sort()
        self.images_list = images_list

        labels_list = glob.glob(labels_path)
        labels_list.sort()
        self.labels_list = labels_list

        assert len(images_list) == len(labels_list), '#images and #labels must be equal'
        self.data_size = len(images_list)
        print("{} images found in directory {}".format(self.data_size, self.images_dir))
        self.n_batches = self.data_size // batch_size
        self.batch_size = batch_size

    def generate_data(self, image_size: Tuple[int, int] = (600, 300), shuffle: bool = True,
                      enable_caching: bool = False):
        """
        Generator for the training, generates (x, y) pair
        -------------------------------------------------
        :param enable_caching: whether or not you want to enable to cache the input+output images
        :param image_size: size to resize the image to
        :param shuffle: whether to shuffle dataset after one epoch
        :return: (x, y) where x: batch of input images, y: batch of one hot encoded pixel-level labels
        """

        while True:
            if shuffle:
                shuffle_indices = list(range(self.data_size))
                random.shuffle(shuffle_indices)
                self.images_list = [self.images_list[i] for i in shuffle_indices]
                self.labels_list = [self.labels_list[i] for i in shuffle_indices]

                images_batch = []
                labels_batch = []

            for img_path, label_path in zip(self.images_list, self.labels_list):

                # --- images  ---

                if img_path in self.cache:
                    img = self.cache[img_path]
                else:
                    img = cv2.resize(np.asarray(Image.open(img_path)) / 255., image_size)
                    if enable_caching:
                        self.cache[img_path] = img

                images_batch.append(img)

                # --- labels ---

                if label_path in self.cache:
                    label = self.cache[label_path]
                else:
                    label = cv2.resize((plt.imread(label_path) * 255).astype(int), # by multiplying by 255 we get the id

                                       image_size, interpolation=cv2.INTER_NEAREST)
                    if enable_caching:
                        self.cache[label_path] = label

                labels_batch.append(label)

                # --- yield batch ---

                if len(images_batch) == self.batch_size:
                    labels_tensor_batch = [keras_utils.to_categorical(label, self.n_classes) for label in labels_batch]

                    images_batch = np.array(images_batch)
                    labels_tensor_batch = np.array(labels_tensor_batch)

                    yield images_batch, labels_tensor_batch

                    images_batch = []
                    labels_batch = []

    def show_random_samples(self, n_samples: int):
        """
        Shows random samples from the dataset.
        --------------------------------------
        :param n_samples: number of samples to show.
        """
        indices = random.sample(range(self.data_size), n_samples)

        images = [np.asarray(Image.open(self.images_list[i])) / 255. for i in indices]
        labels = [plt.imread(self.labels_list[i]) * 255. for i in indices]

        show_grid_images([images, labels_to_images(images, labels, self.n_classes)])


if __name__ == '__main__':
    # images_path = os.path.join("leftImg8bit", "train", "*", "*_leftImg8bit.png")
    # labels_path = os.path.join("converted", "gtCoarse", "train", "*", "*_labelIds.png")

    from config import *

    labels_path = CITYSCAPES_TRAIN_FINE_LABELS
    images_path = CITYSCAPES_TRAIN_IMAGES
    n_classes = 16

    dataset = DataSet(images_path=images_path, labels_path=labels_path, n_classes=n_classes, batch_size=3)
    train_generator = dataset.generate_data(image_size=(640, 320), shuffle=True, enable_caching=True)

    # dataset.show_random_samples(2)
    #
    x, y = next(train_generator)
    # labeled_image = labels_to_images(x, one_hot_tensor_to_label(y), n_classes)
    # show_grid_images([x, labeled_image])
    print(x.shape, x[0])
    print(y.shape, y[0])

    # import time
    #
    # for _ in range(3):
    #     epoch_start = time.time()
    #
    #     for i in range(dataset.n_batches):
    #         next(train_generator)
    #
    #     epoch_end = time.time()
    #     print('epoch time: {}'.format(epoch_end - epoch_start))
