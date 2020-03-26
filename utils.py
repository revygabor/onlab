# import os
# import time
#
# import cv2
# import numpy as np
# import keras.backend as K
# from keras.layers import Layer
# from keras.callbacks import Callback, TensorBoard
# from matplotlib import pyplot as plt
# from keras.models import load_model, Model
# import tensorflow as tf
#
# from data_generator import DataSet, one_hot_tensor_to_label, show_grid_images, labels_to_images
# from config import CITYSCAPES_VAL_IMAGES, CITYSCAPES_VAL_FINE_LABELS
#
#
# def _fake_categorical_focal_loss(x, y):
#     return x-y
#
#
# def create_inference_model(name):
#     model = load_model(name, custom_objects={'categorical_focal_loss_func': _fake_categorical_focal_loss})
#     input_layer = model.layers[0].input
#     output_layer = model.layers[-1].output
#     new_output_layer = Argmax()(output_layer)
#
#     model = Model(inputs=input_layer, outputs=new_output_layer)
#     # model.summary()
#     return model
#
#
# class Argmax(Layer):
#     def __init__(self, axis=-1, **kwargs):
#         super(Argmax, self).__init__(**kwargs)
#         self.supports_masking = True
#         self.axis = axis
#
#     def call(self, inputs, mask=None):
#         return K.argmax(inputs, axis=self.axis)
#
#     def compute_output_shape(self, input_shape):
#         input_shape = list(input_shape)
#         del input_shape[self.axis]
#         return tuple(input_shape)
#
#     def compute_mask(self, x, mask):
#         return None
#
#     def get_config(self):
#         config = {'axis': self.axis}
#         base_config = super(Argmax, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))
#
#
# def plot_confusion_matrix(conf_mtx, classes,
#                           title=None,
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     """
#     if not title:
#         title = 'Normalized confusion matrix'
#
#     print(conf_mtx)
#
#     fig, ax = plt.subplots(figsize=(20, 20))
#     im = ax.imshow(conf_mtx, interpolation='nearest', cmap=cmap)
#     ax.figure.colorbar(im, ax=ax)
#     # We want to show all ticks...
#     ax.set(xticks=np.arange(conf_mtx.shape[1]),
#            yticks=np.arange(conf_mtx.shape[0]),
#            # ... and label them with the respective list entries
#            xticklabels=classes, yticklabels=classes,
#            title=title,
#            ylabel='True label',
#            xlabel='Predicted label')
#
#     # Rotate the tick labels and set their alignment.
#     plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#              rotation_mode="anchor")
#
#     # Loop over data dimensions and create text annotations.
#     fmt = '.2f'
#     thresh = conf_mtx.max() / 2.
#     for i in range(conf_mtx.shape[0]):
#         for j in range(conf_mtx.shape[1]):
#             ax.text(j, i, format(conf_mtx[i, j], fmt),
#                     ha="center", va="center",
#                     color="white" if conf_mtx[i, j] > thresh else "black")
#     # fig.tight_layout()
#     return fig, ax
#
#
# class PlotOnEpochEnd(Callback):
#
#     def __init__(self, save_images: bool = True):
#         val_images_dir = CITYSCAPES_VAL_IMAGES
#         val_labels_dir = CITYSCAPES_VAL_FINE_LABELS
#
#         self.n_classes = 20
#         self.image_size = (640, 320)
#
#         val_dataset = DataSet(batch_size=3, images_path=val_images_dir, labels_path=val_labels_dir,
#                               n_classes=self.n_classes)
#         val_generator = val_dataset.generate_data(image_size=self.image_size, shuffle=True)
#
#         self.x, y = next(val_generator)
#         self.y = one_hot_tensor_to_label(y)
#         self.y_image = labels_to_images(self.x, self.y, self.n_classes)
#
#         self.save_images = save_images
#         # folder to save images to
#         if save_images:
#             self.images_folder = './training_progress_images/{}/'.format(time.strftime('%Y%m%d-%H%M%S'))
#             os.makedirs(self.images_folder)
#
#     def on_epoch_end(self, epoch, logs=None):
#         res = self.model.predict(self.x)
#         res = one_hot_tensor_to_label(res)
#         res_resized = [cv2.resize(res_label, self.image_size, interpolation=cv2.INTER_NEAREST) for
#                        res_label in res]
#
#         fig = show_grid_images([self.x, self.y_image, labels_to_images(self.x, res_resized, self.n_classes)])
#         if self.save_images:
#             fig.savefig(self.images_folder + str(epoch) + '.png', bbox_inches='tight', pad_inches=0)
#
#
# class CyclicalLR(Callback):
#     def __init__(self, lr_max, lr_min, steps_per_epoch, lr_decay=.9997, power=.9):
#         self.lr_max = lr_max
#         self.lr_min = lr_min
#         self.steps_per_epoch = steps_per_epoch
#         self.lr_decay = lr_decay
#         self.power = power
#
#     def on_batch_begin(self, batch, logs=None):
#         lr = self.lr_max * np.power(1 - (batch / self.steps_per_epoch), self.power)
#         lr = np.maximum(lr, self.lr_min)
#         K.set_value(self.model.optimizer.lr, lr)
#
#         logs = logs or {}
#         print(' - lr: {}'.format(lr), end='\r')
#
#     def on_epoch_end(self, epoch, logs=None):
#         self.lr_max = self.lr_max * self.lr_decay
#         print('lr_max: {}'.format(self.lr_max))
#
#
# class MyTensorBoard(TensorBoard):
#     def on_batch_begin(self, batch, logs=None):
#         logs.update({'lr': K.eval(self.model.optimizer.lr)})
#         super().on_epoch_end(batch, logs)
#
#
#


#
# import os
# import time
#
# import cv2
# import numpy as np
# import keras.backend as K
# from keras.layers import Layer
# from keras.callbacks import Callback, TensorBoard
# from matplotlib import pyplot as plt
# from keras.models import load_model, Model
# import tensorflow as tf
#
# from data_generator import DataSet, one_hot_tensor_to_label, show_grid_images, labels_to_images
# from config import CITYSCAPES_VAL_IMAGES, CITYSCAPES_VAL_FINE_LABELS
#
#
# def _fake_categorical_focal_loss(x, y):
#     return x-y
#
from typing import List

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def create_inference_model(model):
    input_layer = model.layers[0].input
    output_layer = model.layers[-1].output
    new_output_layer = Argmax()(output_layer)

    model = tf.keras.Model(inputs=input_layer, outputs=new_output_layer)
    # model.summary()
    return model


class Argmax(tf.keras.layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super(Argmax, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis

    def call(self, inputs, mask=None):
        return tf.argmax(inputs, axis=self.axis)

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        del input_shape[self.axis]
        return tuple(input_shape)

    def compute_mask(self, x, mask):
        return None

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(Argmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


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
            cm(tf.cast(labels[i], tf.float32) / float(n_labels))[..., :3] * .4
            for i in range(len(images))]


# class PlotOnEpochEnd(Callback):
#
#     def __init__(self, save_images: bool = True):
#         val_images_dir = CITYSCAPES_VAL_IMAGES
#         val_labels_dir = CITYSCAPES_VAL_FINE_LABELS
#
#         self.n_classes = 20
#         self.image_size = (640, 320)
#
#         val_dataset = DataSet(batch_size=3, images_path=val_images_dir, labels_path=val_labels_dir,
#                               n_classes=self.n_classes)
#         val_generator = val_dataset.generate_data(image_size=self.image_size, shuffle=True)
#
#         self.x, y = next(val_generator)
#         self.y = one_hot_tensor_to_label(y)
#         self.y_image = labels_to_images(self.x, self.y, self.n_classes)
#
#         self.save_images = save_images
#         # folder to save images to
#         if save_images:
#             self.images_folder = './training_progress_images/{}/'.format(time.strftime('%Y%m%d-%H%M%S'))
#             os.makedirs(self.images_folder)
#
#     def on_epoch_end(self, epoch, logs=None):
#         res = self.model.predict(self.x)
#         res = one_hot_tensor_to_label(res)
#         res_resized = [cv2.resize(res_label, self.image_size, interpolation=cv2.INTER_NEAREST) for
#                        res_label in res]
#
#         fig = show_grid_images([self.x, self.y_image, labels_to_images(self.x, res_resized, self.n_classes)])
#         if self.save_images:
#             fig.savefig(self.images_folder + str(epoch) + '.png', bbox_inches='tight', pad_inches=0)
#
#
# class CyclicalLR(Callback):
#     def __init__(self, lr_max, lr_min, steps_per_epoch, lr_decay=.9997, power=.9):
#         self.lr_max = lr_max
#         self.lr_min = lr_min
#         self.steps_per_epoch = steps_per_epoch
#         self.lr_decay = lr_decay
#         self.power = power
#
#     def on_batch_begin(self, batch, logs=None):
#         lr = self.lr_max * np.power(1 - (batch / self.steps_per_epoch), self.power)
#         lr = np.maximum(lr, self.lr_min)
#         K.set_value(self.model.optimizer.lr, lr)
#
#         logs = logs or {}
#         print(' - lr: {}'.format(lr), end='\r')
#
#     def on_epoch_end(self, epoch, logs=None):
#         self.lr_max = self.lr_max * self.lr_decay
#         print('lr_max: {}'.format(self.lr_max))
#
#
# class MyTensorBoard(TensorBoard):
#     def on_batch_begin(self, batch, logs=None):
#         logs.update({'lr': K.eval(self.model.optimizer.lr)})
#         super().on_epoch_end(batch, logs)

def config_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
