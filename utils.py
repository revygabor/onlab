import os
import time

import cv2
import numpy as np
import keras.backend as K
from keras.layers import Layer
from keras.callbacks import Callback
from matplotlib import pyplot as plt
from keras.models import load_model, Model

from data_generator import DataSet, one_hot_tensor_to_label, show_grid_images, labels_to_images


def create_inference_model(name):
    model = load_model(name)
    input_layer = model.layers[0].input
    output_layer = model.layers[-1].output
    new_output_layer = Argmax()(output_layer)

    model = Model(inputs=input_layer, outputs=new_output_layer)
    # model.summary()
    return model


class Argmax(Layer):
    def __init__(self, axis=-1, **kwargs):
        super(Argmax, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis

    def call(self, inputs, mask=None):
        return K.argmax(inputs, axis=self.axis)

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


def plot_confusion_matrix(conf_mtx, classes,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    if not title:
        title = 'Normalized confusion matrix'

    print(conf_mtx)

    fig, ax = plt.subplots(figsize=(20, 20))
    im = ax.imshow(conf_mtx, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(conf_mtx.shape[1]),
           yticks=np.arange(conf_mtx.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = conf_mtx.max() / 2.
    for i in range(conf_mtx.shape[0]):
        for j in range(conf_mtx.shape[1]):
            ax.text(j, i, format(conf_mtx[i, j], fmt),
                    ha="center", va="center",
                    color="white" if conf_mtx[i, j] > thresh else "black")
    # fig.tight_layout()
    return fig, ax


class PlotOnEpochEnd(Callback):

    def __init__(self, save_images: bool=True):
        val_images_dir = os.path.join("leftImg8bit", "val")
        val_labels_dir = os.path.join("gtFine", "val")

        self.n_classes = 34
        self.image_size = (640, 320)

        val_dataset = DataSet(batch_size=3, images_dir=val_images_dir, labels_dir=val_labels_dir, n_classes=self.n_classes)
        val_generator = val_dataset.generate_data(image_size=self.image_size, shuffle=True)

        self.x, y = next(val_generator)
        self.y = one_hot_tensor_to_label(y)
        self.y_image = labels_to_images(self.x, self.y, self.n_classes)

        self.save_images = save_images
        # folder to save images to
        if save_images:
            self.images_folder = './training_progress_images/{}/'.format(time.strftime('%Y%m%d-%H%M%S'))
            os.makedirs(self.images_folder)

    def on_epoch_end(self, epoch, logs=None):
        res = self.model.predict(self.x)
        res = one_hot_tensor_to_label(res)
        res_resized = [cv2.resize(res_label, self.image_size, interpolation=cv2.INTER_NEAREST) for
                       res_label in res]

        fig = show_grid_images([self.x, self.y_image, labels_to_images(self.x, res_resized, self.n_classes)])
        if self.save_images:
            fig.savefig(self.images_folder+str(epoch)+'.png', bbox_inches='tight', pad_inches=0)
