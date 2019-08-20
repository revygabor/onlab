import os
import progressbar

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from keras.preprocessing import utils as keras_utils

from data_generator import DataSet
from utils import create_inference_model, plot_confusion_matrix

if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    val_images_dir = os.path.join("leftImg8bit", "val")
    val_labels_dir = os.path.join("gtCoarse", "val")
    n_classes = 34
    image_size = (640, 320)

    val_dataset = DataSet(images_dir=val_images_dir, labels_dir=val_labels_dir, n_classes=n_classes, batch_size=2)
    val_generator = val_dataset.generate_data(image_size=image_size, shuffle=True)

    next(val_generator)  # self.n_batches is created

    model = create_inference_model('model.h5')

    progress = progressbar.ProgressBar(widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                                progressbar.Percentage(), ' ',
                                                progressbar.ETA()])

    intersection_pixels = np.zeros((n_classes,))
    union_pixels = np.zeros((n_classes,))

    conf_matrix = np.zeros((n_classes, n_classes))

    for i in progress(range(val_dataset.n_batches)):
        x, y_true_one_hot = next(val_generator)
        y_pred = model.predict(x)

        # computing iou
        y_pred_one_hot = keras_utils.to_categorical(y_pred, n_classes)
        y_pred_bool = np.array(y_pred_one_hot, dtype=bool)
        y_true_bool = np.array(y_true_one_hot, dtype=bool)

        intersection = np.logical_and(y_pred_bool, y_true_bool)
        union = np.logical_or(y_pred_bool, y_true_bool)

        intersection = np.array(intersection, dtype=np.float32)
        union = np.array(union, dtype=np.float32)

        intersection_pixels += np.sum(intersection, axis=(0, 1, 2))  # reduce on batch, width, height
        union_pixels += np.sum(union, axis=(0, 1, 2))  # reduce on batch, width, height

        # computing confusion matrix
        y_true = np.argmax(y_true_one_hot, axis=-1)
        for y_true_class, y_pred_class in zip(y_true.flatten(), y_pred.flatten()):
            conf_matrix[y_true_class, y_pred_class] += 1

    iou = intersection_pixels / union_pixels
    conf_matrix_row_sum = np.sum(conf_matrix, axis=-1)
    conf_matrix_normalized = conf_matrix/np.expand_dims(conf_matrix_row_sum, axis=-1)

    labels = ['unlabeled', 'ego vehicle', 'rectification border', 'out of roi', 'static', 'dynamic', 'ground', 'road',
              'sidewalk', 'parking', 'rail track', 'building', 'wall', 'fence', 'guard rail', 'bridge', 'tunnel',
              'pole', 'polegroup', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider',
              'car', 'truck', 'bus', 'caravan', 'trailer', 'train', 'motorcycle', 'bicycle']
    for i, l in enumerate(labels):
        print('{:20s} - {:3.2f}'.format(l, iou[i]))

    fig, ax = plot_confusion_matrix(conf_matrix_normalized, labels)
    fig.show()
    plt.show()