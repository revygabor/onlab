import glob
import progressbar

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from keras.preprocessing import utils as keras_utils

from data_generator import DataSet
from utils import create_inference_model, plot_confusion_matrix

from config import *


def eval_network(model_name, dataset_name, images_dir, labels_dir):
    labels = ['road', 'sidewalk', 'building/wall/fence',
              'pole', 'traffic light', 'traffic sign',
              'vegetation', 'sky', 'person', 'car', 'truck',
              'bus', 'train', 'motorcycle', 'bicycle', 'other']
    n_classes = len(labels)
    image_size = (640, 320)
    # image_size = (1280, 704)

    dataset = DataSet(images_path=images_dir, labels_path=labels_dir, n_classes=n_classes, batch_size=2)
    generator = dataset.generate_data(image_size=image_size, shuffle=False)

    model = create_inference_model(model_name)

    progress = progressbar.ProgressBar(widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                                progressbar.Percentage(), ' ',
                                                progressbar.ETA()])

    intersection_pixels = np.zeros((n_classes,))
    union_pixels = np.zeros((n_classes,))

    conf_matrix = np.zeros((n_classes, n_classes))

    for i in progress(range(dataset.n_batches)):
        x, y_true_one_hot = next(generator)
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
    conf_matrix_normalized = conf_matrix / np.expand_dims(conf_matrix_row_sum, axis=-1)

    with open('res/iou_{}_{}.csv'.format(model_name, dataset_name), 'a') as iou_file:
        for i, l in enumerate(labels):
            print('{}\t{:3.2f}'.format(l, iou[i]), file=iou_file)

    fig, ax = plot_confusion_matrix(conf_matrix_normalized, labels)
    # fig.show()
    # plt.show()
    fig.savefig('res/conf_mtx_{}_{}.svg'.format(model_name, dataset_name), pad_inches=0, format='svg')

    np.save('res/conf_mtx_{}_{}.npy'.format(model_name, dataset_name), conf_matrix_normalized)


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    models = glob.glob('*.h5')

    for model in models:
        eval_network(model, BDD100K_VAL, BDD100K_VAL_IMAGES, BDD100K_VAL_LABELS)
        eval_network(model, CITYSCAPES_VAL_FINE, CITYSCAPES_VAL_IMAGES, CITYSCAPES_VAL_FINE_LABELS)
