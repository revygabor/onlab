import time

import numpy as np
import progressbar
import tensorflow as tf
from matplotlib import pyplot as plt
import pandas as pd

import HRNet
import new_deep_lab_v3_plus
from data_generator import create_dataset
from utils import create_inference_model, config_gpu
from config import *
from res_plotter import *


def eval_network(model_name, dataset_name, images_dir, labels_dir, create_network_func):
    labels = ['road', 'sidewalk', 'building/wall/fence',
              'pole', 'traffic light', 'traffic sign',
              'vegetation', 'sky', 'person', 'car', 'truck',
              'bus', 'train', 'motorcycle', 'bicycle', 'other']
    n_classes = len(labels)
    image_size = (320, 640)
    # image_size = (1664, 832)

    dataset = create_dataset(inputs_file_pattern=images_dir,
                             labels_file_pattern=labels_dir,
                             n_classes=n_classes, batch_size=5,
                             size=image_size, cache=False, shuffle=False, augment=False)

    model = create_network_func(input_resolution=image_size, n_classes=n_classes)
    # model.compile(optimizer=optimizer, loss=loss, metrics=['acc', 'categorical_accuracy'])
    model.load_weights('models/model.h5')
    model = create_inference_model(model)

    intersection_pixels = tf.zeros((n_classes,), dtype=tf.int64)
    union_pixels = tf.zeros((n_classes,), dtype=tf.int64)

    conf_matrix = np.zeros((n_classes, n_classes))

    i = 0

    dataset = dataset.repeat(1)
    n_batches = tf.data.experimental.cardinality(dataset).numpy()

    progress = progressbar.ProgressBar(widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                                progressbar.Percentage(), ' ',
                                                progressbar.ETA()], max_value=n_batches)

    for batch in dataset:
        x, y_true_one_hot = batch
        y_true_one_hot = tf.cast(y_true_one_hot, dtype=tf.int8)
        y_pred = model.predict(x)

        # computing iou
        y_pred_one_hot = tf.one_hot(y_pred, n_classes, dtype=tf.int8)

        intersection = tf.cast(y_pred_one_hot * y_true_one_hot, dtype=tf.int64)
        union = tf.cast((y_pred_one_hot + y_true_one_hot) > 0, dtype=tf.int64)

        intersection_pixels += tf.reduce_sum(intersection, axis=(0, 1, 2))  # reduce on batch, width, height
        union_pixels += tf.reduce_sum(union, axis=(0, 1, 2))  # reduce on batch, width, height

        # computing confusion matrix
        y_true = tf.argmax(y_true_one_hot, axis=-1)
        for y_true_class, y_pred_class in zip(np.array(y_true).reshape(-1), np.array(y_pred).reshape(-1)):
            conf_matrix[y_true_class, y_pred_class] += 1

        i += 1
        progress.update(i)

    iou = intersection_pixels / union_pixels
    fig, ax = plot_iou_barh(labels=labels, iou_vals=iou)
    fig.show()
    plt.show()

    res_dir = 'res/{}_{}_{}'.format(time.strftime('%Y%m%d-%H%M%S'), model_name, dataset_name)
    os.makedirs(res_dir)

    with open('{}/iou_{}_{}.csv'.format(res_dir, model_name, dataset_name), 'a') as iou_file:
        for i, l in enumerate(labels):
            print('{}\t{:3.2f}'.format(l, iou[i]), file=iou_file)
            # print('{}\t{:3.2f}'.format(l, iou[i]))

    conf_matrix_row_sum = np.sum(conf_matrix, axis=-1)
    conf_matrix_normalized = conf_matrix / np.expand_dims(conf_matrix_row_sum, axis=-1)
    np.save('{}/confmtx_{}_{}'.format(res_dir, model_name, dataset_name), conf_matrix_normalized)

    fig, ax = plot_confusion_matrix(conf_matrix_normalized, labels, title=" ")
    fig.show()
    plt.show()


#     fig.savefig('res/conf_mtx_{}_{}.svg'.format(model_name, dataset_name), pad_inches=0, format='svg')
if __name__ == '__main__':
    #
    # # models = glob.glob('*.h5')
    #
    # # for model in models:
    # #     eval_network(model, BDD100K_VAL, BDD100K_VAL_IMAGES, BDD100K_VAL_LABELS)
    # #     eval_network(model, CITYSCAPES_VAL_FINE, CITYSCAPES_VAL_IMAGES, CITYSCAPES_VAL_FINE_LABELS)
    #
    # # eval_network('model.h5', BDD100K_VAL, BDD100K_VAL_IMAGES, BDD100K_VAL_LABELS)

    config_gpu()
    dataset_config = CityScapesFineConfig()

    create_network_func = new_deep_lab_v3_plus.create_network
    eval_network('model.h5', dataset_config.dataset_name, dataset_config.val_images,
                 dataset_config.val_labels, create_network_func=create_network_func)
