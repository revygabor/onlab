import time

import tensorflow as tf

from config import *
from data_generator import create_dataset
# from own_model import create_network
# from deep_lab_v3_plus import create_network
# from HRNet import create_network
from new_deep_lab_v3_plus import create_network
from utils import config_gpu

# from utils import PlotOnEpochEnd, CyclicalLR, MyTensorBoard


# def categorical_focal_loss(gamma=2., alpha=.25):
#     """
#     Softmax version of focal loss.
#            m
#       FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
#           c=1
#       where m = number of classes, c = class and o = observation
#     Parameters:
#       alpha -- the same as weighing factor in balanced cross entropy
#       gamma -- focusing parameter for modulating factor (1-p)
#     Default value:
#       gamma -- 2.0 as mentioned in the paper
#       alpha -- 0.25 as mentioned in the paper
#     References:
#         Official paper: https://arxiv.org/pdf/1708.02002.pdf
#         https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
#     Usage:
#      model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
#     """
#
#     def categorical_focal_loss_func(y_true, y_pred):
#         """
#         :param y_true: A tensor of the same shape as `y_pred`
#         :param y_pred: A tensor resulting from a softmax
#         :return: Output tensor.
#         """
#
#         # Scale predictions so that the class probas of each sample sum to 1
#         y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
#
#         # Clip the prediction value to prevent NaN's and Inf's
#         epsilon = K.epsilon()
#         y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
#
#         # Calculate Cross Entropy
#         cross_entropy = -y_true * K.log(y_pred)
#
#         # Calculate Focal Loss
#         loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
#
#         # Sum the losses in mini_batch
#         return K.sum(loss, axis=1)
#
#     return categorical_focal_loss_func

if __name__ == '__main__':
    config_gpu()

    LOAD_MODEL_WEIGHTS = True
    train_config = [CityScapesFineConfig]

    train_images_dir = [ds.train_images for ds in train_config]
    train_labels_dir = [ds.train_labels for ds in train_config]

    val_images_dir = [ds.val_images for ds in train_config]
    val_labels_dir = [ds.val_labels for ds in train_config]

    batch_size = 5
    lr_max = .001
    image_size = (320, 640)
    # image_size = (1664, 832)
    loss = tf.losses.categorical_crossentropy
    optimizer = tf.keras.optimizers.Adam(lr=lr_max, epsilon=1e-8)

    n_classes = 16

    train_dataset = create_dataset(inputs_file_pattern=train_images_dir,
                                   labels_file_pattern=train_labels_dir,
                                   n_classes=n_classes, batch_size=batch_size, size=image_size)

    val_dataset = create_dataset(inputs_file_pattern=val_images_dir,
                                 labels_file_pattern=val_labels_dir,
                                 n_classes=n_classes, batch_size=10, size=image_size)

    model = create_network(input_resolution=image_size, n_classes=n_classes)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc', 'categorical_accuracy'])
    for l in model.layers[:-3]:
        l.trainable = False

    if LOAD_MODEL_WEIGHTS:
        model.load_weights('models/model.h5')

    tb = tf.keras.callbacks.TensorBoard(log_dir='./logs/' + time.strftime('%Y%m%d-%H%M%S'),
                                        write_graph=True,
                                        update_freq=20)
    reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=.8)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('models/model.h5', save_best_only=True,
                                                    verbose=2, save_weights_only=True)
    early_stopping = tf.keras.callbacks.EarlyStopping(min_delta=1e-3, patience=20)
    # plotter = PlotOnEpochEnd()

    # cyclical_lr = CyclicalLR(lr_max=lr_max, lr_min=1e-6, steps_per_epoch=train_dataset.n_batches, lr_decay=0.99)

    model.fit(train_dataset,
              callbacks=[
                  tb,
                  checkpoint,
                  reduce_lr_on_plateau,
                  early_stopping
                  # plotter,
                  # #  cyclical_lr,
              ],
              epochs=100, validation_data=val_dataset)
