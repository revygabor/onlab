import random
from glob import glob
import time

import tensorflow as tf

from config import *
from data_generator import create_dataset
# from own_model import create_network
# from deep_lab_v3_plus import create_network
# from HRNet import create_network
from deep_lab_v3_plus import create_network
from utils import config_gpu


# from utils import PlotOnEpochEnd, CyclicalLR, MyTensorBoard


@tf.function
def train_step(x, y, model, optimizer, metrics):
    with tf.GradientTape() as tape:
        predictions = model(x)
        loss = tf.reduce_mean(
            tf.losses.categorical_crossentropy(
                y, predictions, from_logits=False))
        grads = tape.gradient(loss, model.trainable_variables)

        metrics['train_acc'].update_state(y, predictions)
        metrics['train_loss'].update_state(y, predictions)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


@tf.function
def val_step(x, y, model, metrics):
    predictions = model(x)
    metrics['val_acc'].update_state(y, predictions)
    metrics['val_loss'].update_state(y, predictions)


def create_datasets(train_config, n_classes, image_size, batch_size, frames, shift, stride):
    train_images_dirs = glob(train_config.train_image_dirs)
    train_images_dirs.sort()
    train_images_dirs = [os.path.join(images_dir, '*.jpg') for images_dir in train_images_dirs]

    train_labels_dirs = glob(train_config.train_label_dirs)
    train_labels_dirs.sort()
    train_labels_dirs = [os.path.join(labels_dir, '*.png') for labels_dir in train_labels_dirs]
    train_datasets = [create_dataset(inputs_file_pattern=train_images_dir,
                                     labels_file_pattern=train_labels_dir,
                                     n_classes=n_classes, size=image_size,
                                     frames=frames, shift=shift, stride=stride)
                      for train_images_dir, train_labels_dir in zip(train_images_dirs, train_labels_dirs)]


    val_images_dirs = glob(train_config.val_image_dirs)
    val_images_dirs.sort()
    val_images_dirs = [os.path.join(images_dir, '*.jpg') for images_dir in val_images_dirs]

    val_labels_dirs = glob(train_config.val_label_dirs)
    val_labels_dirs.sort()
    val_labels_dirs = [os.path.join(labels_dir, '*.png') for labels_dir in val_labels_dirs]
    val_datasets = [create_dataset(inputs_file_pattern=val_images_dir,
                                   labels_file_pattern=val_labels_dir,
                                   n_classes=n_classes, size=image_size,
                                   frames=frames, shift=shift, stride=stride)
                    for val_images_dir, val_labels_dir in zip(val_images_dirs, val_labels_dirs)]

    train_concat_dataset = train_datasets[0]
    val_concat_dataset = train_datasets[0]

    for ds in train_datasets[1:]:
        train_concat_dataset = train_concat_dataset.concatenate(ds)

    for ds in val_datasets[1:]:
        val_concat_dataset = val_concat_dataset.concatenate(ds)

    train_concat_dataset.shuffle(buffer_size=1000, seed=1)
        # buffer_size=tf.data.experimental.cardinality(train_concat_dataset).numpy())

    train_concat_dataset = train_concat_dataset.batch(batch_size, drop_remainder=True)
    train_concat_dataset.cache()
    train_concat_dataset = train_concat_dataset.prefetch(2)

    val_concat_dataset = val_concat_dataset.batch(batch_size, drop_remainder=True)
    val_concat_dataset.cache()
    val_concat_dataset = val_concat_dataset.prefetch(2)

    return train_concat_dataset, val_concat_dataset


if __name__ == '__main__':
    config_gpu()

    LOAD_MODEL_WEIGHTS = False
    train_config = ApolloScapeConfig

    frames = 5
    shift = 1
    stride = 1

    batch_size = 2
    lr_max = .001
    image_size = (384, 480)
    loss = tf.losses.categorical_crossentropy
    optimizer = tf.keras.optimizers.Adam(lr=lr_max, epsilon=1e-8)

    n_classes = 16

    train_dataset, val_dataset = create_datasets(train_config, n_classes,
                                                   image_size, batch_size,
                                                   frames, shift, stride)

    model = create_network(batch_size=batch_size, frames=frames,
                           input_image_resolution=image_size,
                           n_classes=n_classes, weights='cityscapes')
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc', 'categorical_accuracy'])
    model.summary()

    for l in model.layers[:-19]:
        l.trainable = False

    # if LOAD_MODEL_WEIGHTS:
    #     model.load_weights('models/model.h5', by_name=True)

    # tb = tf.keras.callbacks.TensorBoard(log_dir='./logs/' + time.strftime('%Y%m%d-%H%M%S'),
    #                                     write_graph=True,
    #                                     update_freq=20)
    # reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=.8)
    # checkpoint = tf.keras.callbacks.ModelCheckpoint('models/model.h5', save_best_only=True,
    #                                                 verbose=2, save_weights_only=True)
    # early_stopping = tf.keras.callbacks.EarlyStopping(min_delta=1e-3, patience=20)

    metrics = {
        'train_loss': tf.keras.metrics.CategoricalCrossentropy(),
        'train_acc': tf.keras.metrics.BinaryAccuracy(),
        'val_loss': tf.keras.metrics.CategoricalCrossentropy(),
        'val_acc': tf.keras.metrics.BinaryAccuracy()
    }

    max_train_iterations = None
    max_val_iterations = None
    train_iteration = 0
    val_iteration = 0

    n_epochs = 100
    for epoch in range(n_epochs):

        if train_iteration > 0:
            max_train_iterations = train_iteration

        if val_iteration > 0:
            max_val_iterations = val_iteration

        train_iteration = 0
        [metric.reset_states() for metric in metrics.values()]

        for x, y in train_dataset:
            train_step(x, y, model, optimizer, metrics)
            if train_iteration % 10 == 0 or train_iteration == max_train_iterations:
                tf.print('train iteration: {} | train_acc: {:.4f} \t train_loss: {:.4f}'
                         .format(train_iteration,
                                 metrics['train_acc'].result(),
                                 metrics['train_loss'].result()))
            train_iteration += 1

        for x, y in val_dataset:
            val_step(x, y, model, metrics)
            if val_iteration % 10 == 0:
                tf.print('val iteration: {} | val_acc: {:.4f} \t val_loss: {:.4f}'
                         .format(val_iteration,
                                 metrics['val_acc'].result(),
                                 metrics['val_loss'].result()))
            val_iteration += 1

        tf.print('end of epoch {} | '
                 'train_acc: {:.4f} - val_acc: {:.4f} \t'
                 'train_loss: {:.4f} - val_loss: {:.4f}'
                 .format(epoch,
                         metrics['train_acc'].result(), metrics['val_acc'].result(),
                         metrics['train_loss'].result(), metrics['val_loss'].result()))
