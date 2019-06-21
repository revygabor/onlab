import os
import time

import cv2
import tensorflow as tf
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.optimizers import Adam
from tensorflow import clip_by_value

from data_generator import DataSet, show_grid_images, output_tensor_to_label
from utils import create_inference_model
# from own_model import create_network
from deep_lab_v3_plus import create_network


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    TRAIN_FROM_START = False

    train_images_dir = os.path.join("leftImg8bit", "train")
    # train_labels_dir = os.path.join("gtCoarse", "train")
    train_labels_dir = os.path.join("gtFine", "train")

    val_images_dir = os.path.join("leftImg8bit", "val")
    val_labels_dir = os.path.join("gtFine", "val")
    n_classes = 34
    image_size = (640, 320)

    train_dataset = DataSet(images_dir=train_images_dir, labels_dir=train_labels_dir, n_classes=n_classes, batch_size=6)
    train_generator = train_dataset.generate_data(image_size=image_size, shuffle=True)

    val_dataset = DataSet(images_dir=val_images_dir, labels_dir=val_labels_dir, n_classes=n_classes, batch_size=8)
    val_generator = val_dataset.generate_data(image_size=image_size, shuffle=True)

    if TRAIN_FROM_START:
        model = create_network(input_resolution=image_size[::-1], n_classes=n_classes)
        optimizer = Adam(lr=1e-4, clipvalue=1.)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
    else:
        model = load_model('model.h5')

    tb = TensorBoard(log_dir='./logs/{}'.format(time.strftime('%Y%m%d-%H%M%S')), batch_size=batch_size, update_freq='batch')
    reduce_lr_on_plateau = ReduceLROnPlateau(patience=1)
    checkpointer = ModelCheckpoint('model.h5', save_best_only=True)
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=train_dataset.n_batches,
                        # steps_per_epoch=2,
                        epochs=30, validation_data=val_generator,
                        validation_steps=val_dataset.n_batches,
                        # validation_steps=1,
                        callbacks=[checkpointer, tb, reduce_lr_on_plateau])