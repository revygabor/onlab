import os
import time

import tensorflow as tf
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam, RMSprop, SGD

from data_generator import DataSet
# from own_model import create_network
from deep_lab_v3_plus import create_network
from utils import PlotOnEpochEnd

if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    TRAIN_FROM_START = True

    train_images_dir = os.path.join("leftImg8bit", "train")
    # train_labels_dir = os.path.join("gtCoarse", "train")
    train_labels_dir = os.path.join("gtFine", "train")

    val_images_dir = os.path.join("leftImg8bit", "val")
    val_labels_dir = os.path.join("gtFine", "val")
    n_classes = 34
    image_size = (640, 320)

    batch_size = 6
    train_dataset = DataSet(images_dir=train_images_dir, labels_dir=train_labels_dir, n_classes=n_classes, batch_size=batch_size)
    train_generator = train_dataset.generate_data(image_size=image_size, shuffle=True)

    val_dataset = DataSet(images_dir=val_images_dir, labels_dir=val_labels_dir, n_classes=n_classes, batch_size=8)
    val_generator = val_dataset.generate_data(image_size=image_size, shuffle=True)

    if TRAIN_FROM_START:
        model = create_network(input_resolution=image_size[::-1], n_classes=n_classes)
        # optimizer = Adam(lr=1e-2, clipvalue=1.)
        optimizer = RMSprop(lr=1e-2)
        # optimizer = SGD(lr=1e-2)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
    else:
        model = load_model('model.h5')

    tb = TensorBoard(log_dir='./logs/{}'.format(time.strftime('%Y%m%d-%H%M%S')), batch_size=batch_size, update_freq='batch')
    reduce_lr_on_plateau = ReduceLROnPlateau(patience=3, factor=.98)
    checkpointer = ModelCheckpoint('model.h5', save_best_only=True, verbose=2)
    plotter = PlotOnEpochEnd()
    early_stopping = EarlyStopping(min_delta=1e-3, patience=5)
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=train_dataset.n_batches,
                        # steps_per_epoch=50,
                        epochs=200, validation_data=val_generator,
                        validation_steps=val_dataset.n_batches,
                        # validation_steps=120,
                        callbacks=[checkpointer, tb, reduce_lr_on_plateau, plotter, early_stopping])