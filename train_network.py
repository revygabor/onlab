import time

import tensorflow as tf
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import RMSprop, SGD, Adam

from config import *
from data_generator import DataSet
# from own_model import create_network
from deep_lab_v3_plus import create_network
# from HRNet import create_network
from utils import PlotOnEpochEnd

if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    TRAIN_FROM_START = True

    train_images_dir = CITYSCAPES_TRAIN_IMAGES
    train_labels_dir = CITYSCAPES_TRAIN_FINE_LABELS

    val_images_dir = CITYSCAPES_VAL_IMAGES
    val_labels_dir = CITYSCAPES_VAL_FINE_LABELS

    n_classes = 16
    image_size = (640, 320)

    batch_size = 5
    train_dataset = DataSet(images_path=train_images_dir, labels_path=train_labels_dir,
                            n_classes=n_classes, batch_size=batch_size)
    train_generator = train_dataset.generate_data(image_size=image_size, shuffle=True, enable_caching=True)

    val_dataset = DataSet(images_path=val_images_dir, labels_path=val_labels_dir,
                          n_classes=n_classes, batch_size=8)
    val_generator = val_dataset.generate_data(image_size=image_size, shuffle=True, enable_caching=True)

    if TRAIN_FROM_START:
        model = create_network(input_resolution=image_size[::-1], n_classes=n_classes)
        # model = create_network(input_resolution=(None, None), n_classes=n_classes)
        # optimizer = Adam(lr=1e-2, clipvalue=1.)
        optimizer = RMSprop(lr=1e-2)
        # optimizer = SGD(lr=1e-2)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
    else:
        optimizer = SGD(lr=1e-2)
        # optimizer = Adam(lr=1e-2)
        model = load_model('model.h5')
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])

    tb = TensorBoard(log_dir='./logs/{}'.format(time.strftime('%Y%m%d-%H%M%S')),
                     write_graph=False,
                     batch_size=batch_size,
                     update_freq=100)
    reduce_lr_on_plateau = ReduceLROnPlateau(patience=3, factor=.98)
    checkpointer = ModelCheckpoint('model.h5', save_best_only=True, verbose=2)
    plotter = PlotOnEpochEnd()
    early_stopping = EarlyStopping(min_delta=1e-3, patience=10)
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=train_dataset.n_batches,
                        epochs=200, validation_data=val_generator,
                        validation_steps=val_dataset.n_batches,
                        callbacks=[checkpointer, tb, reduce_lr_on_plateau,
                                   plotter,
                                   early_stopping])
