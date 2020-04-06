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

if __name__ == '__main__':
    config_gpu()

    LOAD_MODEL_WEIGHTS = True
    train_config = ApolloScapeConfig

    batch_size = 5
    lr_max = .001
    image_size = (384, 480)
    loss = tf.losses.categorical_crossentropy
    optimizer = tf.keras.optimizers.Adam(lr=lr_max, epsilon=1e-8)

    n_classes = 16

    train_images_dirs = glob(train_config.train_image_dirs)
    train_images_dirs.sort()
    train_images_dirs = [glob(os.path.join(images_dir, '*.jpg')) for images_dir in train_images_dirs]

    train_labels_dirs = glob(train_config.train_label_dirs)
    train_labels_dirs.sort()
    train_labels_dirs = [glob(os.path.join(labels_dir, '*.jpg')) for labels_dir in train_labels_dirs]

    train_datasets = [create_dataset(inputs_file_pattern=train_images_dir,
                                     labels_file_pattern=train_labels_dir,
                                     n_classes=n_classes, batch_size=batch_size, size=image_size)
                      for train_images_dir, train_labels_dir in zip(train_images_dirs, train_labels_dirs)]

    val_images_dirs = glob(train_config.val_image_dirs)
    val_images_dirs.sort()
    val_images_dirs = [glob(os.path.join(images_dir, '*.jpg')) for images_dir in val_images_dirs]

    val_labels_dirs = glob(train_config.val_label_dirs)
    val_labels_dirs.sort()
    val_labels_dirs = [glob(os.path.join(labels_dir, '*.jpg')) for labels_dir in val_labels_dirs]

    val_datasets = [create_dataset(inputs_file_pattern=val_images_dir,
                                  labels_file_pattern=val_labels_dir,
                                  n_classes=n_classes, batch_size=10, size=image_size)
                   for val_images_dir, val_labels_dir in zip(val_images_dirs, val_labels_dirs)]

    model = create_network(frames=batch_size, input_image_resolution=image_size, n_classes=n_classes, weights=None)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc', 'categorical_accuracy'])
    # for l in model.layers[:-3]:
    #     l.trainable = False

    # if LOAD_MODEL_WEIGHTS:
    #     model.load_weights('models/model.h5', by_name=True)

    tb = tf.keras.callbacks.TensorBoard(log_dir='./logs/' + time.strftime('%Y%m%d-%H%M%S'),
                                        write_graph=True,
                                        update_freq=20)
    reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=.8)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('models/model.h5', save_best_only=True,
                                                    verbose=2, save_weights_only=True)
    early_stopping = tf.keras.callbacks.EarlyStopping(min_delta=1e-3, patience=20)
    # plotter = PlotOnEpochEnd()

    # cyclical_lr = CyclicalLR(lr_max=lr_max, lr_min=1e-6, steps_per_epoch=train_dataset.n_batches, lr_decay=0.99)

    n_epochs = 100
    for epoch in range(n_epochs):
        dss = list(zip(train_datasets, val_datasets))
        random.shuffle(dss)
        train_datasets, val_datasets = zip(*dss)

        for train_dataset, val_dataset in zip(train_datasets[:-1], val_datasets[:-1]):
            model.reset_states()
            model.fit(train_dataset,
                      shuffle=False,
                      initial_epoch=epoch,
                      callbacks=[
                          tb,
                          checkpoint,
                          reduce_lr_on_plateau,
                          early_stopping
                          # plotter,
                          # #  cyclical_lr,
                      ],
                      epochs=1)

        model.reset_states()
        model.fit(train_datasets[-1],
                  shuffle=False,
                  initial_epoch=epoch,
                  callbacks=[
                      tb,
                      checkpoint,
                      reduce_lr_on_plateau,
                      early_stopping
                      # plotter,
                      # #  cyclical_lr,
                  ],
                  epochs=1, validation_data=val_datasets[-1])
