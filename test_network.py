import os

import cv2
import tensorflow as tf

from data_generator import DataSet, one_hot_tensor_to_label, show_grid_images, labels_to_images
from utils import create_inference_model

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

val_images_dir = os.path.join("leftImg8bit", "val", "*", "*_leftImg8bit.png")
val_labels_dir = os.path.join("gtFine", "val", "*", "*_labelIds.png")

n_classes = 20
image_size = (640, 320)

val_dataset = DataSet(batch_size=3, images_path=val_images_dir, labels_path=val_labels_dir, n_classes=n_classes)
val_generator = val_dataset.generate_data(image_size=image_size, shuffle=True)

model = create_inference_model('model.h5')

x, y = next(val_generator)
y = one_hot_tensor_to_label(y)
res = model.predict(x)

res_resized = [cv2.resize(res_label, image_size, interpolation=cv2.INTER_NEAREST) for
               res_label in res]

show_grid_images([x, labels_to_images(x, y, n_classes), labels_to_images(x, res_resized, n_classes)])
