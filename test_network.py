import os

import cv2

from data_generator import DataSet, output_tensor_to_label, show_grid_images, labels_to_images
from utils import create_inference_model

val_images_dir = os.path.join("leftImg8bit", "val")
val_labels_dir = os.path.join("gtFine", "val")

n_classes = 34
image_size = (640, 320)

val_dataset = DataSet(images_dir=val_images_dir, labels_dir=val_labels_dir, n_classes=n_classes)
val_generator = val_dataset.generate_data(batch_size=8, image_size=image_size, shuffle=True)

x, y = next(val_generator)
y = output_tensor_to_label(y)
model = create_inference_model('model.h5')
res = model.predict(x)

res_resized = [cv2.resize(res_label, image_size, interpolation=cv2.INTER_NEAREST) for
               res_label in res]

show_grid_images([x, labels_to_images(x, y, n_classes), res_resized])
