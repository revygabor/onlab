# import glob
#
# import cv2
# import tensorflow as tf
# import matplotlib.pyplot as plt
#
# from data_generator import DataSet, one_hot_tensor_to_label, show_grid_images, labels_to_images
# from utils import create_inference_model
# from config import *
#
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)
#
# val_images_dir = CITYSCAPES_VAL_IMAGES
# val_labels_dir = CITYSCAPES_VAL_FINE_LABELS
#
# n_classes = 20
# image_size = (640, 320)
#
# val_dataset = DataSet(batch_size=3, images_path=val_images_dir, labels_path=val_labels_dir, n_classes=n_classes)
# val_generator = val_dataset.generate_data(image_size=image_size, shuffle=True)
#
# x, y = next(val_generator)
# y = one_hot_tensor_to_label(y)
#
#
# models_list = glob.glob('*.h5')
#
# for file_name in models_list:
#
#     model = create_inference_model(file_name)
#
#     res = model.predict(x)
#
#     res_resized = [cv2.resize(res_label, image_size, interpolation=cv2.INTER_NEAREST) for
#                    res_label in res]
#
#     fig = show_grid_images([x, labels_to_images(x, y, n_classes), labels_to_images(x, res_resized, n_classes)])
#     fig.savefig('{}_test.png'.format(file_name.split('.')[0]), bbox_inches='tight', pad_inches=0)


import tensorflow as tf
import matplotlib.pyplot as plt

from data_generator import create_dataset
from utils import create_inference_model, config_gpu, labels_to_images, show_grid_images
from config import *
import HRNet
import new_deep_lab_v3_plus

config_gpu()

n_classes = 16
image_size = (320, 640)

dataset_name = CityScapesFineConfig()
train_dataset = create_dataset(inputs_file_pattern=dataset_name.val_images,
                               labels_file_pattern=dataset_name.val_labels,
                               n_classes=n_classes, batch_size=5,
                               size=image_size, cache=False, shuffle=False, augment=False)

# model = HRNet.create_network(input_resolution=image_size, n_classes=n_classes)
model = new_deep_lab_v3_plus.create_network(input_resolution=image_size, n_classes=n_classes)
model.load_weights('models/model.h5')
model = create_inference_model(model)
model.summary()
batches = train_dataset.take(5)

for batch in batches:
    x, y = batch
    # x, y = x[0], y[0]
    res = model.predict(x)
    labeled_images = labels_to_images(x, res, n_classes)
    ground_truth_label = labels_to_images(x, tf.argmax(y, axis=-1), n_classes)
    fig = show_grid_images([x, labeled_images, ground_truth_label])
    fig.show()




# res = model.predict(x)
#
# res_resized = [cv2.resize(res_label, image_size, interpolation=cv2.INTER_NEAREST) for
#                res_label in res]
#
# fig = show_grid_images([x, labels_to_images(x, y, n_classes), labels_to_images(x, res_resized, n_classes)])
# fig.savefig('{}_test.png'.format(file_name.split('.')[0]), bbox_inches='tight', pad_inches=0)
