import cv2
import glob
import os
import progressbar

import tensorflow as tf
from matplotlib import pyplot as plt

from utils import create_inference_model

if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    n_classes = 34
    image_size = (640, 320)

    images_dir = os.path.join("leftImg8bit_demoVideo")
    images_path = os.path.join(images_dir, "*", "*_leftImg8bit.png")
    images_list = glob.glob(images_path)
    images_list.sort()

    model = create_inference_model('model.h5')

    progress = progressbar.ProgressBar(widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                                progressbar.Percentage(), ' ',
                                                progressbar.ETA()])

    cm = plt.get_cmap('gist_ncar')
    for i in progress(range(len(images_list))):
        x = cv2.resize(cv2.imread(images_list[i]), dsize=image_size) / 255.
        y_pred = model.predict(x[None, ...])[0]

        plt.axis('off')
        plt.tick_params(axis='both',
                        left=False, top=False,
                        right=False, bottom=False,
                        labelleft=False, labeltop=False,
                        labelright=False, labelbottom=False)
        plt.imshow(x * .6 + cm(y_pred / 34.)[..., :3] * .4)
        plt.savefig('video_pred/{}'.format(images_list[i].split(os.sep)[-1]), bbox_inches='tight', pad_inches=0)
