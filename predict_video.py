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

    n_classes = 20
    image_size = (640, 320)

    images_path = os.path.join('leftImg8bit', 'demoVideo', '*', '*_leftImg8bit.png')
    images_list = glob.glob(images_path)
    images_list.sort()

    model = create_inference_model('deeplab_cityscapes_fine.h5')

    progress = progressbar.ProgressBar(widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                                progressbar.Percentage(), ' ',
                                                progressbar.ETA()])

    pred_path = 'video_pred'
    os.makedirs(pred_path, exist_ok=True)

    cm = plt.get_cmap('gist_ncar')
    for i in progress(range(len(images_list))):
    # for i in progress(range(5)):
        x_orig = cv2.imread(images_list[i]) / 255.
        x = cv2.resize(x_orig, dsize=image_size)
        x_size = tuple(x_orig.shape[1::-1])

        y_pred = model.predict(x[None, ...])[0]

        # plt.axis('off')
        # plt.tick_params(axis='both',
        #                 left=False, top=False,
        #                 right=False, bottom=False,
        #                 labelleft=False, labeltop=False,
        #                 labelright=False, labelbottom=False)
        # plt.imshow(x * .6 + cm(y_pred / float(n_classes))[..., :3] * .4)

        y = cm(y_pred / float(n_classes))[..., :3]
        y = cv2.resize(y, x_size, interpolation=cv2.INTER_NEAREST)

        out_img = (0.6 * x_orig + 0.4 * y) * 255

        image_path = os.path.join(pred_path, *(images_list[i].split(os.sep)[-2:]))
        image_dir = os.path.dirname(image_path)
        os.makedirs(image_dir, exist_ok=True)

        # plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
        cv2.imwrite(image_path, out_img)
