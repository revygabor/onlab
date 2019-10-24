import glob
import os
from collections import namedtuple

import numpy as np
import progressbar
from matplotlib import pyplot as plt
from PIL import Image

Label = namedtuple('Label', [

    'name',  # The identifier of this label, e.g. 'car', 'person', ... .
    # We use them to uniquely name a class

    'id',  # An integer ID that is associated with this label.
    # The IDs are used to represent the label in ground truth images
    # An ID of -1 means that this label does not have an ID and thus
    # is ignored when creating ground truth images (e.g. license plate).
    # Do not modify these IDs, since exactly these IDs are expected by the
    # evaluation server.

    'trainId',  # Feel free to modify these IDs as suitable for your method. Then create
    # ground truth images with train IDs, using the tools provided in the
    # 'preparation' folder. However, make sure to validate or submit results
    # to our evaluation server using the regular IDs above!
    # For trainIds, multiple labels might have the same ID. Then, these labels
    # are mapped to the same class in the ground truth images. For the inverse
    # mapping, we use the label that is defined first in the list below.
    # For example, mapping all void-type classes to the same ID in training,
    # might make sense for some approaches.
    # Max value is 255!

    'myId', # my training id

    'category',  # The name of the category that this label belongs to

    'categoryId',  # The ID of this category. Used to create ground truth images
    # on category level.

    'hasInstances',  # Whether this label distinguishes between single instances or not

    'ignoreInEval',  # Whether pixels having this class as ground truth label are ignored
    # during evaluations or not

    'color',  # The color of this label
])

labels = [
    #      name                     id      trainId     myId   category         catId     hasInstances   ignoreInEval   color
    Label('unlabeled',              0,      255,        15,    'void',          0, False, True, (0, 0, 0)),
    Label('ego vehicle',            1,      255,        15,    'void',          0, False, True, (0, 0, 0)),
    Label('rectification border',   2,      255,        15,    'void',          0, False, True, (0, 0, 0)),
    Label('out of ',                3,      255,        15,    'void',          0, False, True, (0, 0, 0)),
    Label('static',                 4,      255,        15,    'void',          0, False, True, (0, 0, 0)),
    Label('dynamic',                5,      255,        15,    'void',          0, False, True, (111, 74, 0)),
    Label('ground',                 6,      255,        15,    'void',          0, False, True, (81, 0, 81)),
    Label('road',                   7,      0,          0,     'flat',          1, False, False, (128, 64, 128)),
    Label('sidewalk',               8,      1,          1,     'flat',          1, False, False, (244, 35, 232)),
    Label('parking',                9,      255,        15,    'flat',          1, False, True, (250, 170, 160)),
    Label('rail track',             10,     255,        15,    'flat',          1, False, True, (230, 150, 140)),
    Label('building',               11,     2,          2,     'construction',  2, False, False, (70, 70, 70)),
    Label('wall',                   12,     3,          2,     'construction',  2, False, False, (102, 102, 156)),
    Label('fence',                  13,     4,          2,     'construction',  2, False, False, (190, 153, 153)),
    Label('guard rail',             14,     255,        15,    'construction',  2, False, True, (180, 165, 180)),
    Label('bridge',                 15,     255,        15,    'construction',  2, False, True, (150, 100, 100)),
    Label('tunnel',                 16,     255,        15,    'construction',  2, False, True, (150, 120, 90)),
    Label('pole',                   17,     5,          3,     'object',        3, False, False, (153, 153, 153)),
    Label('polegroup',              18,     255,        15,    'object',        3, False, True, (153, 153, 153)),
    Label('traffic light',          19,     6,          4,     'object',        3, False, False, (250, 170, 30)),
    Label('traffic sign',           20,     7,          5,     'object',        3, False, False, (220, 220, 0)),
    Label('vegetation',             21,     8,          6,     'nature',        4, False, False, (107, 142, 35)),
    Label('terrain',                22,     9,          6,     'nature',        4, False, False, (152, 251, 152)),
    Label('sky',                    23,     10,         7,     'sky',           5, False, False, (70, 130, 180)),
    Label('person',                 24,     11,         8,     'human',         6, True, False, (220, 20, 60)),
    Label('rider',                  25,     12,         8,     'human',         6, True, False, (255, 0, 0)),
    Label('car',                    26,     13,         9,     'vehicle',       7, True, False, (0, 0, 142)),
    Label('truck',                  27,     14,         10,    'vehicle',       7, True, False, (0, 0, 70)),
    Label('bus',                    28,     15,         11,    'vehicle',       7, True, False, (0, 60, 100)),
    Label('caravan',                29,     255,        15,    'vehicle',       7, True, True, (0, 0, 90)),
    Label('trailer',                30,     255,        15,    'vehicle',       7, True, True, (0, 0, 110)),
    Label('train',                  31,     16,         12,    'vehicle',       7, True, False, (0, 80, 100)),
    Label('motorcycle',             32,     17,         13,    'vehicle',       7, True, False, (0, 0, 230)),
    Label('bicycle',                33,     18,         14,    'vehicle',       7, True, False, (119, 11, 32)),
    Label('license plate',          -1,     -1,         15,    'vehicle',       7, False, True, (0, 0, 142)),
]

if __name__ == '__main__':
    id_to_myid         = {label.id:      label.myId for label in labels}
    trainid_to_myid    = {label.trainId: label.myId for label in labels}

    print(id_to_myid)
    print(trainid_to_myid)

    progress = progressbar.ProgressBar(widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                                progressbar.Percentage(), ' ',
                                                progressbar.ETA()])

    # choose to convert bdd100k or cityscapes
    dataset = ''
    while dataset not in ['b', 'c']:
        dataset = input('[b]dd100k or [c]ityscapes: ')

    if dataset == 'b':  # bdd100k
        labels_path = os.path.join("bdd100k", "seg", "labels", "*", "*id.png")
        images = glob.glob(labels_path)

        f_trainid_to_myid = np.vectorize(lambda x: trainid_to_myid[x])

        for img_path in progress(images):
            image = plt.imread(img_path) * 255
            image = f_trainid_to_myid(image)

            new_image_path = os.path.join('converted', img_path)
            dirname = os.path.dirname(new_image_path)
            os.makedirs(dirname, exist_ok=True)
            Image.fromarray(image.astype('uint8')).save(new_image_path)

    elif dataset == 'c':  # cityscapes
        images = []
        if os.path.exists("gtFine"):
            labels_path = os.path.join("gtFine", "*", "*", "*_labelIds.png")
            images.extend(glob.glob(labels_path))

        if os.path.exists("gtCoarse"):
            labels_path = os.path.join("gtCoarse", "val", "*", "*_labelIds.png")
            images.extend(glob.glob(labels_path))
            labels_path = os.path.join("gtCoarse", "train", "*", "*_labelIds.png")
            images.extend(glob.glob(labels_path))

        f_id_to_myid = np.vectorize(lambda x: id_to_myid[x])

        for img_path in progress(images):
            image = plt.imread(img_path) * 255
            image = f_id_to_myid(image)

            new_image_path = os.path.join('converted', img_path)
            dirname = os.path.dirname(new_image_path)
            os.makedirs(dirname, exist_ok=True)
            Image.fromarray(image.astype('uint8')).save(new_image_path)

