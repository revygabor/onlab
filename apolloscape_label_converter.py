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
])

labels = [
    #      name                     id      trainId     myId
    Label('others',                 0,      255,        15,), # other
    Label('rover',                  1,      255,        9, ), # car
    Label('sky',                    17,     0,          7, ), # sky
    Label('car',                    33,     1,          9, ), # car
    Label('car_groups',             161,    1,          9, ), # car
    Label('motorbicycle',           34,     2,          13,), # motorcycle
    Label('motorbicycle_group',     162,    2,          13,), # motorcycle
    Label('bicycle',                35,     3,          14,), # bicycle
    Label('bicycle_group',          163,    3,          14,), # bicycle
    Label('person',                 36,     4,          8, ), # person
    Label('person_group',           164,    4,          8, ), # person
    Label('rider',                  37,     5,          8, ), # person
    Label('rider_grup',             165,    5,          8, ), # person
    Label('truck',                  38,     6,          10,), # truck
    Label('truck_group',            166,    6,          10,), # truck
    Label('bus',                    39,     7,          11,), # bus
    Label('bus_group',              167,    7,          11,), # bus
    Label('tricycle',               40,     8,          14,), # bicycle
    Label('tricycle_group',         168,    8,          14,), # bicycle
    Label('road',                   49,     9,          0, ), # road
    Label('sidewalk',               50,     10,         1, ), # sidewalk
    Label('traffic_cone',           65,     11,         15,), # other
    Label('road_pile',              66,     12,         3, ), # pole
    Label('fence',                  67,     13,         2, ), # building/wall/fence
    Label('traffic_light',          81,     14,         4, ), # traffic light
    Label('pole',                   82,     15,         3, ), # pole
    Label('traffic_sign',           83,     16,         5, ), # traffic sign
    Label('wall',                   84,     17,         2, ), # building/wall/fence
    Label('dustbin',                85,     18,         15,), # other
    Label('billboard',              86,     19,         2, ), # building/wall/fence
    Label('building',               97,     20,         2, ), # building/wall/fence
    Label('bridge',                 98,     255,        15,), # other
    Label('tunnel',                 99,     255,        15,), # other
    Label('overpass',               100,    255,        15,), # other
    Label('vegetation',             113,    21,         6, ), # vegetation
    Label('unlabeled',              255,    255,        15,), # other
]


if __name__ == '__main__':
    id_to_myid         = {label.id:      label.myId for label in labels}
    # trainid_to_myid    = {label.trainId: label.myId for label in labels}

    print(id_to_myid)
    # print(trainid_to_myid)

    progress = progressbar.ProgressBar(widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                                progressbar.Percentage(), ' ',
                                                progressbar.ETA()])

    labels_path = os.path.join("apolloscape", "Label", "*", "*", "*_bin.png")
    images = (glob.glob(labels_path))

    f_id_to_myid = np.vectorize(lambda x: id_to_myid[x])

    for img_path in progress(images):
        image = plt.imread(img_path) * 255
        image = f_id_to_myid(image)

        new_image_path = os.path.join('converted', img_path)
        dirname = os.path.dirname(new_image_path)
        os.makedirs(dirname, exist_ok=True)
        Image.fromarray(image.astype('uint8')).save(new_image_path)

