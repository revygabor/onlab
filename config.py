import os


class DataSetConfig:
    dataset_name = ''
    train_images = ''
    train_labels = ''
    val_images = ''
    val_labels = ''
    # only for ApolloScape:
    train_image_dirs = ''
    val_image_dirs = ''
    test_image_dirs = ''
    train_label_dirs = ''
    val_label_dirs = ''
    test_label_dirs = ''


CITYSCAPES_TRAIN_IMAGES = os.path.join("leftImg8bit", "train", "*", "*_leftImg8bit.png")
CITYSCAPES_TRAIN_FINE_LABELS = os.path.join("converted", "gtFine", "train", "*", "*_labelIds.png")
CITYSCAPES_TRAIN_COARSE_LABELS = os.path.join("converted", "gtCoarse", "train", "*", "*_labelIds.png")

CITYSCAPES_VAL_IMAGES = os.path.join("leftImg8bit", "val", "*", "*_leftImg8bit.png")
CITYSCAPES_VAL_FINE = "cityscapes_val_fine"
CITYSCAPES_VAL_FINE_LABELS = os.path.join("converted", "gtFine", "val", "*", "*_labelIds.png")
CITYSCAPES_VAL_COARSE = "cityscapes_val_coarse"
CITYSCAPES_VAL_COARSE_LABELS = os.path.join("converted", "gtCoarse", "val", "*", "*_labelIds.png")

BDD100K_TRAIN_IMAGES = os.path.join("bdd100k", "seg", "images", "train", "*.jpg")
BDD100K_TRAIN_LABELS = os.path.join("converted", "bdd100k", "seg", "labels", "train", "*.png")

BDD100K_VAL = "bdd100k_val"
BDD100K_VAL_IMAGES = os.path.join("bdd100k", "seg", "images", "val", "*jpg")
BDD100K_VAL_LABELS = os.path.join("converted", "bdd100k", "seg", "labels", "val", "*.png")

APOLLOSCAPE = "apolloscape"
APOLLOSCAPE_TRAIN_IMAGE_DIRS = os.path.join("apolloscape", "ColorImage", "train", "*", "*")  # *.jpg
APOLLOSCAPE_VAL_IMAGE_DIRS = os.path.join("apolloscape", "ColorImage", "val", "*", "*")  # *.jpg
APOLLOSCAPE_TEST_IMAGE_DIRS = os.path.join("apolloscape", "ColorImage", "test", "*", "*")  # *.jpg
APOLLOSCAPE_TRAIN_LABEL_DIRS = os.path.join("converted", "apolloscape", "Label", "train", "*", "*")  # *.png
APOLLOSCAPE_VAL_LABEL_DIRS = os.path.join("converted", "apolloscape", "Label", "val", "*", "*")  # *.png
APOLLOSCAPE_TEST_LABEL_DIRS = os.path.join("converted", "apolloscape", "Label", "test", "*", "*")  # *.png


class CityScapesFineConfig(DataSetConfig):
    dataset_name = 'cityscapes_fine'
    train_images = CITYSCAPES_TRAIN_IMAGES
    train_labels = CITYSCAPES_TRAIN_FINE_LABELS
    val_images = CITYSCAPES_VAL_IMAGES
    val_labels = CITYSCAPES_VAL_FINE_LABELS


class CityScapesCoarseConfig(DataSetConfig):
    dataset_name = 'cityscapes_coarse'
    train_images = CITYSCAPES_TRAIN_IMAGES
    train_labels = CITYSCAPES_TRAIN_COARSE_LABELS
    val_images = CITYSCAPES_VAL_IMAGES
    val_labels = CITYSCAPES_VAL_COARSE_LABELS


class BDD100KConfig(DataSetConfig):
    dataset_name = 'bdd100k'
    train_images = BDD100K_TRAIN_IMAGES
    train_labels = BDD100K_TRAIN_LABELS
    val_images = BDD100K_VAL_IMAGES
    val_labels = BDD100K_VAL_LABELS


class ApolloScapeConfig(DataSetConfig):
    dataset_name = 'apolloscape'
    train_image_dirs = APOLLOSCAPE_TRAIN_IMAGE_DIRS
    val_image_dirs = APOLLOSCAPE_VAL_IMAGE_DIRS
    test_image_dirs = APOLLOSCAPE_TEST_IMAGE_DIRS
    train_label_dirs = APOLLOSCAPE_TRAIN_LABEL_DIRS
    val_label_dirs = APOLLOSCAPE_VAL_LABEL_DIRS
    test_label_dirs = APOLLOSCAPE_TEST_LABEL_DIRS
