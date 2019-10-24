import os

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
BDD100K_VAL_LABELS = os.path.join("converted", "bdd100k", "seg", "labels", "val", "*.png")
BDD100K_VAL_IMAGES = os.path.join("bdd100k", "seg", "images", "val", "*jpg")
