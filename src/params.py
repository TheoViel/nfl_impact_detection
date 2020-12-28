import torch
import numpy as np

NUM_WORKERS = 8

DATA_PATH = "../input/"
LOG_PATH = "../logs/"
LOG_PATH_CLS = "../logs_cls/"
OUT_DIR = "../output/"

SIZE = 512
IMG_SHAPE = (720, 1280)

TRAIN_VID_PATH = DATA_PATH + "train/"

IMG_PATH_F = "../../../data/nfl/"
IMG_PATH = "../../../data/nfl_512/"
CROP_PATH = "../../../data/nfl_crops/"
CROP_PATH_3D = "../../../data/nfl_crops_3d/"

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
