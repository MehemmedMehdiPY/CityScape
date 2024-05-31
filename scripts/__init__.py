import numpy as np
from .data_preparation import CityScapeDataset
from .train import Trainer
from .loss import DiceLoss

CLASSES = (
    'unlabeled',
    'dynamic',
    'ground',
    'road',
    'rail track',
    'building',
    'fence',
    'pole',
    'vegetation',
    'sky',
    'person',
    'vehicle'
 )

COLORS = np.array(
    [[0, 0, 0],
    [111, 74, 0],
    [81, 0, 81],
    [128, 64, 128],
    [230, 150, 140],
    [70, 70, 70],
    [190, 153, 153],
    [153, 153, 153],
    [107, 142, 35],
    [70, 130, 180],
    [220, 20, 60],
    [0, 0, 142]]
 )

CAT_TO_COLOR = dict(zip(
    CLASSES, COLORS
))

DEVICE = 'cuda'
TRAIN_SIZE = 0.75
BATCH_SIZE = 8
SEED = 42
EPOCHS = 10