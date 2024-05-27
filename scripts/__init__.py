import numpy as np
from .data_preparation import CityScapeDataset
from .train import Trainer
from .loss import DiceLoss

CLASSES = (
    'unlabeled',
    'dynamic',
    'ground',
    'road',
    'sidewalk',
    'parking',
    'rail track',
    'building',
    'wall',
    'fence',
    'guard rail',
    'bridge',
    'tunnel',
    'pole',
    'traffic sign',
    'vegetation',
    'terrain',
    'sky',
    'person',
    'vehicle'
)

COLORS = np.array(
    (
    (0, 0, 0),
    (111, 74, 0),
    (81, 0, 81),
    (128, 64, 128),
    (244, 35, 232),
    (250, 170, 160),
    (230, 150, 140),
    (70, 70, 70),
    (102, 102, 156),
    (190, 153, 153),
    (180, 165, 180),
    (150, 100, 100),
    (150, 120, 90),
    (153, 153, 153),
    (220, 220, 0),
    (107, 142, 35),
    (152, 251, 152),
    (70, 130, 180),
    (220, 20, 60),
    (0, 0, 142)
    ))

CAT_TO_COLOR = dict(zip(
    CLASSES, COLORS
))

DEVICE = 'cuda'
TRAIN_SIZE = 0.75
BATCH_SIZE = 8
SEED = 42
EPOCHS = 10