import warnings 
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

from scripts import COLORS, DEVICE, TRAIN_SIZE, BATCH_SIZE, SEED, EPOCHS
from scripts import CityScapeDataset

from torch.utils.data import DataLoader
import torch
import albumentations as A

print('Starting...')
print('Data is being initialized')

transform = A.Compose([
    A.Rotate(limit=15, p=1)
])

transform = None

transform_image = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(0.2, 0.3), p=1),
    A.CLAHE(clip_limit=4, p=1),
    A.Blur(blur_limit=3, p=1)
])

dataset = CityScapeDataset(root='../data/train', colors=COLORS, to_loader=True, transform=transform,
                                transform_image=transform_image, num_classes = 20, train_size=TRAIN_SIZE, seed=SEED)

image, mask = dataset[0]

print(image.shape)
print(mask.shape)

image = image.numpy().transpose([1, 2, 0])
mask = COLORS[mask.argmax(0).numpy()]

print(image.shape, mask.shape)

plt.imshow(image)
plt.show()

plt.imshow(mask)
plt.show()