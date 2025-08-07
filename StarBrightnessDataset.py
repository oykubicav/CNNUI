import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import pandas as pd

class StarBrightnessDataset(Dataset):
    def __init__(self, img_path, coord_csv, patch_size=32, num_offset=100, transform=None):
        self.image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        self.h, self.w = self.image.shape
        self.patch_size = patch_size
        self.transform = transform
        self.patches = []
        self.positions = []
        self.brightness = []

        coor = pd.read_csv(coord_csv, header=None, names=['class', 'x', 'y', 'w', 'h', 'brightness'])
        self.star_coords = [(int(x * self.w), int(y * self.h), b) for x, y, b in coor[['x', 'y', 'brightness']].values]

        count, attempts = 0, 0
        while count < num_offset and attempts < num_offset * 10:
            x = np.random.randint(patch_size // 2, self.w - patch_size // 2)
            y = np.random.randint(patch_size // 2, self.h - patch_size // 2)
            if not self.is_valid_patch(x, y):
                attempts += 1
                continue

            dx, dy, star_x, star_y, brightness = self.find_closest_star_offset(x, y)


            if dx is None:  # No star inside patch
                attempts += 1
                continue

            patch = self.image[y - patch_size // 2:y + patch_size // 2,
                               x - patch_size // 2:x + patch_size // 2]

            self.patches.append(patch)
            self.positions.append((x, y))
            self.brightness.append(brightness)

            count += 1
            attempts += 1

    def is_valid_patch(self, x, y):
        half = self.patch_size // 2
        return (x - half >= 0) and (x + half < self.w) and (y - half >= 0) and (y + half < self.h)

    def find_closest_star_offset(self, x, y):
        half = self.patch_size // 2
        min_dist = float('inf')
        closest_offset = (None, None)
        star_pos = (None, None)
        brightness = None

        for sx, sy, b in self.star_coords:
            if abs(sx - x) > half or abs(sy - y) > half:
                continue
            dx = sx - x
            dy = sy - y
            dist = dx ** 2 + dy ** 2

            if dist < min_dist:
                min_dist = dist
                closest_offset = (dx, dy)
                star_pos = (sx, sy)
                brightness = b

        return (*closest_offset, *star_pos, brightness)


    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = np.expand_dims(self.patches[idx], axis=0)
        brightness = self.brightness[idx]
        return (
            torch.tensor(patch, dtype=torch.float32),
            torch.tensor(brightness, dtype=torch.float32),
        )
