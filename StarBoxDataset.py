import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import pandas as pd

class StarBoxDataset(Dataset):
    def __init__(self, img_path, coord_csv, patch_size=32, num_offset=10):
        self.image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        self.h, self.w = self.image.shape
        self.patch_size = patch_size
        self.max_offset = patch_size // 2
        self.patches, self.offsets, self.positions, self.stars, self.brightness = [], [], [], [], []

        coor = pd.read_csv(coord_csv, header=None, names=['class', 'x', 'y', 'w', 'h', 'brightness'])
        self.star_coords = [(int(x * self.w), int(y * self.h), float(b))
                            for x, y, b in coor[['x', 'y', 'brightness']].values]

        count, attempts = 0, 0
        while count < num_offset and attempts < num_offset * 10:
            x = np.random.randint(self.max_offset, self.w - self.max_offset)
            y = np.random.randint(self.max_offset, self.h - self.max_offset)
            dx, dy, sx, sy, b = self.closest_star(x, y)
            if dx is None: attempts += 1; continue
            patch = self.image[y - self.max_offset:y + self.max_offset, x - self.max_offset:x + self.max_offset]
            self.patches.append(patch)
            self.offsets.append((dx / self.max_offset, dy / self.max_offset))
            self.positions.append((x, y))
            self.stars.append((sx, sy))
            self.brightness.append(b)
            count += 1
            attempts += 1

    def closest_star(self, x, y):
        min_d, res = float('inf'), (None, None, None, None, 0)
        for sx, sy, b in self.star_coords:
            if abs(sx - x) > self.max_offset or abs(sy - y) > self.max_offset:
                continue
            dx, dy = sx - x, sy - y
            d = dx**2 + dy**2
            if d < min_d:
                res = (dx, dy, sx, sy, b)
                min_d = d
        return res

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.patches[idx][None], dtype=torch.float32),
            torch.tensor(self.offsets[idx], dtype=torch.float32),
            self.positions[idx],
            self.stars[idx],
            torch.tensor(self.brightness[idx], dtype=torch.float32)
        )
