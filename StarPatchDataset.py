import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import random
import os


    # StarPatchDataset.py

class StarPatchDataset(Dataset):
    def __init__(self, img_path, coord_csv=None, patch_size=32, num_negative=10, transform=None, use_unrelated=False, unrelated_path=None):
        self.patches = []
        self.labels = []
        self.positions = []  # (x, y, filename)
        self.transform = transform
        self.patch_size = patch_size

        if not use_unrelated:
            self.image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            self.h, self.w = self.image.shape

            filename = os.path.basename(img_path)

            df = pd.read_csv(coord_csv, header=None, names=['class', 'x', 'y', 'w', 'h', 'brightness'])
            df = df[df['x'].astype(str).str.contains(r'^\d', regex=True)]
            df['x'] = pd.to_numeric(df['x'], errors='coerce')
            df['y'] = pd.to_numeric(df['y'], errors='coerce')
            df = df.dropna(subset=['x', 'y'])
            self.star_coords = df[['x', 'y']].to_numpy()

            for cx, cy in self.star_coords:
                x = int(cx * self.w)
                y = int(cy * self.h)
                if self.is_valid_patch(x, y):
                    patch = self.extract_patch(x, y)
                    self.patches.append(patch)
                    self.labels.append(1.0)
                    self.positions.append((x, y, filename))  # ✅ filename eklendi

            negatives, tries = 0, 0
            while negatives < num_negative and tries < num_negative * 20:
                x = random.randint(patch_size // 2, self.w - patch_size // 2)
                y = random.randint(patch_size // 2, self.h - patch_size // 2)
                if self.is_valid_patch(x, y) and not self.is_near_star(x, y):
                    patch = self.extract_patch(x, y)
                    self.patches.append(patch)
                    self.labels.append(0.0)
                    self.positions.append((x, y, filename))
                    negatives += 1
                tries += 1

        else:
            unrelated_img = cv2.imread(unrelated_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            h, w = unrelated_img.shape
            filename = os.path.basename(unrelated_path)

            for _ in range(num_negative):
                x = np.random.randint(patch_size // 2, w - patch_size // 2)
                y = np.random.randint(patch_size // 2, h - patch_size // 2)
                patch = unrelated_img[y - patch_size//2 : y + patch_size//2,
                                      x - patch_size//2 : x + patch_size//2]
                self.patches.append(patch)
                self.labels.append(0.0)
                self.positions.append((x, y, filename))  # ✅ filename eklendi

    def __getitem__(self, idx):
        patch = np.expand_dims(self.patches[idx], 0)
        label = self.labels[idx]
        x, y, fname = self.positions[idx]

        if self.transform:
            patch = self.transform(patch)

        return torch.tensor(patch, dtype=torch.float32), torch.tensor(label, dtype=torch.float32), (x, y, fname)


    def is_valid_patch(self, x, y):
        half = self.patch_size // 2
        return (x - half >= 0) and (x + half < self.w) and (y - half >= 0) and (y + half < self.h)

    def is_near_star(self, x, y, threshold=10):
        for sx, sy in self.star_coords:
            sx = int(sx * self.w)
            sy = int(sy * self.h)
            if np.sqrt((sx - x) ** 2 + (sy - y) ** 2) < threshold:
                return True
        return False

    def extract_patch(self, x, y):
        half = self.patch_size // 2
        return self.image[y - half : y + half, x - half : x + half]

    def __len__(self):
        return len(self.patches)

   