import cv2
import numpy as np

def is_valid_patch(x, y, image_shape, patch_size):
    h, w = image_shape
    return 0 <= x and x + patch_size <= w and 0 <= y and y + patch_size <= h

def sliding_window(image_path, size=32, stride=16):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0

    patches = []
    coords = []

    for y in range(0, img.shape[0] - size + 1, stride):
        for x in range(0, img.shape[1] - size + 1, stride):
            if is_valid_patch(x, y, img.shape, size):
                patch = img[y:y+size, x:x+size]
                patches.append(patch)
                coords.append((x + size//2, y + size//2)) 

    return np.array(patches), coords
