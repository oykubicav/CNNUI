import cv2
import numpy as np
def draw_detections(image, coords, patch_size=32):
    half = patch_size // 2
    for (x, y) in coords:
        top_left = (x - half, y - half)
        bottom_right = (x + half, y + half)
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
    return image
