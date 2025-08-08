import torch
import numpy as np
import cv2
from cnnmodel import CNNModel
from StarBoxCNN import StarBoxCNN

def calculate_brightness(gray_image, x, y, radius=5):
    h, w = gray_image.shape
    y_grid, x_grid = np.ogrid[:h, :w]
    mask = (x_grid - x)**2 + (y_grid - y)**2 <= radius**2
    brightness = gray_image[mask].mean()
    return brightness

def load_models(device, classifier_path="star_classifier.pth", regressor_path="best_model.pth"):
    classifier = CNNModel().to(device)
    regressor = StarBoxCNN().to(device)

    classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    regressor.load_state_dict(torch.load(regressor_path, map_location=device))

    classifier.eval()
    regressor.eval()
    return classifier, regressor
import math

def filter_duplicates(detections, distance_thresh=3.0):
    filtered = []
    for det in detections:
        x1, y1 = det["pos"]
        too_close = False
        for kept in filtered:
            x2, y2 = kept["pos"]
            dist = math.hypot(x1 - x2, y1 - y2)
            if dist < distance_thresh:
                too_close = True
                break
        if not too_close:
            filtered.append(det)
    return filtered


def detect_multiple_stars(image_path, device, classifier_model, regressor_model,
                          patch_size=32, stride=8, prob_thr=0.2):
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    h, w = gray.shape
    half = patch_size // 2

    detected_stars = []

    for y in range(half, h - half + 1, stride):
        for x in range(half, w - half + 1, stride):
            patch = gray[y - half:y + half, x - half:x + half]
            if patch.shape != (patch_size, patch_size):
                patch = cv2.copyMakeBorder(
                patch,
                top=0, bottom=patch_size - patch.shape[0],
                left=0, right=patch_size - patch.shape[1],
                borderType=cv2.BORDER_REFLECT)

            patch_tensor = torch.tensor(patch).unsqueeze(0).unsqueeze(0).float().to(device)

            with torch.no_grad():
                prob = torch.sigmoid(classifier_model(patch_tensor)).item()
            if prob < prob_thr:
                continue

            with torch.no_grad():
                offset = regressor_model(patch_tensor)[0].cpu().numpy() * half
                dx, dy = offset

                star_x = np.clip(x + dx, 0, w - 1)
                star_y = np.clip(y + dy, 0, h - 1)

            brightness = calculate_brightness(gray, star_x, star_y, radius=5)

            detected_stars.append({
                "pos": (star_x, star_y),
                "offset": (dx, dy),
                "center": (x, y),
                "brightness": brightness,
                "prob": prob
            })

    return filter_duplicates(detected_stars)

