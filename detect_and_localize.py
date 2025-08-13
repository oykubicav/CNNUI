import torch
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, maximum_filter

from cnnmodel import CNNModel
from StarBoxCNN import StarBoxCNN

def calculate_brightness(gray_image, x, y, radius=5):
    h, w = gray_image.shape
    y_grid, x_grid = np.ogrid[:h, :w]
    mask = (x_grid - x)**2 + (y_grid - y)**2 <= radius**2
    return gray_image[mask].mean()

def load_models(device, classifier_path="star_classifier.pth", regressor_path="best_model.pth"):
    classifier = CNNModel().to(device)
    regressor = StarBoxCNN().to(device)

    classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    regressor.load_state_dict(torch.load(regressor_path, map_location=device))

    classifier.eval()
    regressor.eval()
    return classifier, regressor

def accumulate_heatmap(h, w, detections, sigma=1.5, splat_strength="prob"):
    
    heat = np.zeros((h, w), dtype=np.float32)
    for d in detections:
        x, y = d["pos"]
        v = float(d.get(splat_strength, 1.0))
        ix, iy = int(round(x)), int(round(y))
        if 0 <= iy < h and 0 <= ix < w:
            heat[iy, ix] += v

    heat = gaussian_filter(heat, sigma=sigma)

    m = heat.max()
    if m > 0:
        heat /= m
    return heat

def nms_on_heatmap(heat, thr=0.2, min_distance=6):
    #calcs max value for every pixels min*min surr.
    neighborhood = maximum_filter(heat, size=min_distance)
    peaks_mask = (heat == neighborhood) & (heat >= thr)

    ys, xs = np.where(peaks_mask)
    peaks = [(float(x), float(y), float(heat[y, x])) for x, y in zip(xs, ys)]
    peaks.sort(key=lambda t: t[2], reverse=True)
    return peaks

def detect_multiple_stars(image_path, device, classifier_model, regressor_model,
                          patch_size=32, stride=8, prob_thr=0.2,
                          sigma=1.5, thr=0.25, min_distance=6):
  
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    h, w = gray.shape
    half = patch_size // 2

    raw_detections = []


    for y in range(half, h - half + 1, stride):
        for x in range(half, w - half + 1, stride):
            patch = gray[y - half:y + half, x - half:x + half]
            if patch.shape != (patch_size, patch_size):
                #fill if patch is smaller
                patch = cv2.copyMakeBorder(
                    patch,
                    top=0, bottom=patch_size - patch.shape[0],
                    left=0, right=patch_size - patch.shape[1],
                    borderType=cv2.BORDER_REFLECT
                )

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

            raw_detections.append({
                "pos": (star_x, star_y),
                "offset": (dx, dy),
                "center": (x, y),
                "brightness": brightness,
                "prob": prob
            })

    heat = accumulate_heatmap(h, w, raw_detections, sigma=sigma, splat_strength="prob")

    peaks = nms_on_heatmap(heat, thr=thr, min_distance=min_distance)

    final_detections = []
    for px, py, score in peaks:
        b = calculate_brightness(gray, px, py, radius=5)
        final_detections.append({
            "pos": (px, py),
            "offset": (0.0, 0.0),     
            "center": (px, py),
            "brightness": b,
            "prob": score
        })

    return final_detections
