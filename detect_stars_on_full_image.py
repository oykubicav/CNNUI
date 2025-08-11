import cv2
import numpy as np
import torch
from torchvision import transforms
from cnnmodel import CNNModel
from sliding_window import sliding_window 

def detect_stars_on_full_image(model, image_path, device='cpu', threshold=0.5):
    model.eval()
    model.to(device)
    transform = transforms.ToTensor()
    patches, coords = sliding_window(image_path)

    star_coordinates = []
    with torch.no_grad():
        for patch, (x, y) in zip(patches, coords):
            patch_tensor = transform(patch).unsqueeze(0).to(device)  
            y_pred = model(patch_tensor)
            prob = torch.sigmoid(y_pred).item()
            if prob > threshold:
                star_coordinates.append((x, y))

    return star_coordinates
