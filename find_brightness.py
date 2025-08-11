import cv2
import numpy as np

def find_brightness(img_path, star_x, star_y, radius=5):

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Görüntü okunamadı: {img_path}")

    h, w = img.shape  


    y_grid, x_grid = np.ogrid[:h, :w]

    
    mask = (x_grid - star_x) ** 2 + (y_grid - star_y) ** 2 <= radius ** 2

    # Maskelenmiş bölgedeki parlaklığı hesapla
    brightness = np.mean(img[mask])

    return brightness
