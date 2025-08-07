import torch
import cv2
import numpy as np
import pandas as pd
from cnnmodel import CNNModel
from StarBoxCNN import StarBoxCNN
from CNNBrightness import CNNBrightness
from sliding_window import sliding_window
from find_brightness import find_brightness  

def FullCNN(
    img_path,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    cls_thresh=0.5,
    save_path_img="detected_stars.png",
    save_path_csv="detected_stars.csv"
):
   
    clf_model = CNNModel().to(device)
    clf_model.load_state_dict(torch.load('best_clf_model.pth', map_location=device))
    clf_model.eval()

    reg_model = StarBoxCNN().to(device)
    reg_model.load_state_dict(torch.load('best_reg_model.pth', map_location=device))
    reg_model.eval()

    brightness_model = CNNBrightness().to(device)
    brightness_model.load_state_dict(torch.load('best_brightness_model.pth', map_location=device))
    brightness_model.eval()

    # 🔹 Görseli oku
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    color_img = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # 🔹 Sliding window ile patch'leri al
    patches, centers = sliding_window(image)
    results = []

    for patch, (cx, cy) in zip(patches, centers):
        patch_tensor = torch.tensor(patch).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 32, 32]

        with torch.no_grad():
            # 🔹 1. Sınıflandırma: yıldız var mı?
            cls_pred = clf_model(patch_tensor)
            cls_prob = torch.sigmoid(cls_pred).item()

            if cls_prob > cls_thresh:
                # 🔹 2. Offset regresyonu: dx, dy
                reg_pred = reg_model(patch_tensor)
                dx, dy = reg_pred[0].item(), reg_pred[1].item()

                star_x = cx + dx
                star_y = cy + dy

                # 🔹 3. Parlaklık tahmini
                brightness_pred = brightness_model(patch_tensor)
                brightness = brightness_pred.item()

                # 🔹 Opsiyonel: Doğrulama amaçlı gerçek brightness
                # brightness = find_brightness(image, int(star_x), int(star_y))

                print(f"⭐ Star @ ({star_x:.1f}, {star_y:.1f}) | Brightness: {brightness:.3f} | Confidence: {cls_prob:.3f}")

                # 🔹 Görsel üzerine çiz
                cv2.circle(color_img, (int(star_x), int(star_y)), 4, (0, 255, 0), -1)
                cv2.putText(
                    color_img,
                    f"{brightness:.2f}",
                    (int(star_x) + 5, int(star_y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1
                )

                # 🔹 Kayıt
                results.append({
                    "x": star_x,
                    "y": star_y,
                    "brightness": brightness,
                    "confidence": cls_prob
                })

    # 🔹 Sonuçları kaydet
    cv2.imwrite(save_path_img, color_img)
    pd.DataFrame(results).to_csv(save_path_csv, index=False)
    print(f"\n📄 {len(results)} star(s) detected. Results saved to {save_path_csv} and {save_path_img}.")
