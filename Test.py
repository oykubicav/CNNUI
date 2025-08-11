import torch
from torch import nn
from cnnmodel import CNNModel
from StarPatchDatasetMulti import StarPatchDatasetMulti
from torch.utils.data import DataLoader
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
P = 32
THRESHOLD = 0.3  


model = CNNModel().to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()


test = StarPatchDatasetMulti(
    folder1_path="/Users/oykubicav/Desktop/CNN/data/stars/test",
    folder2_path="/Users/oykubicav/Desktop/CNN/data/unrel/test",
    patch_size=P,
    num_negative=1000
)
test_loader = DataLoader(test, batch_size=1, shuffle=False)


criterion = nn.BCEWithLogitsLoss()
total_loss, predictions, targets = 0.0, [], []
box_dict = defaultdict(list)  


with torch.no_grad():
    for x, y, (px, py, img_name) in test_loader:
        x, y = x.to(device), y.to(device)
        y_pred = model(x)

        prob = torch.sigmoid(y_pred.view(-1))
        print(f"{img_name[0]} → prob: {prob.item():.4f}")

        loss = criterion(y_pred.view(-1), y.view(-1))

        total_loss += loss.item()
        predictions.extend(prob.cpu().numpy())
        targets.extend(y.view(-1).cpu().numpy())

        if prob.item() > THRESHOLD:
            box_dict[img_name[0]].append((px.item(), py.item()))


predictions = np.array(predictions)
targets = np.array(targets)
mae = np.mean(np.abs(predictions - targets))

print(f"\n Test Loss: {total_loss / len(test_loader):.4f}")
print(f" Mean Absolute Error: {mae:.4f}")
print(f" Total Detected Boxes: {sum(len(b) for b in box_dict.values())}")
print(f" Detected Images: {list(box_dict.keys())}")


all_boxes = []
for fname, boxes in box_dict.items():
    for x, y in boxes:
        all_boxes.append((fname, x, y))
if all_boxes:
    pd.DataFrame(all_boxes, columns=['image', 'x', 'y']).to_csv("detected_boxes.csv", index=False)
    print("Detected boxes saved to detected_boxes.csv")
else:
    print("No boxes to save in CSV.")


image_folder = "/Users/oykubicav/Desktop/CNN/data/stars/test"
for fname, boxes in box_dict.items():
    img_path = os.path.join(image_folder, fname)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Could not read image: {img_path}")
        continue
    for x, y in boxes:
        cv2.rectangle(img, (int(x - P//2), int(y - P//2)), (int(x + P//2), int(y + P//2)), (0, 255, 0), 1)
    out_name = f"detections_{fname}"
    cv2.imwrite(out_name, img)
    print(f"Saved: {out_name}")


try:
    data = np.load('loss_curves.npz')
    train_losses = data['train_losses']
    val_losses = data['val_losses']

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.scatter(targets, predictions, alpha=0.6)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.title('Predicted vs True')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300)
    print("📊 Saved training_results.png")
    plt.close()

except Exception as e:
    print("Could not plot training curves:", e)
