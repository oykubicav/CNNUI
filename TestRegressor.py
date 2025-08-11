import torch
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader
from StarBoxCNN import StarBoxCNN
from StarBoxDatasetMulti import StarBoxDatasetMulti
from torch import nn


def visualize(model, dataset, device, filename="test_visualization.png", num_samples=5):
    model.eval()
    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        patch, offset, (x, y), (sx, sy), _ = dataset[i]
        patch = patch.to(device).unsqueeze(0)
        pred = model(patch).detach().cpu()[0]
        dx_pred, dy_pred = pred[0] * 16, pred[1] * 16
        px, py = x + dx_pred, y + dy_pred

        img = patch[0][0].cpu().numpy()
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img, cmap='gray')
        plt.scatter([sx - x + 16], [sy - y + 16], c='lime', label='GT', marker='x')
        plt.scatter([px - x + 16], [py - y + 16], c='red', label='Pred', marker='o')
        plt.axis('off')

    plt.suptitle("Green = GT, Red = Prediction", fontsize=12)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Test görselleştirmesi kaydedildi: {filename}")

def visualize_all(model, dataset, device, max_offset=16, save_dir="predictions"):
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    for i in range(len(dataset)):
        patch, offset, (x, y), (sx, sy), _ = dataset[i]
        pred = model(patch.unsqueeze(0).to(device)).detach().cpu()[0]
        dx_pred, dy_pred = pred[0] * max_offset, pred[1] * max_offset
        px, py = x + dx_pred, y + dy_pred

        img = patch[0].numpy()
        plt.figure()
        plt.imshow(img, cmap='gray')
        plt.scatter([sx - x + 16], [sy - y + 16], c='lime', label='GT', marker='x')
        plt.scatter([px - x + 16], [py - y + 16], c='red', label='Pred', marker='o')
        plt.legend()
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"pred_{i:03}.png"))
        plt.close()

def test(model_path="best_model.pth", test_folder="data/stars/test"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = StarBoxCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_dataset = StarBoxDatasetMulti(test_folder, patch_size=32, num_negative=10)
    test_loader = DataLoader(test_dataset, batch_size=32)

    criterion = nn.MSELoss()
    total_loss = 0

    with torch.no_grad():
        for x, y, *_ in test_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            total_loss += criterion(y_pred, y).item() * x.size(0)

    avg_loss = total_loss / len(test_loader.dataset)
    print(f"\n🧪 Test Loss (MSE): {avg_loss:.4f}")

    visualize_all(model, test_dataset, device)


if __name__ == "__main__":
    test()
