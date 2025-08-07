import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from cnnmodel import CNNModel
from StarPatchDatasetMulti import StarPatchDatasetMulti


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNModel().to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0], device=device))

    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    train = StarPatchDatasetMulti(
        folder1_path = "/Users/oykubicav/Desktop/CNN/data/stars/train",
        folder2_path = "/Users/oykubicav/Desktop/CNN/data/unrel/train",
        patch_size=32,
        num_negative=10
    )

    # 🟢 folder2_path parametresi verilmedi (None da verilmedi)
    val = StarPatchDatasetMulti(
        folder1_path='/Users/oykubicav/Desktop/CNN/data/stars/val',
        patch_size=32,
        num_negative=10
    )

    print("Train sets:", len(train.datasets))
    print("Total train patches:", len(train))
    train_loader = DataLoader(train, batch_size=32, shuffle=True)
    val_loader = DataLoader(val, batch_size=32, shuffle=False)
    total_pos = sum([sum(ds.labels) for ds in train.datasets])
    total = sum([len(ds.labels) for ds in train.datasets])
    print(f"Pozitif patch sayısı: {int(total_pos)} / {total} → oran: {total_pos / total:.4f}")

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    num_epochs = 50
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0

        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x).squeeze()
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * x.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct_train += (preds == y).sum().item()
            total_train += y.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_accuracy = correct_train / total_train
        train_losses.append(avg_train_loss)

        
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x).squeeze()
                loss = criterion(logits, y)
                total_val_loss += loss.item() * x.size(0)

                preds = (torch.sigmoid(logits) > 0.5).float()
                correct_val += (preds == y).sum().item()
                total_val += y.size(0)

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_accuracy = correct_val / total_val
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, "
              f"Train Acc = {train_accuracy:.4f}, Val Acc = {val_accuracy:.4f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            np.savez('loss_curves.npz', train_losses=train_losses, val_losses=val_losses)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("No improvement")
                break

    print(f"Final Validation Accuracy: {val_accuracy:.4f}")

    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title("Loss Curves")
    plt.savefig("loss_plot.png")
    plt.close()

if __name__ == "__main__":
    train()
