import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from StarBoxCNN import StarBoxCNN
from StarBoxDataset import StarBoxDataset
import numpy as np
from StarBoxDatasetMulti import StarBoxDatasetMulti

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = StarBoxCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    train_losses, val_losses = [], []
    train_dataset = StarBoxDatasetMulti(folder_path='/Users/oykubicav/Desktop/CNN2/data/stars/train', patch_size=32, num_negative=10)
    val_dataset = StarBoxDatasetMulti(folder_path='/Users/oykubicav/Desktop/CNN2/data/stars/val', patch_size=32, num_negative=10)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(20):
        model.train()
        total_train_loss = 0

        for x, y, *_ in train_loader:   # *_ tüm ekstra öğeleri yoksayar

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_pred = model(x)  
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * x.size(0)
    
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")


        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x, y, *_ in val_loader:
                x = x.to(device)
                y = y.to(device)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                total_val_loss += loss.item() * x.size(0)

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1}, Val Loss: {avg_val_loss:.4f}")
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            np.savez('loss2_curves.npz', train_losses=train_losses, val_losses=val_losses)
            torch.save(model.state_dict(), "best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("No improvement.")
                break

if __name__ == "__main__":
    train()
