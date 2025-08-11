import torch.nn as nn

class CNNBrightness(nn.Module):
    def __init__(self):
        super(CNNBrightness, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # [B, 1, 32, 32] -> [B, 16, 32, 32]
            nn.ReLU(),
            nn.MaxPool2d(2),                            # -> [B, 16, 16, 16]
            nn.Conv2d(16, 32, kernel_size=3, padding=1),# -> [B, 32, 16, 16]
            nn.ReLU(),
            nn.MaxPool2d(2),                            # -> [B, 32, 8, 8]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),# -> [B, 64, 8, 8]
            nn.ReLU(),
            nn.MaxPool2d(2)                             # -> [B, 64, 4, 4]
        )

        self.classifier = nn.Sequential(            
            nn.Flatten(),                              # -> [B, 64*4*4]
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 1)                           
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
