import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallVGGNet(nn.Module):
    def __init__(self, width, height, depth, num_classes):
        super(SmallVGGNet, self).__init__()

        input_shape = (depth, height, width)

        self.features = nn.Sequential(
            # CONV -> RELU -> POOL layer
            nn.Conv2d(input_shape[0], 32, kernel_size=(3,3), padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(pool_size=(3,3)),
            nn.Dropout(0.5),

            # (CONV -> RELU) * 2 -> POOL layer
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(pool_size=(2,2)),
            nn.Dropout(0.25),

            # Same thing as above block
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(pool_size=(2,2)),
            nn.Dropout(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Set FC -> ReLU layer
            nn.Linear(128 * (width // 8) * (height //8), 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            # Softmax
            nn.Linear(1024, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x