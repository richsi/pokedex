import torch
import torch.nn as nn

class SmallVGGNet(nn.Module):
    def __init__(self, width, height, depth, num_classes):
        super(SmallVGGNet, self).__init__()

        self.features = nn.Sequential(
            # CONV -> RELU -> POOL layer
            nn.Conv2d(depth, 32, kernel_size=(3,3), padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(3,3)),
            nn.Dropout(0.5),

            # (CONV -> RELU) * 2 -> POOL layer
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.25),

            # Same thing as above block
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.25),
        )

        feature_size = self._get_conv_output_size((depth, height, width))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Set FC -> ReLU layer
            nn.Linear(feature_size, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            # Softmax
            nn.Linear(1024, num_classes),
            nn.Softmax(dim=1)
        )

    def _get_conv_output_size(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feature = self.features(input)
        n_size = output_feature.data.view(batch_size, -1).size(1)
        return n_size

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x