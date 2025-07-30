from torch import nn


class Network(nn.Module):
    def __init__(self, num_classes=94):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> (128, 256, 256)

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> (256, 128, 128)

            # Global average pool to 1×1
            nn.AdaptiveAvgPool2d((1, 1))  # -> (256, 1, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),  # -> (256,)
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
