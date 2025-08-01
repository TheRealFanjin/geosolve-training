from torch import nn


class RN50Model(nn.Module):
    def __init__(self, resnet_model, num_classes=92):
        super().__init__()
        self.resnet_model = resnet_model
        self.resnet_model.fc = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        # x: (batch size, num images, channels, height, width)
        b, n, c, h, w = x.shape
        x = x.view(b * n, c, h, w)
        features = self.resnet_model(x)
        features = features.view(b, n, -1)
        features = features.mean(dim=1)
        output = self.classifier(features)
        return output
