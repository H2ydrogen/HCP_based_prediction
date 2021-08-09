from torchvision import models
from torch import nn
import torch.nn.functional as F
import torch


def ResNet50(classes, pretrained=True):
    model = models.resnet50(pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(2048, classes)
    return model


def DenseNet169(classes, pretrained=True):
    model = models.densenet169(pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Linear(1664, classes)
    return model


class CNN(nn.Module):
    def __init__(self, n_features, n_output):
        super(CNN, self).__init__()
        self.layer1 = nn.Linear(n_features, 1024)
        self.layer2 = nn.Linear(1024, 2048)
        self.layer3 = nn.Linear(2048, 4096)
        self.layer4 = nn.Linear(4096, 4096)
        self.output = nn.Linear(4096, n_output)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer4(x))
        x = self.output(x)
        return x


class HARmodel(nn.Module):
    """Model for human-activity-recognition."""

    def __init__(self, input_channel, num_classes):
        super().__init__()

        # Extract features, 1D conv layers
        self.features = nn.Sequential(
            nn.Conv1d(input_channel, 64, 5),
            nn.BatchNorm1d(64, momentum=0.5),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(64, 64, 5),
            nn.BatchNorm1d(64, momentum=0.5),
            nn.ReLU(),
            nn.Dropout(),
            nn.BatchNorm1d(64, momentum=0.5),
            nn.Conv1d(64, 64, 5),
            nn.ReLU(),
        )
        # Classify output, fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(50432, 128),
            nn.BatchNorm1d(128, momentum=0.5),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)

        return out

# class CAM_CNN(nn.Module):
#     """Model for human-activity-recognition."""
#
#     def __init__(self, input_channel, num_classes):
#         super().__init__()
#
#         # Extract features, 1D conv layers
#         self.features = nn.Sequential(
#             nn.Conv1d(input_channel, 64, 5),
#             nn.BatchNorm1d(64, momentum=0.5),
#             nn.ReLU(),
#             nn.Dropout(),
#             nn.Conv1d(64, 64, 5),
#             nn.BatchNorm1d(64, momentum=0.5),
#             nn.ReLU(),
#             nn.Dropout(),
#             nn.BatchNorm1d(64, momentum=0.5),
#             nn.Conv1d(64, 64, 5),
#             nn.ReLU(),
#         )
#         # Classify output, fully connected layers
#         self.classifier = nn.Sequential(
#             nn.Linear(64, num_classes)
#         )
#
#     def forward(self, x):
#         P = nn.AdaptiveAvgPool1d(1)
#         L = nn.Linear(64, 2)
#         x = self.features(x)
#         x = P(x)
#         x = x.squeeze()
#         x = self.classifier(x)
#
#         return x

