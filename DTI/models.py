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


# net = CNN(100, 5)
# print(net)
# optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
# loss_func = torch.nn.MSELoss()

