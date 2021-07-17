from DTI import models
import torch

model = models.CNN_GP(4, 2)
input = torch.rand(1, 4, 800)
output = model(input)
a = 0