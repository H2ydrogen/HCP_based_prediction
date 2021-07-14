from PIL import Image
import torch.nn as nn
import torch

m = nn.Conv1d(16, 33, 5)
x = torch.randn(21, 16, 51)
x = x.unsqueeze(2)
# y = m(x)
print(x.shape)
