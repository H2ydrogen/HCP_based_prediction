from DTI import models
import torch
<<<<<<< HEAD
import time
import numpy as np

drop = torch.rand(100)
idx = torch.rand(100)

=======

model = models.CNN_GP(4, 2)
input = torch.rand(1, 4, 800)
output = model(input)
a = 0
>>>>>>> refs/remotes/origin/master
