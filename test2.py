import logging
import time
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch
import torch.nn.functional as functional
import numpy as np
import matplotlib.pyplot as plt
from DTI import models, dataset, utils, analyse, cli
import os
import argparse

x = np.zeros((38, 100))
print(x[0:2, :].shape)

