import sys 
sys.path.append('..')
import torch
import torch.nn as nn
from model.lib import SynchronizedBatchNorm2d
BatchNorm2d = SynchronizedBatchNorm2d

