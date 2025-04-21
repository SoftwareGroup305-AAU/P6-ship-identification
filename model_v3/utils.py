import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import math
from dataset import YOLODataset
from core import YOLO
from utils import build_dfl_targets
import torch.nn.functional as F

def bbox2dist(anchor_points, bbox, reg_max):
    x1y1, x2y2, = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)
