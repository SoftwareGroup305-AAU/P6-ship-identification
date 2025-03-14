import unittest
import torch
import torch.nn as nn
import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from yolo.yolo import calculate_loss

class TestYOLO(unittest.TestCase):
    pass

if __name__ == '__main__':
    unittest.main()
