import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os
import pandas as pd
from PIL import Image

# Define transforms for the dataset
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load the dataset
train_dataset = '/Users/paramanandbhat/Downloads/dataset_pascalVOCDetection-200625-193221/train_images'
