import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Load annotations
train_annotations = pd.read_csv('/Users/paramanandbhat/Downloads/dataset_pascalVOCDetection-200625-193221/train.csv')

print(train_annotations.head())
