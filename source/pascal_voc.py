import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

#Step 1: Explore and Understand the Dataset
# Load annotations
train_annotations = pd.read_csv('/Users/paramanandbhat/Downloads/dataset_pascalVOCDetection-200625-193221/train.csv')

print(train_annotations.head())

# Function to visualize an image with its bounding boxes
def visualize_image(image_path, annotations):
    img = Image.open(image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    for index, row in annotations.iterrows():
        box = [row['xmin'], row['ymin'], row['xmax'] - row['xmin'], row['ymax'] - row['ymin']]
        rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()

#Step 2: Preprocess the Data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import torch

class VOCDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_id = self.annotations.iloc[index, 0]
        img = Image.open(os.path.join(self.img_dir, img_id)).convert("RGB")
        boxes = self.annotations[self.annotations['image_name'] == img_id][['xmin', 'ymin', 'xmax', 'ymax']].values
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)  # Assuming all objects are of a single class
        
        if self.transform:
            img = self.transform(img)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        
        return img, target

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

