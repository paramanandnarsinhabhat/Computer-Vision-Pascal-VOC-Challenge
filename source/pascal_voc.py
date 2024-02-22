import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

#Step 1: Explore and Understand the Dataset
# Load annotations
train_annotations = pd.read_csv('/Users/paramanandbhat/Downloads/dataset_pascalVOCDetection-200625-193221/train.csv')

print(train_annotations.head())



# Print all column names
print(train_annotations.columns)

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Load annotations
train_annotations = pd.read_csv('/Users/paramanandbhat/Downloads/dataset_pascalVOCDetection-200625-193221/train.csv')

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

# Example usage
sample_image_path = '/Users/paramanandbhat/Downloads/dataset_pascalVOCDetection-200625-193221/train_images/2007_000027.jpg'  # Update this path
sample_annotations = train_annotations[train_annotations['filename'] == '2007_000027.jpg']
visualize_image(sample_image_path, sample_annotations)


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
        boxes = self.annotations[self.annotations['filename'] == img_id][['xmin', 'ymin', 'xmax', 'ymax']].values
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

# Load dataset
train_dataset = VOCDataset(csv_file='/Users/paramanandbhat/Downloads/dataset_pascalVOCDetection-200625-193221/train.csv', img_dir='/Users/paramanandbhat/Downloads/dataset_pascalVOCDetection-200625-193221/train_images/', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

#Step 3 & 4: Choose a Model Architecture and Implementing the Model
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# Load a pretrained model and replace the classifier with a new one
import torchvision.models.detection as detection

# Load the model using the new `weights` parameter
model = detection.fasterrcnn_resnet50_fpn(weights=detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1)

num_classes = 21  # 20 classes + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

#Step 5 : Train the model
import torch
from torch.optim import SGD
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

# Assuming model is already defined and moved to the appropriate device
# Assuming train_loader is defined from your dataset

# Move model to the right device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Parameters and optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Number of training epochs
num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, targets in train_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Calculate loss
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backpropagation
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()
    
    print(f"Epoch #{epoch+1} loss: {running_loss/len(train_loader)}")

print("Training complete.")

val_dataset = VOCDataset(csv_file='/Users/paramanandbhat/Downloads/dataset_pascalVOCDetection-200625-193221/val.csv', img_dir='/Users/paramanandbhat/Downloads/dataset_pascalVOCDetection-200625-193221/val_images/', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

import numpy as np
import torch

def evaluate_model(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    detections = []  # Store all the detections here
    ground_truths = []  # Store all the ground truths here
    
    with torch.no_grad():  # No need to compute gradient when evaluating
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)
            
            # Convert outputs to CPU and numpy for metric calculation (optional)
            for output in outputs:
                detection = {
                    'boxes': output['boxes'].to('cpu').numpy(),
                    'labels': output['labels'].to('cpu').numpy(),
                    'scores': output['scores'].to('cpu').numpy(),
                }
                detections.append(detection)
            
            # Similarly convert targets to CPU and numpy (optional)
            for target in targets:
                gt = {
                    'boxes': target['boxes'].to('cpu').numpy(),
                    'labels': target['labels'].to('cpu').numpy(),
                }
                ground_truths.append(gt)
    
    # Here you would calculate your evaluation metrics based on detections and ground_truths
    # This might involve comparing the IoU of detections vs ground truths, etc.
    # For simplicity, this part is not implemented here.
    # You can integrate COCO evaluation or other metric calculations as needed.
    
    return detections, ground_truths  # Optionally return raw results for further analysis

# Evaluate the model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
detections, ground_truths = evaluate_model(model.to(device), val_loader, device)


