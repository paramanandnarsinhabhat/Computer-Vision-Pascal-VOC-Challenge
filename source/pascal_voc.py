import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

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

