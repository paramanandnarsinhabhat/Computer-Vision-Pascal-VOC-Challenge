
# PASCAL VOC Object Detection with Faster R-CNN

This repository contains an implementation of an object detection model using the PASCAL VOC dataset and Faster R-CNN with a ResNet-50 backbone in PyTorch.

## Introduction

Object detection is a computer vision technique that involves identifying and locating objects within an image or video. This project uses the PASCAL VOC dataset, a well-known dataset for object detection tasks. We implement a Faster R-CNN model, which is a state-of-the-art deep learning model for object detection.

Project uses data files from local paths .
Please make changes in code accordingly.
Please download the dataset from here : https://import.cdn.thinkific.com/118220/dataset_pascalVOCDetection-200625-193221.zip

## Installation

Before running the code, ensure that you have the following dependencies installed:

```plaintext
torch==1.9.1
torchvision==0.10.1
pandas==1.3.3
matplotlib==3.4.3
Pillow==8.4.0
```

You can install these dependencies using `pip`:

```sh
pip install -r requirements.txt
```

## Dataset

The dataset used is PASCAL VOC Detection, which should be placed in the following directory structure:

```
dataset_pascalVOCDetection-200625-193221/
    train.csv
    val.csv
    train_images/
        image_1.jpg
        image_2.jpg
        ...
    val_images/
        image_1.jpg
        image_2.jpg
        ...
```

Please update the paths in the code to match where you've stored the dataset.

## Usage

The project includes several steps, which are outlined in the provided Jupyter Notebook (`Untitled.ipynb`):

1. **Explore and Understand the Dataset**: Load the dataset and visualize the images with bounding boxes.

2. **Prepare the Dataset for Training**: Define a custom PyTorch Dataset to load images and their corresponding annotations.

3. **Model Architecture**: Utilize a pre-trained Faster R-CNN model and adapt it for the PASCAL VOC dataset.

4. **Training**: Train the model using the prepared dataset.

5. **Evaluation**: Evaluate the model on a validation dataset and calculate the performance metrics.

To run the code, you can open the Jupyter Notebook in the `notebook` directory and execute the cells sequentially.

## Contributing

Feel free to fork this repository and submit pull requests to contribute to this project. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

