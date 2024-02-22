import torchvision.models as models
import torchvision.transforms as transforms

# Load a pretrained model, e.g., Faster R-CNN
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)




