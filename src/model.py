import torch.nn as nn
import torchvision.models as models

def get_model(num_classes=4):
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
