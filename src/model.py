import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # 🔥 Unfreeze last layer block (layer4)
    for param in model.layer4.parameters():
        param.requires_grad = True

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    model = model.to(device)  # 🔥 move to GPU

    return model


if __name__ == "__main__":
    model = get_model(num_classes=4)
    print(model)