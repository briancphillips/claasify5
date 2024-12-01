"""Model factory functions."""

import torch
import torch.nn as nn
import torchvision.models as models
from .architectures import WideResNet


class GTSRBNet(nn.Module):
    """ResNet18-based architecture for GTSRB."""

    def __init__(self, num_classes=43):
        super(GTSRBNet, self).__init__()
        # Start with pretrained ResNet18
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Modify first conv layer for smaller input size
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()  # Remove maxpool

        # Remove final FC layer
        self.features = nn.Sequential(*list(model.children())[:-1])

        # Add custom classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def extract_features(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


class ImageNetteNet(nn.Module):
    """ResNet18-based architecture for ImageNette optimized for smaller images."""

    def __init__(self, num_classes=10):
        super(ImageNetteNet, self).__init__()
        # Use ResNet18 instead of ResNet50 for smaller memory footprint
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Modify first conv layer for smaller input
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)  # Smaller pooling
        
        # Remove final FC layer
        self.features = nn.Sequential(*list(model.children())[:-2])
        
        # Add custom classifier with reduced dimensions
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),  # Reduced from 2048
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # Reduced dropout
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def extract_features(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


def get_model(dataset_name: str, model_name: str = None) -> nn.Module:
    """Get model for dataset."""
    if model_name is None:
        # Use default model for dataset
        if dataset_name == "cifar100":
            return WideResNet(depth=28, widen_factor=10, num_classes=100)
        elif dataset_name == "gtsrb":
            return GTSRBNet(num_classes=43)
        elif dataset_name == "imagenette":
            return ImageNetteNet(num_classes=10)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    else:
        # Use specified model
        if model_name == "wrn-28-10":
            return WideResNet(depth=28, widen_factor=10, num_classes=100)
        elif model_name == "gtsrb-net":
            return GTSRBNet(num_classes=43)
        elif model_name == "imagenette-net":
            return ImageNetteNet(num_classes=10)
        elif model_name == "resnet50":
            # Use latest pretrained weights
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            model = models.resnet50(weights=weights)

            # Adjust final layer based on dataset
            if dataset_name == "cifar100":
                model.fc = nn.Linear(model.fc.in_features, 100)
            elif dataset_name == "gtsrb":
                model.fc = nn.Linear(model.fc.in_features, 43)
            elif dataset_name == "imagenette":
                model.fc = nn.Linear(model.fc.in_features, 10)

            # Add feature extraction method
            def extract_features(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                return x

            # Add method to model
            import types

            model.extract_features = types.MethodType(extract_features, model)

            return model
        else:
            raise ValueError(f"Unknown model: {model_name}")
