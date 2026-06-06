import torch
from torch import nn
from torchvision import models


class AgeGenderCNN(nn.Module):
    def __init__(self, num_genders=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )
        self.age_head = nn.Linear(128, 1)
        self.gender_head = nn.Linear(128, num_genders)

    def forward(self, x):
        x = self.features(x)
        x = self.shared(x)
        age = self.age_head(x).squeeze(1)
        gender = self.gender_head(x)
        return age, gender


class AgeGenderResNet18(nn.Module):
    def __init__(self, num_genders=2, pretrained=True, freeze_backbone=False):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        self.shared = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
        )
        self.age_head = nn.Linear(256, 1)
        self.gender_head = nn.Linear(256, num_genders)

    def forward(self, x):
        x = self.backbone(x)
        x = self.shared(x)
        age = self.age_head(x).squeeze(1)
        gender = self.gender_head(x)
        return age, gender


def build_model(model_name="cnn", pretrained=True, freeze_backbone=False):
    if model_name == "cnn":
        return AgeGenderCNN()
    if model_name == "resnet18":
        return AgeGenderResNet18(pretrained=pretrained, freeze_backbone=freeze_backbone)
    raise ValueError(f"Unknown model: {model_name}")
