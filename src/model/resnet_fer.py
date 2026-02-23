import torch
import torch.nn as nn
import torchvision.models as models


class ResNetFER(nn.Module):
    def __init__(self, num_classes=7, dropout=0.5, pretrained=True):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Adapt conv1 from 3-channel RGB to 1-channel grayscale.
        # Average the pretrained weights across the 3 input channels.
        old_conv = backbone.conv1
        new_conv = nn.Conv2d(
            1, 64,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
        with torch.no_grad():
            new_conv.weight = nn.Parameter(old_conv.weight.mean(dim=1, keepdim=True))
        backbone.conv1 = new_conv

        # Do not change this without retraining — the architecture must match
        # exactly what was used when best_model.pth was saved.
        backbone.maxpool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)

        # Remove the final FC layer — replaced with my own head below.
        in_features = backbone.fc.in_features  # 512 for ResNet-18
        backbone.fc = nn.Identity()

        self.features = backbone

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def freeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.features.layer4.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = True

    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)