"""Model builders for the three experiments: baseline, frozen backbone, fine-tuned backbone."""
import os

import torch
import torch.nn as nn
from torchvision import models

# torchvision's normal weight download (download.pytorch.org) is not reachable from this
# environment, so we ship a local copy of the same ImageNet-pretrained ResNet18 checkpoint
# (verified to match torchvision's resnet18 state_dict keys exactly -- see training/README.md).
PRETRAINED_PATH = os.path.join(os.path.dirname(__file__), "..", "pretrained", "resnet18-5c106cde.pth")


class SmallCNN(nn.Module):
    """A small CNN trained from scratch -- the baseline the assignment asks us to compare against."""

    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(mode, n_classes=10):
    """
    mode:
      - "baseline":  small CNN trained from scratch (no pretraining)
      - "frozen":    pretrained ResNet18 backbone frozen, only the new FC head trained
      - "finetuned": pretrained ResNet18, all layers trainable (this is the deployed model,
                     and clears the >=10M trainable parameter requirement on its own)
    Returns (model, trainable_param_count)
    """
    if mode == "baseline":
        model = SmallCNN(n_classes)
        trainable = sum(p.numel() for p in model.parameters())
        return model, trainable

    if mode in ("frozen", "finetuned"):
        model = models.resnet18(weights=None)
        state_dict = torch.load(PRETRAINED_PATH, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        assert not missing and not unexpected, f"weight mismatch: {missing} {unexpected}"
        if mode == "frozen":
            for p in model.parameters():
                p.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, n_classes)  # new layer is trainable by default
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return model, trainable

    raise ValueError(f"unknown mode: {mode}")
