import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.class_weights = class_weights

    def forward(self, logits, targets):
        return F.cross_entropy(logits, targets, weight=self.class_weights)


class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes=7, smoothing=0.1, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.class_weights = class_weights

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / (self.num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = -(smooth_targets * log_probs).sum(dim=-1)

        if self.class_weights is not None:
            weights = self.class_weights[targets]
            loss = (loss * weights).mean()
        else:
            loss = loss.mean()

        return loss
