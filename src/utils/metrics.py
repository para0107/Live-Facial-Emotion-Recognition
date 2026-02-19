import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(logits, labels):
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        correct = preds.eq(labels).sum().item()
        return correct / labels.size(0)


def evaluate_model(model, loader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    overall_acc = (all_preds == all_labels).mean()

    return {
        'accuracy': overall_acc,
        'confusion_matrix': cm,
        'report': report,
        'predictions': all_preds,
        'labels': all_labels,
    }
