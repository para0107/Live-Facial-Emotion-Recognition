import os
from PIL import Image
from torch.utils.data import Dataset


class FER2013Dataset(Dataset):
    CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    def __init__(self, root, split='train', transform=None):
        self.root = os.path.join(root, split)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.CLASSES)}

        for class_name in self.CLASSES:
            class_dir = os.path.join(self.root, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((
                        os.path.join(class_dir, fname),
                        self.class_to_idx[class_name]
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_class_weights(self):
        import torch
        counts = [0] * len(self.CLASSES)
        for _, label in self.samples:
            counts[label] += 1
        total = sum(counts)
        weights = [total / (len(self.CLASSES) * c) if c > 0 else 0.0 for c in counts]
        return torch.tensor(weights, dtype=torch.float)
