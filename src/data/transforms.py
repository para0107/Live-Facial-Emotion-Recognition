import torchvision.transforms as T


def get_train_transforms(image_size=48):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.RandomErasing(p=0.3, scale=(0.02, 0.15)),
        T.Normalize(mean=[0.507], std=[0.255]),
    ])


def get_val_transforms(image_size=48):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.507], std=[0.255]),
    ])


def get_inference_transforms(image_size=48):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.507], std=[0.255]),
    ])
