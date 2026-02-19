import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader
from datetime import datetime

# Resolve project root regardless of where the script is launched from
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data.dataset import FER2013Dataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.model.resnet_fer import ResNetFER
from src.training.trainer import Trainer
from src.training.losses import LabelSmoothingLoss


def load_config():
    config_path = os.path.join(PROJECT_ROOT, 'configs', 'config.yaml')
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    # Resolve dataset root relative to project root
    data_root = os.path.join(PROJECT_ROOT, config['data']['root'])
    config['data']['root'] = data_root

    # Resolve checkpoint and log paths
    config['paths']['checkpoints'] = os.path.join(PROJECT_ROOT, config['paths']['checkpoints'])
    config['paths']['logs'] = os.path.join(PROJECT_ROOT, config['paths']['logs'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

    # Windows requires num_workers=0 to avoid multiprocessing issues
    num_workers = 0 if os.name == 'nt' else config['data']['num_workers']

    image_size = config['data']['image_size']
    train_dataset = FER2013Dataset(
        root=data_root,
        split='train',
        transform=get_train_transforms(image_size)
    )
    val_dataset = FER2013Dataset(
        root=data_root,
        split='test',
        transform=get_val_transforms(image_size)
    )

    if len(train_dataset) == 0:
        print('\n[ERROR] Training dataset is empty!')
        print(f'  Looked in: {os.path.join(data_root, "train")}')
        print('  Make sure the folder exists and contains subfolders: angry, disgust, fear, happy, neutral, sad, surprise')
        sys.exit(1)

    print(f'Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}')

    class_weights = train_dataset.get_class_weights().to(device)
    print('Class weights:', [f'{w:.2f}' for w in class_weights])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
    )

    model = ResNetFER(
        num_classes=config['model']['num_classes'],
        dropout=config['model']['dropout'],
        pretrained=config['model']['pretrained'],
    ).to(device)

    print(f'Trainable params: {model.get_trainable_params():,}')

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
        eta_min=1e-6,
    )

    criterion = LabelSmoothingLoss(
        num_classes=config['model']['num_classes'],
        smoothing=0.1,
        class_weights=class_weights,
    )

    run_dir = os.path.join(config['paths']['logs'], datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(run_dir, exist_ok=True)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        config=config,
        run_dir=run_dir,
    )

    trainer.fit(train_loader, val_loader)


if __name__ == '__main__':
    main()