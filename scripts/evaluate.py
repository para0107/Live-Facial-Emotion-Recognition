import os
import sys
import yaml
import argparse
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data.dataset import FER2013Dataset
from src.data.transforms import get_val_transforms
from src.model.resnet_fer import ResNetFER
from src.utils.metrics import evaluate_model
from src.utils.visualization import plot_confusion_matrix


def load_config():
    config_path = os.path.join(PROJECT_ROOT, 'configs', 'config.yaml')
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default=os.path.join(PROJECT_ROOT, 'checkpoints', 'best_model.pth'))
    args = parser.parse_args()

    config = load_config()
    data_root = os.path.join(PROJECT_ROOT, config['data']['root'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = 0 if os.name == 'nt' else 4

    test_dataset = FER2013Dataset(
        root=data_root,
        split='test',
        transform=get_val_transforms(config['data']['image_size'])
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=num_workers)

    model = ResNetFER(
        num_classes=config['model']['num_classes'],
        dropout=0.0,
        pretrained=False,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded checkpoint from epoch {checkpoint["epoch"]} (val acc {checkpoint["val_acc"]:.4f})')

    results = evaluate_model(model, test_loader, device, FER2013Dataset.CLASSES)

    print(f'\nTest Accuracy: {results["accuracy"]:.4f}')
    print('\nClassification Report:')
    print(results['report'])

    logs_dir = os.path.join(PROJECT_ROOT, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    plot_confusion_matrix(
        results['confusion_matrix'],
        FER2013Dataset.CLASSES,
        save_path=os.path.join(logs_dir, 'confusion_matrix.png')
    )


if __name__ == '__main__':
    main()