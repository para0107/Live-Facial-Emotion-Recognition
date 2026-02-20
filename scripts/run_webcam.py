import os
import sys
import yaml
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.inference.webcam import WebcamFER


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
    inf_cfg = config['inference']

    app = WebcamFER(
        checkpoint_path=args.checkpoint,
        smoothing_window=inf_cfg['smoothing_window'],
        scale_factor=inf_cfg['face_scale_factor'],
        min_neighbors=inf_cfg['face_min_neighbors'],
        min_size=tuple(inf_cfg['face_min_size']),
        confidence_threshold=inf_cfg['confidence_threshold'],
        uncertainty_threshold=inf_cfg.get('uncertainty_threshold', 0.40),
    )

    app.run()


if __name__ == '__main__':
    main()