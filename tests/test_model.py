import torch
from src.model.resnet_fer import ResNetFER


def test_output_shape():
    model = ResNetFER(num_classes=7, pretrained=False)
    model.eval()
    x = torch.randn(4, 1, 48, 48)
    out = model(x)
    assert out.shape == (4, 7), f"Expected (4,7), got {out.shape}"


def test_no_softmax():
    model = ResNetFER(num_classes=7, pretrained=False)
    model.eval()
    x = torch.randn(1, 1, 48, 48)
    out = model(x)
    assert not torch.allclose(out.sum(), torch.tensor(1.0)), "Output should be logits, not probabilities"