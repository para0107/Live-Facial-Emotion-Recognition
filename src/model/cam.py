import torch
import torch.nn.functional as F
import numpy as np
import cv2


class CAMExtractor:
    def __init__(self, model):
        self.model = model
        self.feature_maps = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.feature_maps = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.model.features.layer4.register_forward_hook(forward_hook)
        self.model.features.layer4.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_class=None):
        self.model.eval()

        input_tensor = input_tensor.unsqueeze(0) if input_tensor.dim() == 3 else input_tensor
        input_tensor.requires_grad_(True)

        logits = self.model(input_tensor)

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class] = 1.0
        logits.backward(gradient=one_hot)

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.feature_maps).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[-1], input_tensor.shape[-2]))

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam, target_class

    def overlay_cam(self, original_image, cam, alpha=0.4):
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        if original_image.ndim == 2:
            original_image = np.stack([original_image] * 3, axis=-1)

        original_image = (original_image * 255).astype(np.uint8) if original_image.max() <= 1.0 else original_image
        overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
        return overlay
