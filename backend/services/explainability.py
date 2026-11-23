import base64
import io
import math
from typing import Any, Callable, Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image


def generate_gradcam(
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    pixels: torch.Tensor,
    forward: Callable[[torch.Tensor], torch.Tensor],
    channels_last: bool = False,
) -> Tuple[torch.Tensor, np.ndarray]:
    """Return logits and class-specific Grad-CAM maps for one batch."""
    captured: Dict[str, torch.Tensor] = {}

    def capture_activation(_module: torch.nn.Module, _inputs: Tuple[Any, ...], output: torch.Tensor) -> None:
        captured["activation"] = output

    hook = target_layer.register_forward_hook(capture_activation)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = forward(pixels)
            activation = captured.get("activation")
            if activation is None or not activation.requires_grad:
                raise RuntimeError("The selected Grad-CAM layer did not produce differentiable activations.")
            predicted_classes = logits.argmax(dim=1)
            class_scores = logits.gather(1, predicted_classes.unsqueeze(1)).sum()
            gradients = torch.autograd.grad(class_scores, activation)[0]

        if channels_last and activation.ndim == 3:
            token_count = activation.shape[1]
            spatial_size = math.isqrt(token_count)
            if spatial_size * spatial_size != token_count:
                token_count -= 1
                spatial_size = math.isqrt(token_count)
                if spatial_size * spatial_size != token_count:
                    raise RuntimeError("Grad-CAM could not map transformer tokens to a spatial feature grid.")
                activation = activation[:, 1:, :]
                gradients = gradients[:, 1:, :]
            activation = activation.transpose(1, 2).reshape(
                activation.shape[0], activation.shape[2], spatial_size, spatial_size
            )
            gradients = gradients.transpose(1, 2).reshape(
                gradients.shape[0], gradients.shape[2], spatial_size, spatial_size
            )
        elif channels_last and activation.ndim == 4:
            activation = activation.permute(0, 3, 1, 2)
            gradients = gradients.permute(0, 3, 1, 2)
        if activation.ndim != 4 or gradients.ndim != 4:
            raise RuntimeError("Grad-CAM requires a spatial feature map with four dimensions.")

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        maps = functional.relu((weights * activation).sum(dim=1))
        maps = functional.interpolate(
            maps.unsqueeze(1),
            size=pixels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        maxima = maps.flatten(1).amax(dim=1).view(-1, 1, 1)
        maps = torch.where(maxima > 0, maps / maxima.clamp(min=1e-12), maps)
        return logits.detach(), maps.detach().cpu().numpy()
    finally:
        hook.remove()


def render_gradcam(
    image: Image.Image,
    activation: np.ndarray,
    include_original: bool = False,
) -> Dict[str, str]:
    """Render a model-derived heatmap and overlay as compact JPEG data URIs."""
    original = image.convert("RGB")
    original.thumbnail((512, 512), Image.Resampling.LANCZOS)
    pixels = np.asarray(original, dtype=np.uint8)
    resized = cv2.resize(activation, (original.width, original.height), interpolation=cv2.INTER_LINEAR)
    normalized = np.clip(resized, 0, 1)
    heatmap = cv2.applyColorMap(np.uint8(normalized * 255), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(np.clip(pixels.astype(np.float32) * 0.55 + heatmap * 0.45, 0, 255))

    result = {
        "heatmap": _jpeg_data_uri(Image.fromarray(heatmap)),
        "overlay": _jpeg_data_uri(Image.fromarray(overlay)),
        "region_summary": summarize_gradcam_region(activation),
    }
    if include_original:
        result["original"] = _jpeg_data_uri(original)
    return result


def summarize_gradcam_region(activation: np.ndarray) -> str:
    """Describe the location of the strongest Grad-CAM activation."""
    if not np.isfinite(activation).any() or float(np.nanmax(activation)) <= 0:
        return "The model produced no localized Grad-CAM activation for this prediction."

    finite = np.nan_to_num(activation, nan=0.0, posinf=0.0, neginf=0.0)
    threshold = np.quantile(finite, 0.9)
    rows, columns = np.where(finite >= threshold)
    center_x = float(np.mean(columns)) / max(finite.shape[1] - 1, 1)
    center_y = float(np.mean(rows)) / max(finite.shape[0] - 1, 1)
    horizontal = "left" if center_x < 1 / 3 else "right" if center_x >= 2 / 3 else "center"
    vertical = "top" if center_y < 1 / 3 else "bottom" if center_y >= 2 / 3 else "middle"
    return f"Strongest model activation is concentrated in the {vertical} {horizontal} region."


def _jpeg_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=82, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
