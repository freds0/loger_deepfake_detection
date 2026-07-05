"""Image preprocessing and configurable forgery-oriented augmentation.

The default preprocessing path is intentionally simple: resize to the requested
square input resolution, convert to tensor and normalise. Training augmentation
is enabled only when ``augmentation.enabled`` is true. Each augmentation has its
own probability so ablations can switch perturbations on and off independently.
"""

from __future__ import annotations

import io
import random

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# SigLIP normalisation constants (range [-1, 1]). Other backbone stats can be
# passed by the LOGER entry points through ``normalization_mean/std``.
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)

DEFAULT_AUG: dict = {
    "enabled": False,
    "hflip": 0.5,
    "random_resize": 0.0,
    "resize_scale": (0.25, 0.75),
    "downscale": 0.0,              # backward-compatible alias
    "downscale_range": (0.25, 0.75),
    "jpeg": 0.0,
    "jpeg_quality": (40, 90),
    "gaussian_blur": 0.0,
    "blur_sigma": (0.1, 2.0),
    "motion_blur": 0.0,
    "motion_kernel": (3, 9),
    "gaussian_noise": 0.0,
    "noise_std": 0.03,
    "brightness": 0.0,
    "brightness_limit": 0.2,
    "contrast": 0.0,
    "contrast_limit": 0.2,
    "color_jitter": 0.0,
    "color_jitter_strength": 0.2,
    "hue": 0.05,
    "random_crop": 0.0,
    "crop_scale": (0.8, 1.0),
    "random_resized_crop": False,  # legacy always-on crop flag
    "rrc_scale": (0.8, 1.0),
    "random_erasing": 0.0,
}


class RandomJPEG:
    """Randomly recompress a PIL image as JPEG."""

    def __init__(self, p: float, quality: tuple[int, int]) -> None:
        self.p = p
        self.quality = quality

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        q = random.randint(int(self.quality[0]), int(self.quality[1]))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class RandomDownscale:
    """Randomly down-sample then up-sample a PIL image."""

    def __init__(self, p: float, scale: tuple[float, float]) -> None:
        self.p = p
        self.scale = scale

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        w, h = img.size
        s = random.uniform(float(self.scale[0]), float(self.scale[1]))
        small = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)


class RandomMotionBlur:
    """Apply a simple horizontal, vertical or diagonal motion-blur kernel.

    Convolves with OpenCV (``filter2D``) because PIL's ``ImageFilter.Kernel``
    only supports 3x3 and 5x5 kernels.
    """

    def __init__(self, p: float, kernel_size: tuple[int, int]) -> None:
        self.p = p
        self.kernel_size = kernel_size

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        import cv2

        lo, hi = int(self.kernel_size[0]), int(self.kernel_size[1])
        k = random.randrange(max(3, lo) | 1, max(3, hi) + 1, 2)
        kernel = np.zeros((k, k), dtype=np.float32)
        direction = random.choice(("horizontal", "vertical", "diag", "anti_diag"))
        if direction == "horizontal":
            kernel[k // 2, :] = 1.0
        elif direction == "vertical":
            kernel[:, k // 2] = 1.0
        elif direction == "diag":
            np.fill_diagonal(kernel, 1.0)
        else:
            np.fill_diagonal(np.fliplr(kernel), 1.0)
        kernel /= kernel.sum()
        blurred = cv2.filter2D(np.asarray(img.convert("RGB")), -1, kernel)
        return Image.fromarray(blurred)


class RandomGaussianNoise:
    """Add Gaussian noise to a tensor in [0, 1] before normalization."""

    def __init__(self, p: float, std: float) -> None:
        self.p = p
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return tensor
        return (tensor + torch.randn_like(tensor) * self.std).clamp(0.0, 1.0)


def _merge_aug(augmentation: dict | None) -> dict:
    cfg = dict(DEFAULT_AUG)
    if augmentation:
        cfg.update({k: v for k, v in augmentation.items() if v is not None})
    return cfg


def _tuple(value) -> tuple:
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value, value)


def build_transform(
    image_size: int = 224,
    train: bool = False,
    augmentation: dict | None = None,
    mean: tuple[float, float, float] = SIGLIP_MEAN,
    std: tuple[float, float, float] = SIGLIP_STD,
) -> T.Compose:
    """Build preprocessing and optional training augmentation.

    Args:
        image_size: Target square resolution.
        train: Whether this transform is for the training split.
        augmentation: Probability/strength config for training augmentation.
        mean: Channel-wise normalization mean.
        std: Channel-wise normalization std.
    """
    aug = _merge_aug(augmentation)
    use_aug = train and bool(aug["enabled"])

    ops: list = []
    if use_aug and aug["random_resized_crop"]:
        ops.append(
            T.RandomResizedCrop(
                image_size,
                scale=_tuple(aug["rrc_scale"]),
                interpolation=T.InterpolationMode.BICUBIC,
            )
        )
    else:
        ops.append(T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC))

    if use_aug:
        if aug["hflip"] > 0:
            ops.append(T.RandomHorizontalFlip(p=float(aug["hflip"])))
        if aug["random_crop"] > 0:
            ops.append(
                T.RandomApply(
                    [
                        T.RandomResizedCrop(
                            image_size,
                            scale=_tuple(aug["crop_scale"]),
                            interpolation=T.InterpolationMode.BICUBIC,
                        )
                    ],
                    p=float(aug["random_crop"]),
                )
            )
        random_resize_p = aug["random_resize"] or aug["downscale"]
        resize_scale = aug["resize_scale"] if aug["random_resize"] else aug["downscale_range"]
        if random_resize_p > 0:
            ops.append(RandomDownscale(float(random_resize_p), _tuple(resize_scale)))
        if aug["brightness"] > 0:
            ops.append(
                T.RandomApply(
                    [T.ColorJitter(brightness=float(aug["brightness_limit"]))],
                    p=float(aug["brightness"]),
                )
            )
        if aug["contrast"] > 0:
            ops.append(
                T.RandomApply(
                    [T.ColorJitter(contrast=float(aug["contrast_limit"]))],
                    p=float(aug["contrast"]),
                )
            )
        if aug["color_jitter"] > 0:
            strength = float(aug["color_jitter_strength"])
            ops.append(
                T.RandomApply(
                    [
                        T.ColorJitter(
                            brightness=strength,
                            contrast=strength,
                            saturation=strength,
                            hue=float(aug["hue"]),
                        )
                    ],
                    p=float(aug["color_jitter"]),
                )
            )
        if aug["gaussian_blur"] > 0:
            ops.append(
                T.RandomApply(
                    [T.GaussianBlur(kernel_size=5, sigma=_tuple(aug["blur_sigma"]))],
                    p=float(aug["gaussian_blur"]),
                )
            )
        if aug["motion_blur"] > 0:
            ops.append(RandomMotionBlur(float(aug["motion_blur"]), _tuple(aug["motion_kernel"])))
        if aug["jpeg"] > 0:
            ops.append(RandomJPEG(float(aug["jpeg"]), _tuple(aug["jpeg_quality"])))

    ops.append(T.ToTensor())
    if use_aug and aug["gaussian_noise"] > 0:
        ops.append(RandomGaussianNoise(float(aug["gaussian_noise"]), float(aug["noise_std"])))
    ops.append(T.Normalize(mean=mean, std=std))

    if use_aug and aug["random_erasing"] > 0:
        ops.append(T.RandomErasing(p=float(aug["random_erasing"])))

    return T.Compose(ops)
