"""Standalone inference for single images or folders.

Loads a trained OSDFD or LOGER checkpoint (auto-detected from the saved
config) and predicts forgery probability / label for one image or every image
in a directory. The Forgery Style Mixture module is always disabled at
inference time (paper Sec. III-C).
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import torch
from omegaconf import OmegaConf
from PIL import Image

from ..data.face_detection import FaceCropper
from ..data.transforms import SIGLIP_MEAN, SIGLIP_STD, build_transform
from ..lightning import load_module_from_checkpoint

_IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")


@dataclass
class Prediction:
    """One image prediction."""

    path: str
    label: str          # "fake" or "real"
    probability: float  # P(fake)
    confidence: float   # max(P(fake), 1 - P(fake))
    logit: float


class OSDFDPredictor:
    """Thin inference wrapper around a trained LightningModule.

    Args:
        ckpt_path: Path to a Lightning ``.ckpt`` checkpoint.
        device: Torch device string (default: cuda if available).
        image_size: Model input resolution.
        use_face_crop: Enable face detection before preprocessing.
        face_backend: Face-detection backend.
        face_margin: Crop enlargement factor.
    """

    def __init__(
        self,
        ckpt_path: str,
        device: str | None = None,
        image_size: int = 224,
        use_face_crop: bool = False,
        face_backend: str = "opencv",
        face_margin: float = 1.3,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # Loads OSDFD or LOGER depending on the checkpoint's saved config.
        self.module = load_module_from_checkpoint(ckpt_path, map_location=self.device)
        self.module.eval()
        self.module.to(self.device)
        # Use the normalisation stats the model was trained with (falls back to
        # the SigLIP [-1, 1] stats for older checkpoints).
        mean = OmegaConf.select(self.module.cfg, "data.normalization_mean") or SIGLIP_MEAN
        std = OmegaConf.select(self.module.cfg, "data.normalization_std") or SIGLIP_STD
        self.transform = build_transform(image_size, train=False, mean=tuple(mean), std=tuple(std))
        self.cropper = FaceCropper(face_backend, face_margin) if use_face_crop else None

    @torch.no_grad()
    def predict_image(self, path: str) -> Prediction:
        image = Image.open(path).convert("RGB")
        if self.cropper is not None:
            image = self.cropper(image)
        x = self.transform(image).unsqueeze(0).to(self.device)
        # LOGER modules expose multi-resolution eval logits; OSDFD does not.
        if hasattr(self.module, "_eval_logits"):
            logits = self.module._eval_logits(x)
        else:
            logits = self.module.model(x, apply_fsm=False).logits
        logit = float(logits.item())
        prob = float(torch.sigmoid(logits).item())
        return Prediction(
            path=path,
            label="fake" if prob >= 0.5 else "real",
            probability=prob,
            confidence=max(prob, 1.0 - prob),
            logit=logit,
        )

    def predict_folder(self, folder: str) -> list[Prediction]:
        files: list[str] = []
        for ext in _IMG_EXTS:
            files.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))
        return [self.predict_image(p) for p in sorted(files)]
