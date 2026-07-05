"""Standalone LOGER + FSM inference wrapper.

FSM is always disabled at inference time.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import torch
from omegaconf import OmegaConf
from PIL import Image

from ..data.face_detection import FaceCropper
from ..data.transforms import build_transform
from ..lightning.loger_module import LOGERLightningModule
from ..models.backbones import normalization_stats

_IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")


@dataclass
class Prediction:
    path: str
    label: str
    probability: float
    confidence: float
    logit: float


class LOGERPredictor:
    """Inference wrapper around a trained LOGER Lightning checkpoint."""

    def __init__(
        self,
        ckpt_path: str,
        device: str | None = None,
        image_size: int | None = None,
        use_face_crop: bool = False,
        face_backend: str = "opencv",
        face_margin: float = 1.3,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.module = LOGERLightningModule.load_from_checkpoint(ckpt_path, map_location=self.device)
        self.module.eval()
        self.module.to(self.device)

        cfg = self.module.cfg
        backbone_name = OmegaConf.select(cfg, "backbone.name") or "siglip2-base"
        mean, std = normalization_stats(backbone_name)
        size = image_size or int(OmegaConf.select(cfg, "data.image_size") or 224)
        self.transform = build_transform(size, train=False, mean=mean, std=std)
        self.cropper = FaceCropper(face_backend, face_margin) if use_face_crop else None

    @torch.no_grad()
    def predict_image(self, path: str) -> Prediction:
        image = Image.open(path).convert("RGB")
        if self.cropper is not None:
            image = self.cropper(image)
        x = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.module._eval_logits(x)
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
        return [self.predict_image(path) for path in sorted(files)]
