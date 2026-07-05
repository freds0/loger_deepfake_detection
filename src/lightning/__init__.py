"""OSDFD / LOGER lightning subpackage."""

from __future__ import annotations

import torch


def load_module_from_checkpoint(ckpt_path: str, map_location=None):
    """Load the right LightningModule (OSDFD or LOGER) from a checkpoint.

    The saved Hydra config (``hyper_parameters``) carries ``model.name``, so
    the module family is recovered from the checkpoint itself rather than from
    the caller's config.
    """
    from .loger_module import LOGERLightningModule
    from .module import OSDFDLightningModule

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters", {})
    name = hparams.get("model", {}).get("name", "osdfd")
    cls = LOGERLightningModule if name == "loger" else OSDFDLightningModule
    return cls.load_from_checkpoint(ckpt_path, map_location=map_location)
