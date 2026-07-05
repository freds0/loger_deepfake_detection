"""Training entry point (PyTorch Lightning + Hydra) for OSDFD and LOGER.

Usage:
    python train.py                                   # OSDFD defaults
    python train.py --config-name loger_fsm_siglip2   # LOGER + FSM (SigLIP 2)
    python train.py --config-name loger_fsm_dinov2    # LOGER + FSM (DINOv2)
    python train.py --config-name loger_fsm_baseline  # LOGER, FSM disabled
    python train.py data.root=/data/ffpp_frames optimizer.lr=1e-4

The model family is selected by ``model.name`` (``osdfd`` or ``loger``). The
Forgery Style Mixture module is active only during training; validation uses
AUC as the primary metric. With ``resume.enabled=true`` training automatically
resumes from the newest ``checkpoints/last.ckpt`` of the same experiment.
"""

from __future__ import annotations

import glob
import os

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

from src.data.datamodule import ForgeryDataModule
from src.lightning.loger_module import LOGERLightningModule
from src.lightning.module import OSDFDLightningModule
from src.utils.lightning_setup import build_callbacks, build_loggers, build_trainer
from src.utils.seed import seed_everything


def module_class(cfg: DictConfig):
    """Select the LightningModule family from ``model.name``."""
    name = OmegaConf.select(cfg, "model.name", default="osdfd")
    return LOGERLightningModule if name == "loger" else OSDFDLightningModule


def find_resume_checkpoint(cfg: DictConfig) -> str | None:
    """Return the checkpoint to resume from, honouring the ``resume`` config.

    Priority: explicit ``ckpt_path`` > newest ``checkpoints/last.ckpt`` under
    ``resume.dir`` (the experiment root, set per experiment config) > the
    current run directory. ``resume.dir`` is scanned recursively so timestamped
    Hydra run directories of the same experiment are found; no directory
    outside it is ever considered (a stale checkpoint from a different
    experiment would not be architecture-compatible).
    """
    if cfg.ckpt_path is not None:
        return cfg.ckpt_path
    if not OmegaConf.select(cfg, "resume.enabled", default=False):
        return None

    search_dir = OmegaConf.select(cfg, "resume.dir", default=None)
    if search_dir is None:
        candidate = os.path.join(cfg.output_dir, "checkpoints", "last.ckpt")
        candidates = [candidate] if os.path.isfile(candidate) else []
    else:
        candidates = glob.glob(
            os.path.join(search_dir, "**", "checkpoints", "last.ckpt"), recursive=True
        )
    if not candidates:
        return None
    latest = max(candidates, key=os.path.getmtime)
    print(f"[resume] Resuming from {latest}")
    return latest


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Bake the concrete Hydra run directory into the config so checkpoints do
    # not store an unresolvable `${hydra:...}` interpolation.
    with open_dict(cfg):
        cfg.output_dir = HydraConfig.get().runtime.output_dir
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)

    datamodule = ForgeryDataModule(**OmegaConf.to_container(cfg.data, resolve=True))
    model = module_class(cfg)(cfg)

    loggers = build_loggers(cfg)
    callbacks = build_callbacks(cfg)
    trainer = build_trainer(cfg, loggers, callbacks)

    trainer.fit(model, datamodule=datamodule, ckpt_path=find_resume_checkpoint(cfg))
    trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
