"""LOGER + Forgery Style Mixture training entry point.

Usage:
    conda activate loger
    python train_loger.py --config-name loger_fsm_siglip2
    python train_loger.py --config-name loger_fsm_siglip2 data.root=/path/to/ffpp_frames
"""

from __future__ import annotations

import os

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

from src.data.datamodule import ForgeryDataModule
from src.lightning.loger_module import LOGERLightningModule
from src.models.backbones import normalization_stats
from src.utils.lightning_setup import build_callbacks, build_loggers, build_trainer
from src.utils.seed import seed_everything


def _prepare_cfg(cfg: DictConfig) -> None:
    with open_dict(cfg):
        cfg.output_dir = HydraConfig.get().runtime.output_dir
        mean, std = normalization_stats(cfg.backbone.name)
        cfg.data.normalization_mean = list(mean)
        cfg.data.normalization_std = list(std)
        if cfg.ckpt_path is None and cfg.resume.enabled:
            resume_dir = cfg.resume.dir or cfg.output_dir
            candidate = os.path.join(resume_dir, "checkpoints", "last.ckpt")
            if os.path.exists(candidate):
                cfg.ckpt_path = candidate


@hydra.main(version_base=None, config_path="configs", config_name="loger_fsm_siglip2")
def main(cfg: DictConfig) -> None:
    _prepare_cfg(cfg)
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)

    datamodule = ForgeryDataModule(**OmegaConf.to_container(cfg.data, resolve=True))
    model = LOGERLightningModule(cfg)

    loggers = build_loggers(cfg)
    callbacks = build_callbacks(cfg)
    trainer = build_trainer(cfg, loggers, callbacks)

    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
