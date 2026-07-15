"""Exponential Moving Average (EMA) of the underlying model's weights.

Maintains a shadow copy of ``pl_module.model.state_dict()`` updated after
every training batch, and transparently swaps it into ``pl_module.model`` for
the duration of validation/test (so ``val/auc`` -- the metric
``ModelCheckpoint`` monitors -- is measured on the EMA weights, not the raw
ones). Checkpoints keep the raw weights in ``state_dict``; the EMA shadow
lives in the callback's own state and is restored on resume. To run inference
with the EMA weights instead of the raw ones, load them explicitly with
:func:`load_ema_weights`.
"""

from __future__ import annotations

import lightning as L
import torch


class EMACallback(L.Callback):
    """Exponential moving average of ``pl_module.model`` parameters/buffers.

    ``shadow <- decay * shadow + (1 - decay) * param`` after every training
    batch. Swapped into ``pl_module.model`` for validation/test and restored
    immediately after, so training itself is unaffected and checkpoints keep
    optimising the raw weights.

    Args:
        decay: EMA decay rate (closer to 1 = slower-moving average).
    """

    def __init__(self, decay: float = 0.999) -> None:
        super().__init__()
        self.decay = decay
        self._shadow: dict[str, torch.Tensor] | None = None
        self._backup: dict[str, torch.Tensor] | None = None

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        # A resumed run already restored `_shadow` via load_state_dict (called
        # before on_fit_start); don't clobber it with the raw weights.
        if self._shadow is None:
            self._shadow = {k: v.detach().clone() for k, v in pl_module.model.state_dict().items()}

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        with torch.no_grad():
            for k, v in pl_module.model.state_dict().items():
                shadow = self._shadow[k]
                if torch.is_floating_point(shadow):
                    shadow.mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
                else:  # e.g. num_batches_tracked -- not meaningful to average
                    shadow.copy_(v)

    def _swap_in(self, pl_module: L.LightningModule) -> None:
        if self._shadow is None:
            return
        self._backup = {k: v.detach().clone() for k, v in pl_module.model.state_dict().items()}
        pl_module.model.load_state_dict(self._shadow, strict=True)

    def _swap_out(self, pl_module: L.LightningModule) -> None:
        if self._backup is None:
            return
        pl_module.model.load_state_dict(self._backup, strict=True)
        self._backup = None

    def on_validation_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._swap_in(pl_module)

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._swap_out(pl_module)

    def on_test_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._swap_in(pl_module)

    def on_test_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._swap_out(pl_module)

    # ---- checkpoint persistence (Lightning's callback state-dict protocol) --
    def state_dict(self) -> dict:
        return {"shadow": self._shadow}

    def load_state_dict(self, state_dict: dict) -> None:
        self._shadow = state_dict.get("shadow")


def load_ema_weights(ckpt_path: str, model: torch.nn.Module) -> bool:
    """Load ``EMACallback``'s shadow weights from a checkpoint into ``model``.

    Args:
        ckpt_path: Path to a Lightning ``.ckpt`` file.
        model: The underlying model (e.g. ``LOGERLightningModule.model``) --
            not the ``LightningModule`` wrapper.

    Returns:
        ``True`` if EMA weights were found and loaded, ``False`` if the
        checkpoint has no ``EMACallback`` state (e.g. EMA was disabled for
        that run); ``model`` is left untouched in that case.
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ema_state = checkpoint.get("callbacks", {}).get(EMACallback.__qualname__)
    shadow = ema_state.get("shadow") if ema_state else None
    if not shadow:
        return False
    model.load_state_dict(shadow, strict=True)
    return True
