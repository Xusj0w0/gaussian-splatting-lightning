import math
import os
import sys

from lightning.pytorch.callbacks import Callback

from internal.gaussian_splatting import GaussianSplatting

__all__ = ["RecordGradient"]


class RecordGradient(Callback):
    @staticmethod
    def log_gradient(outputs, batch, gaussian_model, global_step, pl_module: GaussianSplatting):
        for name, param in gaussian_model.named_parameters():
            if getattr(param, "grad", None) is not None:
                pl_module.logger.experiment.add_histogram(
                    f"gradients/{name}",
                    param.grad,
                    global_step=global_step,
                )

    def on_fit_start(self, trainer, pl_module: GaussianSplatting):
        pl_module.on_after_backward_hooks.append(self.log_gradient)


class ZeroLearningRate(Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if pl_module.trainer.global_step >= 50000:
            for optimizer in pl_module.optimizers():
                for param_group in optimizer.param_groups:
                    param_group["lr"] = 0.0
