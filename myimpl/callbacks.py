import math
import os
import sys

from lightning.pytorch.callbacks import Callback

from internal.gaussian_splatting import GaussianSplatting


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
