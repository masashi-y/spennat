
import torch
import numpy as np


EPS = 1e-6


class EntropicMirrorAscentOptimizer(object):
    def __init__(
            self,
            lr: float,
            use_sqrt_decay: bool = True,
            track_higher_grads: bool = True,
    ) -> None:
        self._lr = lr
        self.use_sqrt_decay = use_sqrt_decay
        self._track_higher_grads = track_higher_grads

    @property
    def lr(self) -> float:
        iteration = 1
        while True:
            if self.use_sqrt_decay:
                yield self._lr / np.sqrt(iteration)
            else:
                yield self._lr / iteration
            iteration += 1

    def step(
            self,
            loss: torch.Tensor,
            ys: torch.Tensor) -> torch.Tensor:
        grad, = torch.autograd.grad(
            loss,
            ys,
            create_graph=self._track_higher_grads,
            allow_unused=True
        )
        lr_grad = next(self.lr) * grad
        max_grad, _ = lr_grad.max(dim=-1, keepdim=True)
        ys = ys * torch.exp(lr_grad - max_grad)
        ys = ys / (ys.sum(dim=-1, keepdim=True) + EPS)
        if self._track_higher_grads:
            return ys
        return ys.detach().requires_grad_()
