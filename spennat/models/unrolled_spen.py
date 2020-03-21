
import torch
import torch.nn.functional as F
import numpy as np

from spennat.utils import random_probabilities


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
        # ys = ys + next(self.lr) * grad
        lr_grad = next(self.lr) * grad
        max_grad, _ = lr_grad.max(dim=-1, keepdim=True)
        ys = ys * torch.exp(lr_grad - max_grad)
        ys = ys / (ys.sum(dim=-1, keepdim=True) + EPS)
        if self._track_higher_grads:
            return ys
        return ys.detach().requires_grad_()


class UnrolledSPENModel(torch.nn.Module):
    """Structured Prediction Energy Network described in 
        - David Belanger,Bishan Yang and Andrew McCallum, End-to-End Learning for Structured Prediction Energy Networks 
          https://arxiv.org/abs/1703.05667
    """
    def __init__(self, feature_network, global_network, cfg, *, num_nodes, num_vals):
        super().__init__()
        self.feature_network = feature_network
        self.global_network = global_network
        self.num_nodes = num_nodes
        self.num_vals = num_vals
        self.inference_iterations = cfg.inference.iterations
        self.inference_eps = cfg.inference.eps
        self.inference_region_eps = cfg.inference.region_eps
        self.entropy_coef = cfg.entropy_coef
        self.optim_kwargs = dict(
            lr=cfg.inference.learning_rate,
            use_sqrt_decay=cfg.inference.use_sqrt_decay,
            track_higher_grads=True
        )

    def forward(self, xs):
        """
        Arguments:
            xs {torch.Tensor} -- (batch size, INPUTS)
        
        Returns:
            [torch.Tensor] -- (batch size, num_nodes, num_vals)
        """
        potentials = self.feature_network(xs)
        ys = self._unrolled_gradient_descent(potentials)
        return ys

    def _unrolled_gradient_descent(self, potentials):
        batch_size = potentials.size(0)
        potentials = potentials.detach()
        device = potentials.device
        ys = torch.log(random_probabilities(
            batch_size, self.num_nodes, self.num_vals,
            device=device, requires_grad=True))
        # hs = ys.new_zeros(1)  # batch size?
        prev_ys = ys
        prev_energy = prev_ys.new_full((batch_size,), -float('inf'))
        opt = EntropicMirrorAscentOptimizer(**self.optim_kwargs)
        for _ in range(self.inference_iterations):
            ys = torch.softmax(ys, dim=-1)
            energy = self.global_network(ys, potentials) \
                   - self.entropy_coef * (ys * torch.log(ys + EPS)).sum(dim=(1, 2))
            if (
                self.inference_eps is not None
                and torch.all((energy - prev_energy).abs() < self.inference_eps)
            ): break
            prev_energy = energy
            ys = opt.step(energy.sum(), ys)
            if (
                self.inference_region_eps is not None
                and torch.all((prev_ys - ys).norm(dim=2) < self.inference_region_eps)
            ): break
            prev_ys = ys
        return ys

    def loss(self, xs, ys):
        pred = self(xs)
        return F.cross_entropy(pred.view(-1, self.num_vals), ys.view(-1))

    def predict_beliefs(self, xs):
        return self.forward(xs)

    def predict(self, xs):
        return self.forward(xs).argmax(dim=2)
