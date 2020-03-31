
import torch
import torch.nn.functional as F

from spennat.utils import random_probabilities, entropy
from spennat.optim import (
    EntropicMirrorAscentOptimizer,
    GradientAscentOptimizer
)


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
        def get_optimizer():
            optim_kwargs = dict(
                lr=cfg.inference.learning_rate,
                use_sqrt_decay=cfg.inference.use_sqrt_decay,
                track_higher_grads=True
            )
            if cfg.inference.optim_type == 'simple':
                return GradientAscentOptimizer(**optim_kwargs)
            elif cfg.inference.optim_type == 'entropic':
                return EntropicMirrorAscentOptimizer(**optim_kwargs)
            else:
                raise RuntimeError("Unknown optim_type")
        self.get_optimizer = get_optimizer

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
        ys = torch.log(random_probabilities(
            batch_size, self.num_nodes, self.num_vals,
            device=potentials.device, requires_grad=True))
        prev_ys = ys
        prev_energy = prev_ys.new_full((batch_size,), -float('inf'))
        opt = self.get_optimizer()
        for _ in range(self.inference_iterations):
            energy = self.global_network(torch.softmax(ys, dim=2), potentials)
            if self.entropy_coef > 0.0:
                energy += self.entropy_coef * entropy(ys, dim=(1, 2))
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
        return self.forward(xs).softmax(dim=2)

    def predict(self, xs):
        return self.forward(xs).argmax(dim=2)
