
import torch
import numpy as np
from spen.utils import random_probabilities


EPS = 1e-6


class SPENModel(torch.nn.Module):
    """Structured Prediction Energy Network described in 
        - David Belanger and Andrew McCallum "Structured Prediction Energy Networks." ICML 2016. 
          https://arxiv.org/abs/1511.06350.
    """
    def __init__(self, feature_network, global_network, cfg, *, num_nodes, num_vals):
        """
        Arguments:
            feature_network {torch.nn.Module} -- feature network mapping (batch size, input dim) -> (batch size, num_nodes, num_vals)
            global_network {torch.nn.Module} -- energy network mapping (batch size, output dim) -> (batch size,)
            cfg {dict} --  config dictionary
            num_nodes {int} -- 
            num_vals {int} -- 
        """
        super().__init__()
        self.feature_network = feature_network
        self.global_network = global_network
        self.num_nodes = num_nodes
        self.num_vals = num_vals
        self.inference_iterations = cfg.inference.iterations
        self.inference_learning_rate = cfg.inference.learning_rate
        self.inference_eps = cfg.inference.eps
        self.inference_region_eps = cfg.inference.region_eps
        self.use_sqrt_decay = cfg.inference.use_sqrt_decay
        self.entropy_coef = cfg.entropy_coef

    def _lr(self, iteration):
        if self.use_sqrt_decay:
            return self.inference_learning_rate / np.sqrt(iteration)
        return self.inference_learning_rate / iteration

    def _gradient_based_inference(self, potentials):
        batch_size = potentials.size(0)
        potentials = potentials.detach()
        pred = random_probabilities(
            batch_size, self.num_nodes, self.num_vals, potentials.device)
        prev = pred
        prev_energy = prev.new_full((batch_size,), -float('inf'))
        for iteration in range(1, self.inference_iterations):
            self.global_network.zero_grad()
            pred = pred.detach().requires_grad_()
            energy = self.global_network(pred, potentials) \
                   - self.entropy_coef * (pred * torch.log(pred + EPS)).sum(dim=(1, 2))
            if (
                self.inference_eps is not None
                and torch.all((energy - prev_energy).abs() < self.inference_eps)
            ): break
            prev_energy = energy

            energy.sum().backward()
            lr_grad = self._lr(iteration) * pred.grad
            max_grad, _ = lr_grad.max(dim=-1, keepdim=True)
            pred = pred * torch.exp(lr_grad - max_grad)
            pred = pred / (pred.sum(dim=-1, keepdim=True) + EPS)
            if (
                self.inference_region_eps is not None
                and torch.all((prev - pred).norm(dim=2) < self.inference_region_eps)
            ): break
            prev = pred
        return pred

    def loss(self, xs, ys):
        potentials = self.feature_network(xs)
        preds = self._gradient_based_inference(potentials)
        self.zero_grad()
        pred_energy = self.global_network(preds, potentials)
        true_energy = self.global_network(ys, potentials)
        loss = pred_energy - true_energy \
             - self.entropy_coef * (preds * torch.log(preds + EPS)).sum(dim=(1, 2))
        return loss.mean()

    def predict_beliefs(self, xs):
        potentials = self.feature_network(xs)
        preds = self._gradient_based_inference(potentials)
        return preds

    def predict(self, xs):
        potentials = self.feature_network(xs)
        preds = self._gradient_based_inference(potentials)
        return preds.argmax(dim=2)
