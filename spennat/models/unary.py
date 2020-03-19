
import torch
import torch.nn.functional as F


class UnaryModel(torch.nn.Module):
    def __init__(self, feature_network, *, num_nodes, num_vals):
        super().__init__()
        self.feature_network = feature_network
        self.num_nodes = num_nodes
        self.num_vals = num_vals

    def loss(self, xs, ys):
        pred = self.feature_network(xs)
        return F.cross_entropy(pred.view(-1, self.num_vals), ys.view(-1))

    def predict_beliefs(self, xs):
        pred = self.feature_network(xs)
        return F.softmax(pred, dim=2)

    def predict(self, xs):
        pred = self.feature_network(xs)
        return pred.argmax(dim=2)
