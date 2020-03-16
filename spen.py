import arff
import hydra
import numpy as np
import logging
import time

from pathlib import Path
from omegaconf import DictConfig

import torch
from torch.utils.data import DataLoader, Dataset
from torch.autograd import variable
from torch.distributions.bernoulli import Bernoulli
import torch.nn as nn
import torch.nn.functional as F

from experiment_utils import TensorBoard

logger = logging.getLogger(__file__)

EPS = 1e-6
INPUTS = 1836
LABELS = 159


def onehot(x):
    x0 = x.view(-1, 1)
    x1 = x.new_zeros(len(x0), 2, dtype=torch.float)
    x1.scatter_(1, x0, 1)
    return x1.view(x.size() + (2,))


def get_learing_conf(model, cfg):
    return dict(
        params=[p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        momentum=cfg.momentum
    )

class FeatureEnergyNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_dim, out_dim = INPUTS, 150
        layers = []
        for _ in range(cfg.num_layers - 1):
            if cfg.dropout is not None:
                layers.append(nn.Dropout(cfg.dropout))
            linear = nn.Linear(in_dim, out_dim)
            linear.weight.data.normal_(std=np.sqrt(2. / in_dim))
            linear.bias.data.fill_(0.)
            layers.extend([linear, nn.ReLU(inplace=True)])
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, LABELS))
        layers[-1].weight.data.normal_(std=np.sqrt(2. / in_dim))
        layers[-1].bias.data.fill_(0.)
        self.model = nn.Sequential(*layers)
        self.learning_conf = get_learing_conf(self, cfg)

    def forward(self, xs):
        """
        Arguments:
            xs {torch.Tensor} -- (batch size, feature size (i.e., INPUTS))
        
        Returns:
            [torch.Tensor] -- (batch size, label size (i.e., LABELS), 2),
        """
        batch_size = xs.size(0)
        result = xs.new_zeros(batch_size, LABELS, 2)
        result[:, :, 1].copy_(self.model(xs))
        return result


class GlobalEnergyNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(LABELS, cfg.hidden_size, bias=False),
            nn.Softplus(),
            nn.Linear(cfg.hidden_size, 1, bias=False))
        self.learning_conf = get_learing_conf(self, cfg)

    def forward(self, ys, potentials):
        global_energy = (ys * potentials).sum(dim=(1, 2))
        label_energy = self.model(ys[:, :, 1]).squeeze(0)
        return global_energy + label_energy
        

class UnaryModel(nn.Module):
    def __init__(self, num_nodes, num_vals, cfg):
        super().__init__()
        self.feature_network = FeatureEnergyNetwork(cfg.feature_network)
        self.num_nodes = num_nodes
        self.num_vals = num_vals

    def get_optimizer(self):
        return torch.optim.SGD(**self.feature_network.learning_conf)

    def loss(self, xs, ys):
        pred = self.feature_network(xs)
        return F.cross_entropy(pred.view(-1, self.num_vals), ys.view(-1))

    def predict_beliefs(self, xs):
        pred = self.feature_network(xs)
        return F.softmax(pred, dim=2)

    def predict(self, xs):
        pred = self.feature_network(xs)
        return pred.argmax(dim=2)


class SPENModel(nn.Module):
    def __init__(self, num_nodes, num_vals, cfg):
        super().__init__()
        self.feature_network = FeatureEnergyNetwork(cfg.feature_network)
        self.global_network = GlobalEnergyNetwork(cfg.global_network)
        self.num_nodes = num_nodes
        self.num_vals = num_vals
        self.inference_iterations = cfg.inference.iterations
        self.inference_learning_rate = cfg.inference.learning_rate
        self.inference_eps = cfg.inference.eps
        self.inference_region_eps = cfg.inference.region_eps
        self.use_sqrt_decay = cfg.inference.use_sqrt_decay
        self.entropy_coef = cfg.entropy_coef

    def get_optimizer(self):
        return torch.optim.SGD([
            self.feature_network.learning_conf,
            self.global_network.learning_conf
        ])

    def _random_probabilities(self, batch_size, device=None):
        """returns a tensor with shape (batch size, self.num_nodes, self.num_vals),
        that sums to one at the last dimension
        
        Arguments:
            batch_size {int} -- batch size
        
        Returns:
            torch.Tensor -- (batch size, self.num_nodes, self.num_vals)
        """
        x = torch.rand(self.num_nodes, self.num_vals + 1, device=device)
        x[:, 0] = 0.
        x[:, -1] = 1.
        x, _ = x.sort(1)
        return (x[:, 1:] - x[:, :-1]).unsqueeze(0) \
                                     .expand(batch_size, -1, -1) \
                                     .contiguous()

    def _lr(self, iteration):
        if self.use_sqrt_decay:
            if self.use_sqrt_decay:
                lr = self.inference_learning_rate / np.sqrt(iteration)
            else:
                lr = self.inference_learning_rate / iteration
            return self.inference_learning_rate / np.sqrt(iteration)
        else:
            return self.inference_learning_rate / iteration

    def _gradient_based_inference(self, potentials):
        batch_size = potentials.size(0)
        potentials = potentials.detach()
        pred = self._random_probabilities(batch_size, potentials.device)
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


def load_bibtex(path, device):
    xs, ys = [], []
    with open(path) as f:
        for sample in arff.load(f)['data']:
            sample = list(map(int, sample))
            xs.append(sample[:-LABELS])
            ys.append(sample[-LABELS:])
    xs = torch.tensor(xs, dtype=torch.float, device=device)
    ys = torch.tensor(ys, dtype=torch.long, device=device)
    return xs, ys


class BibtexDataset(Dataset):
    def __init__(self, xs, ys, flip_prob=None):
        self.xs = xs
        self.ys = ys
        self.onehot_ys = onehot(ys)
        if flip_prob is not None:
            self.bernoulli = Bernoulli(probs=flip_prob)
        else:
            self.bernoulli = None

    def __len__(self):
        return self.xs.size(0)

    def __getitem__(self, index):
        xs = self.xs[index, :]
        if self.bernoulli:
            mask = self.bernoulli.sample((INPUTS,))
            mask = mask.to(xs.device)
            xs = xs * (1 - mask) + (1 - xs) * mask
        return xs, self.ys[index, :], self.onehot_ys[index, :]


def test(model, dataset, cfg, threshold=None):
    total_f1 = total_precision = total_recall = total_correct = total_vars = 0
    for xs, ys, _ in DataLoader(dataset, batch_size=cfg.batch_size):
        if threshold is None:
            preds = model.predict(xs)
        else:
            node_beliefs = model.predict_beliefs(xs)[:, :, 1]
            preds = (node_beliefs > threshold).long()
        correct = (preds * ys).sum(1)
        total_correct += (preds == ys).float().sum()
        total_vars += ys.numel()
        prec = correct / (preds.sum(1).float() + EPS)
        rec = correct / (ys.sum(1).float() + EPS)
        total_precision += prec.sum()
        total_recall += rec.sum()
        total_f1 += ((2 * prec * rec) / (prec + rec + EPS)).sum()
    acc = (total_correct / total_vars).item()
    f1 = (total_f1 / len(dataset)).item()
    precision = (total_precision / len(dataset)).item()
    recall = (total_recall / len(dataset)).item()
    return acc, precision, recall, f1


def test_with_thresholds(model, dataset, cfg):
    num_vars = 0
    thresholds = np.arange(0.05, 0.80, 0.05)
    total_accs = np.zeros_like(thresholds)
    total_precs = np.zeros_like(thresholds)
    total_recs = np.zeros_like(thresholds)
    total_f1s = np.zeros_like(thresholds)
    for xs, ys, _ in DataLoader(dataset, batch_size=cfg.batch_size):
        num_vars += ys.numel()
        node_beliefs = model.predict_beliefs(xs)[:, :, 1]
        for i, threshold in enumerate(thresholds):
            preds = (node_beliefs > threshold).long()
            correct = (preds * ys).sum(1).float()
            prec = correct / (preds.sum(1).float() + EPS)
            rec = correct / (ys.sum(1).float() + EPS)
            total_accs[i] += (preds == ys).float().sum()
            total_recs[i] += rec.sum()
            total_precs[i] += prec.sum()
            total_f1s[i] += ((2 * prec * rec) / (prec + rec + EPS)).sum()
    accs = total_accs / num_vars
    precs = total_precs / len(dataset)
    recs = total_recs / len(dataset)
    f1s = total_f1s / len(dataset)
    best = f1s.argmax()
    return (accs[best], precs[best], recs[best], f1s[best]), thresholds[best]


def save_model(model, file_path):
    with open(file_path, 'wb') as f:
        torch.save(model.state_dict(), f)


def load_model(model, file_path):
    with open(file_path, 'rb') as f:
        model.load_state_dict(torch.load(f))
            

def train(model, train_data, val_data, cfg, train_logger, val_logger):
    best_f1, best_epoch, best_threshold = 0., -1, -1

    def validation(epoch):
        nonlocal best_f1, best_epoch, best_threshold
        model.eval()
        if cfg.tune_thresholds:
            test_fun = test_with_thresholds
        else:
            test_fun = lambda *args: (test(*args), None)

        (acc, prec, rec, f1), threshold = test_fun(model, train_data, cfg)
        logger.info(
            f'train results: {(acc, prec, rec, f1)}, (threshold: {threshold})')
        train_logger.plot_for_current_epoch('F1', f1)
        train_logger.plot_for_current_epoch('Accuracy', acc)
        if val_data is not None:
            (acc, prec, rec, f1), threshold = test_fun(model, val_data, cfg)
            logger.info(
                f'val results: {(acc, prec, rec, f1)}, (threshold: {threshold})')
            val_logger.plot_for_current_epoch('F1', f1)
            val_logger.plot_for_current_epoch('Accuracy', acc)

        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch
            if cfg.tune_thresholds:
                best_threshold = threshold
            logger.info('new best f1 found, saving model')
            save_model(model, 'best_model')

        save_model(model, f'model_checkpoint_{epoch}')
        subset = 'val' if val_data is not None else 'train'
        logger.info(
            f'best {subset} results (epoch {best_epoch}, threshold {best_threshold}): {best_f1}')

    train_data_loader = DataLoader(train_data,
                                   batch_size=cfg.batch_size,
                                   shuffle=True,
                                   drop_last=True)
    opt = model.get_optimizer()
    for epoch in range(1, cfg.num_epochs + 1):
        logger.info(f'epoch {epoch}')
        train_logger.update_epoch()
        val_logger.update_epoch()
        if epoch % cfg.val_interval == 0:
            validation(epoch)
        avg_loss, count = 0, 0
        for xs, ys, onehot_ys in train_data_loader:
            if not cfg.use_cross_ent:
                ys = onehot_ys
            model.train()
            model.zero_grad()
            loss = model.loss(xs, ys)
            logger.info(
                f'loss of batch {count + 1}/{len(train_data_loader)}: {loss.item()}')
            loss.backward()
            avg_loss += loss
            count += 1
            if cfg.clip_grad:
                nn.utils.clip_grad_value_(model.parameters(), cfg.clip_grad)
            elif cfg.clip_grad_norm:
                nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.clip_grad_norm)
            opt.step()
        train_logger.plot_obj_val((avg_loss / count).item())
    validation(cfg.num_epochs)


@hydra.main(config_path='config.yaml')
def main(cfg: DictConfig) -> None:
    logger.info(cfg.pretty())

    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)

    if cfg.device >= 0:
        device = torch.device('cuda', cfg.device)
    else:
        device = torch.device('cpu')

    train_xs, train_ys = load_bibtex(
        hydra.utils.to_absolute_path(cfg.train), device)
    index = int(len(train_xs) * cfg.train_ratio)
    train_data = BibtexDataset(train_xs[:index, :],
                               train_ys[:index, :],
                               flip_prob=cfg.flip_prob)
    val_data = BibtexDataset(train_xs[index:, :],
                             train_ys[index:, :])
    test_xs, test_ys = load_bibtex(
        hydra.utils.to_absolute_path(cfg.test), device)
    test_data = BibtexDataset(test_xs, test_ys)

    model = hydra.utils.instantiate(cfg.model, LABELS, 2, cfg)
    if cfg.pretrained_unary:
        unary_model = UnaryModel(LABELS, 2, cfg)
        load_model(unary_model, hydra.utils.to_absolute_path(cfg.pretrained_unary))
        model.feature_network = unary_model.feature_network
    model = model.to(device)

    with TensorBoard(f'{cfg.model.name}_train') as train_logger, \
            TensorBoard(f'{cfg.model.name}_val') as val_logger:
        train(model,
              train_data,
              val_data,
              cfg,
              train_logger,
              val_logger)

    logger.info('loading best model...')
    load_model(model, 'best_model')
    logger.info('finding best threshold on validation data...')
    (val_acc, val_prec ,val_rec ,val_f1), threshold = test_with_thresholds(model, val_data, cfg)
    logger.info('testing on training data...')
    train_acc, train_prec ,train_rec ,train_f1 = test(model, train_data, cfg, threshold=threshold)
    logger.info('testing on test data...')
    test_acc, test_prec ,test_rec ,test_f1 = test(model, test_data, cfg, threshold=threshold)

    logger.info(f'final results: (threshold: {threshold})')
    logger.info(f'train: {train_acc}, {train_prec}, {train_rec}, {train_f1}')
    logger.info(f'val:   {val_acc}, {val_prec}, {val_rec}, {val_f1}')
    logger.info(f'test:  {test_acc}, {test_prec}, {test_rec}, {test_f1}')
            

if __name__ == '__main__':
    main()
