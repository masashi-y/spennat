import arff
import hydra
import numpy as np
import logging
import time

from pathlib import Path
from omegaconf import DictConfig

import torch
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from torch.distributions.bernoulli import Bernoulli
import torch.nn as nn

from experiment_utils import TensorBoard

logger = logging.getLogger(__file__)

EPS = 1e-6
INPUTS = 1836
LABELS = 159


def onehot(x):
    x0 = x.view(-1, 1)
    x1 = torch.zeros(len(x0), 2)
    x1.scatter_(1, x0, 1)
    return x1.view(x.size(0), -1)


class FeatureEnergyNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_dim, out_dim = INPUTS, 150
        layers = []
        for _ in range(cfg.num_layers-1):
            if cfg.unary_dropout is not None:
                layers.append(nn.Dropout(cfg.unary_dropout))
            linear = nn.Linear(in_dim, out_dim)
            linear.weight.data.normal_(std=np.sqrt(2. / in_dim))
            linear.bias.data.fill_(0.)
            layers.extend([linear, nn.ReLU(inplace=True)])
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, LABELS))
        layers[-1].weight.data.normal_(std=np.sqrt(2. / in_dim))
        layers[-1].bias.data.fill_(0.)
        self.model = nn.Sequential(*layers)

    def forward(self, xs):
        """
        Arguments:
            xs {torch.Tensor} -- (batch size, feature size (i.e., INPUTS))
        
        Returns:
            [torch.Tensor] -- (batch size, label size (i.e., LABELS) * 2),
        """
        batch_size = xs.size(0)
        result = torch.zeros(batch_size * LABELS, 2)
        result[:, 1].copy_(self.model(xs).view(-1))
        return result.view(batch_size, -1)


class GlobalEnergyNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden_size = cfg.t_hidden_size
        self.model = nn.Sequential(
            nn.Linear(LABELS, hidden_size, bias=False),
            nn.Softplus(),
            nn.Linear(hidden_size, 1, bias=False))

    def forward(self, ys, pots):
        local_energy = (ys * pots).sum(dim=1)
        ys = ys.reshape(-1, 2)[:, 1].reshape(ys.size(0), LABELS)
        label_energy = self.model(ys).squeeze()
        return local_energy + label_energy
        

class UnaryModel(nn.Module):
    def __init__(self, num_nodes, num_vals, cfg):
        super().__init__()
        self.model = FeatureEnergyNetwork(cfg)
        self.num_nodes = num_nodes
        self.num_vals = num_vals

    def get_optimizer(self, cfg):
        opt = torch.optim.SGD(
            [p for p in self.parameters() if p.requires_grad],
            lr=cfg.unary_lr,
            weight_decay=cfg.get('unary_wd', 0.),
            momentum=cfg.get('unary_wd', 0.)
        )
        return opt

    def loss(self, epoch, xs, onehot_ys):
        pred = self.model(xs)
        loss = nn.CrossEntropyLoss()
        return loss(pred.view(-1, self.num_vals), onehot_ys.view(-1))

    def calculate_beliefs(self, xs):
        batch_size = xs.size(0)
        pred = self.model(xs)
        return nn.Softmax(dim=1)(pred.view(-1, self.num_vals)).view(batch_size, -1)

    def predict(self, xs):
        batch_size = xs.size(0)
        pred = self.model(xs)
        return pred.view(-1, self.num_vals).argmax(dim=1).view(batch_size, -1)


class SPENModel(nn.Module):
    def __init__(self, num_nodes, num_vals, cfg):
        super().__init__()
        self.feature_network = FeatureEnergyNetwork(cfg)
        self.global_network = GlobalEnergyNetwork(cfg)
        self.num_nodes = num_nodes
        self.num_vals = num_vals
        self.inference_iteration = cfg.num_inf_itrs
        self.inf_lr = cfg.get('inf_lr', 0.05)
        self.use_linear_decay = cfg.use_linear_decay
        self.inf_eps = cfg.get('inf_eps', None)  # 1e-4
        self.inf_region_eps = cfg.get('inf_region_eps', None)  # None
        self.entropy_coef = cfg.entropy_coef

    def get_optimizer(self, cfg):
        unary_conf = dict(
            params=[p for p in self.feature_network.parameters() if p.requires_grad],
            lr=cfg.unary_lr,
            weight_decay=cfg.get('unary_wd', 0.),
            momentum=cfg.get('unary_mom', 0.)
        )
        unary_t_conf = dict(
            params=[p for p in self.global_network.parameters() if p.requires_grad],
            lr=cfg.t_unary_lr,
            weight_decay=cfg.get('t_unary_wd', 0.),
            momentum=cfg.get('t_unary_mom', 0.)
        )
        opt = torch.optim.SGD([unary_conf, unary_t_conf])
        return opt

    def _random_probabilities(self, batch_size):
        """returns a tensor with shape (batch size, self.num_nodes * self.num_vals),
        that sums to one at the last dimension when reshaped to (batch size, self.num_nodes, self.num_vals)
        
        Arguments:
            batch_size {int} -- batch size
        
        Returns:
            Tensor -- torch Tensor
        """
        x = torch.rand(self.num_nodes, self.num_vals + 1)
        x[:, 0] = 0.
        x[:, -1] = 1.
        x, _ = x.sort(1)
        return (x[:, 1:] - x[:, :-1]).view(-1) \
                                     .expand(batch_size, -1) \
                                     .contiguous()

    def _emd_inf(self, pots):
        batch_size = pots.size(0)
        pots = pots.detach()
        pots.requires_grad = False
        pred = self._random_probabilities(batch_size)
        prev = pred
        region_prediction_eps = float('inf')
        for iteration in range(1, self.inference_iteration):
            self.global_network.zero_grad()
            pred_var = Variable(pred, requires_grad=True)
            energy = self.global_network(pred_var, pots) \
                   - self.entropy_coef * (pred_var * torch.log(pred_var + EPS)).sum(dim=1)
            energy.sum().backward()

            if self.use_linear_decay:
                lr = self.inf_lr / iteration
            else:
                lr = self.inf_lr / np.sqrt(iteration)
            result = lr * pred_var.grad.view(-1, self.num_vals)
            max_result, _ = torch.max(result, dim=1)
            result = pred.view(-1, self.num_vals) * torch.exp(result - max_result.view(-1, 1))
            pred = (result / (result.sum(dim=1, keepdim=True) + EPS)).view(batch_size, -1)
            region_prediction_eps = torch.norm(prev.view(-1, self.num_vals) - pred.view(-1, self.num_vals), dim=1).max().item()
            if self.inf_region_eps is not None and region_prediction_eps < self.inf_region_eps:
                break
            prev = pred
        return pred

    def loss(self, epoch, xs, ys):
        self.global_network.train()
        self.feature_network.train()
        pots = self.feature_network(xs)
        preds = self._emd_inf(pots)
        self.zero_grad()
        pred_energy = self.global_network(preds, pots)
        true_energy = self.global_network(ys, pots)
        loss = pred_energy - true_energy \
             - self.entropy_coef * (preds * torch.log(preds + EPS)).sum(dim=1)
        return loss.mean()

    def calculate_beliefs(self, xs):
        pots = self.feature_network(xs)
        preds = self._emd_inf(pots)
        return preds

    def predict(self, xs):
        pots = self.feature_network(xs)
        preds = self._emd_inf(pots)
        preds = preds.view(-1, self.num_vals).argmax(dim=1).view(-1, self.num_nodes)
        return preds
            

def load_bibtex(path):
    xs, ys = [], []
    with open(path) as f:
        for sample in arff.load(f)['data']:
            sample = list(map(int, sample))
            xs.append(sample[:-LABELS])
            ys.append(sample[-LABELS:])
    xs = torch.tensor(xs, dtype=torch.float32)
    ys = torch.tensor(ys, dtype=torch.int64)
    return xs, ys


class BibtexDataset(Dataset):
    def __init__(self, xs, ys, flip_prob=None):
        self.xs = xs
        self.ys = ys
        self.onehot_ys = onehot(self.ys)
        if flip_prob is not None:
            self.bernoulli = Bernoulli(probs=flip_prob)
        else:
            self.bernoulli = None

    def __len__(self):
        return self.xs.size(0)

    def __getitem__(self, indices):
        xs = self.xs[indices, :]
        if self.bernoulli:
            mask = self.bernoulli.sample(torch.Size([INPUTS]))
            xs = xs * (1 - mask) + (1 - xs) * mask
        return xs, self.ys[indices, :], self.onehot_ys[indices, :]


def test(model, dataset, cfg):
    total_f1 = total_precision = total_recall = total_correct = total_vars = 0
    for xs, ys, _ in DataLoader(
            dataset, pin_memory=cfg.gpu, batch_size=cfg.batch_size):
        preds = model.predict(xs)
        correct = (preds * ys).sum(1).float()
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
    for xs, ys, _ in DataLoader(
            dataset, pin_memory=cfg.gpu, batch_size=cfg.batch_size):
        num_vars += ys.numel()
        node_beliefs = model.calculate_beliefs(
            xs).reshape(-1, 2)[:, 1]
        for i, threshold in enumerate(thresholds):
            preds = (node_beliefs > threshold).long().view(-1, 159)
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


def save_model(model, path, base_name, cfg):
    pass


def train(model, train_data, val_data, cfg, train_logger, val_logger):
    global best_f1, best_epoch, best_threshold
    best_f1, best_epoch, best_threshold = 0., -1, -1

    def validation(epoch):
        global best_f1, best_epoch, best_threshold
        model.eval()
        if cfg.tune_thresholds:
            test_fun = test_with_thresholds
        else:
            test_fun = lambda *args: test(*args), None

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
            save_model(model, cfg.working_dir, 'model', cfg)

        save_model(model, cfg.working_dir, 'model_checkpoint', cfg)
        subset = 'val' if val_data is not None else 'train'
        logger.info(
            f'best {subset} results (epoch {best_epoch}, thresh {best_threshold}): {best_f1}')

    train_data_loader = DataLoader(train_data,
                                   batch_size=cfg.batch_size,
                                   pin_memory=cfg.gpu,
                                   shuffle=True,
                                   drop_last=True)
    opt = model.get_optimizer(cfg)
    for epoch in range(cfg.num_epochs):
        logger.info(f'epoch {epoch + 1}')
        train_logger.update_epoch()
        val_logger.update_epoch()
        if epoch % cfg.val_interval == 0:
            validation(epoch)
        avg_loss, count = 0, 0
        model.train()
        for xs, ys, onehot_ys in train_data_loader:
            if not cfg.use_cross_ent:
                ys = onehot_ys
            loss = model.loss(epoch, xs, ys)
            logger.info(
                f'loss of batch {count}/{len(train_data_loader)}: {loss.item()}')
            loss.backward()
            avg_loss += loss
            count += 1
            if cfg.get('clip_grad', False):
                nn.utils.clip_grad_value_(model.parameters(), cfg.clip_grad)
            elif cfg.get('clip_grad_norm', False):
                nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.clip_grad_norm)
            opt.step()
        train_logger.plot_obj_val(avg_loss / count)
    validation(cfg.num_epochs)


@hydra.main(config_path='config.yaml')
def main(cfg: DictConfig) -> None:
    print(cfg.pretty())
    print(cfg.get('aaa', False))
    cwd = Path(hydra.utils.get_original_cwd())

    train_xs, train_ys = load_bibtex(cwd / cfg.train)
    index = int(len(train_xs) * cfg.train_ratio)
    train_data = BibtexDataset(
        train_xs[:index, :], train_ys[:index, :], flip_prob=cfg.flip_prob)
    val_data = BibtexDataset(train_xs[index:, :], train_ys[index:, :])
    # test_data = BibtexDataset(*load_bibtex(cwd / cfg.test))

    model = UnaryModel(LABELS, 2, cfg)

    with TensorBoard(cwd / cfg.working_dir / 'train') as train_logger, \
            TensorBoard(cwd / cfg.working_dir / 'val') as val_logger:
        train(model,
              train_data,
              val_data,
              cfg,
              train_logger,
              val_logger)


if __name__ == '__main__':
    main()
