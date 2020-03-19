import hydra
import numpy as np
import logging

from omegaconf import DictConfig

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from spennat.tensorboard import TensorBoard
import spennat.utils as utils
from spennat.datasets.bibtex import load_bibtex, INPUTS, LABELS
from spennat.models import UnaryModel, SPENModel


logger = logging.getLogger(__file__)

EPS = 1e-6


class FeatureEnergyNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, cfg):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        layers = []
        for _ in range(cfg.num_layers - 1):
            if cfg.dropout is not None:
                layers.append(nn.Dropout(cfg.dropout))
            linear = nn.Linear(in_dim, cfg.hidden_size)
            linear.weight.data.normal_(std=np.sqrt(2. / in_dim))
            linear.bias.data.fill_(0.)
            layers.extend([linear, nn.ReLU(inplace=True)])
            in_dim = cfg.hidden_size
        layers.append(nn.Linear(in_dim, self.out_dim))
        layers[-1].weight.data.normal_(std=np.sqrt(2. / in_dim))
        layers[-1].bias.data.fill_(0.)
        self.model = nn.Sequential(*layers)

    def forward(self, xs):
        """
        Arguments:
            xs {torch.Tensor} -- (batch size, feature size (i.e., INPUTS))

        Returns:
            [torch.Tensor] -- (batch size, label size (i.e., LABELS), 2),
        """
        batch_size = xs.size(0)
        result = xs.new_zeros(batch_size, self.out_dim, 2)
        result[:, :, 1].copy_(self.model(xs))
        return result


class GlobalEnergyNetwork(nn.Module):
    def __init__(self, num_labels, cfg):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(num_labels, cfg.hidden_size, bias=False),
            nn.Softplus(),
            nn.Linear(cfg.hidden_size, 1, bias=False))

    def forward(self, ys, potentials):
        global_energy = (ys * potentials).sum(dim=(1, 2))
        label_energy = self.model(ys[:, :, 1]).squeeze(0)
        return global_energy + label_energy


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
            'train results: (%f, %f, %f, %f), (threshold: %f)', acc, prec, rec, f1, threshold)
        train_logger.plot_for_current_epoch('F1', f1)
        train_logger.plot_for_current_epoch('Accuracy', acc)
        if val_data is not None:
            (acc, prec, rec, f1), threshold = test_fun(model, val_data, cfg)
            logger.info(
               'val results: (%f, %f, %f, %f), (threshold: %f)', acc, prec, rec, f1, threshold)
            val_logger.plot_for_current_epoch('F1', f1)
            val_logger.plot_for_current_epoch('Accuracy', acc)

        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch
            if cfg.tune_thresholds:
                best_threshold = threshold
            logger.info('new best f1 found, saving model')
            utils.save_model(model, 'best_model')

        utils.save_model(model, f'model_checkpoint_{epoch}')
        subset = 'val' if val_data is not None else 'train'
        logger.info(
            'best %s results (epoch %d, threshold %f): %f', subset, best_epoch, best_threshold, best_f1)

    train_data_loader = DataLoader(
        train_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True
    )
    optimizer = utils.optimizer_of(
        cfg.optimizer,
        [param for param in model.parameters() if param.requires_grad]
    )
    for epoch in range(1, cfg.num_epochs + 1):
        logger.info('epoch %d', epoch)
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
            logger.info('loss of batch %d/%d: %f', count, len(train_data_loader), loss.item())
            loss.backward()
            avg_loss += loss
            count += 1
            if cfg.clip_grad:
                nn.utils.clip_grad_value_(model.parameters(), cfg.clip_grad)
            elif cfg.clip_grad_norm:
                nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.clip_grad_norm)
            optimizer.step()
        train_logger.plot_obj_val((avg_loss / count).item())
    validation(cfg.num_epochs)


@hydra.main(config_path='config.yaml')
def main(cfg: DictConfig) -> None:
    logger.info(cfg.pretty())

    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)

    device = utils.get_device(cfg.device)
    train_data, val_data, test_data =  load_bibtex(
        hydra.utils.to_absolute_path(cfg.train),
        hydra.utils.to_absolute_path(cfg.test),
        train_ratio=cfg.train_ratio,
        device=device)

    if cfg.model == 'spen':
        model = SPENModel(
            FeatureEnergyNetwork(INPUTS, LABELS, cfg.feature_network),
            GlobalEnergyNetwork(LABELS, cfg.global_network),
            cfg,
            num_nodes=LABELS,
            num_vals=2)
    elif cfg.model == 'unary':
        model = UnaryModel(
            FeatureEnergyNetwork(INPUTS, LABELS, cfg.feature_network),
            num_nodes=LABELS,
            num_vals=2)
    else:
        assert False

    if cfg.pretrained_unary:
        unary_model = UnaryModel(
            FeatureEnergyNetwork(INPUTS, LABELS, cfg.feature_network),
            num_nodes=LABELS,
            num_vals=2)
        utils.load_model(unary_model, hydra.utils.to_absolute_path(cfg.pretrained_unary))
        model.feature_network = unary_model.feature_network
        if cfg.feature_network.freeze:
            model.feature_network.requires_grad_(False)
    model = model.to(device)

    with TensorBoard(f'{cfg.model}_train') as train_logger, \
            TensorBoard(f'{cfg.model}_val') as val_logger:
        train(model,
              train_data,
              val_data,
              cfg,
              train_logger,
              val_logger)

    logger.info('loading best model...')
    utils.load_model(model, 'best_model')
    logger.info('finding best threshold on validation data...')
    (val_acc, val_prec, val_rec, val_f1), threshold = test_with_thresholds(model, val_data, cfg)
    logger.info('testing on training data...')
    train_acc, train_prec, train_rec, train_f1 = test(model, train_data, cfg, threshold=threshold)
    logger.info('testing on test data...')
    test_acc, test_prec, test_rec, test_f1 = test(model, test_data, cfg, threshold=threshold)

    logger.info('final results: (threshold: %d)', threshold)
    logger.info('train: %f, %f, %f, %f', train_acc, train_prec, train_rec, train_f1)
    logger.info('val:   %f, %f, %f, %f', val_acc, val_prec, val_rec, val_f1)
    logger.info('test:  %f, %f, %f, %f', test_acc, test_prec, test_rec, test_f1)
            

if __name__ == '__main__':
    main()
