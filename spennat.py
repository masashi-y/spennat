import arff
import hydra
import numpy as np
import logging
import time

import spacy
from spacy.tokens import Doc

from pathlib import Path
from omegaconf import DictConfig
from collections import Counter

import torch
from torch.utils.data import DataLoader, Dataset
from torch.autograd import variable
from torch.distributions.bernoulli import Bernoulli
import torch.nn as nn
import torch.nn.functional as F

from experiment_utils import TensorBoard

logger = logging.getLogger(__file__)

EPS = 1e-6

def get_device(gpu_id):
    if gpu_id is not None and gpu_id >= 0:
        return torch.device('cuda', gpu_id)
    else:
        return torch.device('cpu')


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


class FeatureNetwork(nn.Module):
    def __init__(self, hidden_size, cfg):
        super().__init__()
        self.linear = nn.Linear(hidden_size, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.learning_conf = get_learing_conf(self, cfg)

    def forward(self, xs):
        """
        Arguments:
            xs {torch.Tensor} -- (source sent length, batch size, hidden size)
        
        Returns:
            torch.Tensor -- (source sent length, batch size, d_model)
        """
        xs = self.dropout(self.linear(xs))
        return xs


class AttentionalEnergyNetwork(nn.Module):
    def __init__(self, bit_size, cfg):
        super().__init__()
        self.linear = nn.Linear(bit_size, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.linear_out = nn.Linear(cfg.d_model, 1)
        self.model = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.nhead,
                dim_feedforward=cfg.dim_feedforward,
                dropout=cfg.dropout,
                activation=cfg.activation
            ),
            cfg.num_layers
        )
        self.learning_conf = get_learing_conf(self, cfg)

    def forward(self, ys, potentials):
        """
        Arguments:
            ys {torch.Tensor} -- (target sent length, batch size, bit size, 2)
            potentials {torch.Tensor} -- (source sent length, batch size, hidden size)
        
        Returns:
            torch.Tensor -- (batch size,)
        """
        ys = self.dropout(self.linear(ys[:, :, :, 1]))
        ys = self.model(ys, potentials)
        ys = self.linear_out(ys).squeeze().sum(0)
        return ys


class SPENModel(nn.Module):
    def __init__(self, hidden_size, bit_size, cfg):
        super().__init__()
        self.bit_size = bit_size
        self.hidden_size = hidden_size
        self.feature_network = FeatureNetwork(hidden_size, cfg.feature_network)
        self.global_network = AttentionalEnergyNetwork(bit_size, cfg.global_network)
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

    def _random_probabilities(self, batch_size, target_size, device=None):
        """returns a tensor with shape (target sent length, batch size, bit size, 2)
        that sums to one at the last dimension
        
        Arguments:
            batch_size {int} -- batch size
            target_size {int} -- target sent length
            device {torch.device} -- torch device
        
        Returns:
            torch.Tensor -- (target sent length, batch size, bit size, 2)
        """
        x = torch.rand(target_size, self.bit_size, 2, device=device)
        x[:, :, 0] = 1 - x[:, :, 1]
        return x[:, None, :, :].expand(-1, batch_size, -1, -1).contiguous()

    def _gradient_based_inference(self, potentials, max_target_length):
        """
        Arguments:
            potentials {torch.Tensor} -- (source sent length, batch size, hidden size)
        
        Returns:
            torch.Tensor -- (target sent length, batch size, bit size, 2)
        """
        batch_size = potentials.size(1)
        potentials = potentials.detach()
        pred = self._random_probabilities(
            batch_size, max_target_length, potentials.device)
        prev = pred
        prev_energy = prev.new_full((batch_size,), -float('inf'))
        for iteration in range(1, self.inference_iterations):
            self.global_network.zero_grad()
            pred = pred.detach().requires_grad_()
            energy = self.global_network(pred, potentials) \
                   - self.entropy_coef * (pred * torch.log(pred + EPS)).sum(dim=(0,2,3))
            eps = torch.abs(energy - prev_energy).max().item()
            if self.inference_eps is not None and eps < self.inference_eps:
                break
            prev_energy = energy.detach()

            energy.sum().backward()
            if self.use_sqrt_decay:
                lr = self.inference_learning_rate / np.sqrt(iteration)
            else:
                lr = self.inference_learning_rate / iteration
            lr_grad = lr * pred.grad
            max_grad, _ = lr_grad.max(dim=-1, keepdim=True)
            pred = pred * torch.exp(lr_grad - max_grad)
            pred = pred / (pred.sum(dim=-1, keepdim=True) + EPS)
            eps = (prev - pred).norm(dim=-1).max().item()
            if self.inference_region_eps is not None and eps < self.inference_region_eps:
                break
            prev = pred
        return pred

    def loss(self, xs, ys):
        """
        Arguments:
            xs {torch.Tensor} -- (source sent length, batch size, hidden size)
            ys {torch.Tensor} -- (target sent length, batch size, bit size, 2)
        
        Returns:
            torch.Tensor -- loss (batch size,)
        """
        max_target_length = ys.size(0)
        potentials = self.feature_network(xs)
        preds = self._gradient_based_inference(potentials, max_target_length)
        self.zero_grad()
        pred_energy = self.global_network(preds, potentials)
        true_energy = self.global_network(ys, potentials)
        loss = pred_energy - true_energy \
             - self.entropy_coef * (preds * torch.log(preds + EPS)).sum(dim=(0,2,3))
        return loss.mean()

    def predict_beliefs(self, xs, max_target_length):
        """
        Arguments:
            xs {torch.Tensor} -- (source sent length, batch size, hidden size)
            max_target_length {int} -- max target sentence length
        
        Returns:
            torch.Tensor -- (max target sent length, batch size, bit size, 2)
        """
        potentials = self.feature_network(xs)
        preds = self._gradient_based_inference(potentials, max_target_length)
        return preds

    def predict(self, xs, max_target_length):
        """
        Arguments:
            xs {torch.Tensor} -- (source sent length, batch size, hidden size)
            max_target_length {int} -- max target sentence length
        
        Returns:
            torch.Tensor -- (max target sent length, batch size, bit size)
        """
        preds = self.predict_beliefs(xs, max_target_length)
        return preds.argmax(dim=3)


def bit_representation(num, max_size):
    bit_str = bin(num)[2:].rjust(max_size, '0')
    return list(map(int, bit_str))


class FraEngDataset(Dataset):
    def __init__(self, path, spacy_model, device=None, num_samples=10000):
        self.nlp = spacy_model
        self.device = device
        self.source_docs = []
        self.target_sents = []
        self.target_bits = []
        self.meta_data = []
        target_word_count = Counter()
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
        for line in lines[: min(num_samples, len(lines) - 1)]:
            x, y, _ = line.split('\t')
            y = y.lower()
            self.meta_data.append({ 'source': x, 'target': y })
            self.source_docs.append(self.nlp.tokenizer(x))
            y = y.split(' ')
            for word in y:
                target_word_count[word] += 1
            self.target_sents.append(y)
        with self.nlp.disable_pipes(['sentencizer']):
            for _, proc in self.nlp.pipeline:
                self.source_docs = proc.pipe(self.source_docs, batch_size=32)
        self.source_docs = list(self.source_docs)
        self.bit_size = len(bin(len(target_word_count))) - 2
        self.rank = {
            word: bit_representation(rank, self.bit_size) \
            for rank, (word, _) in enumerate(target_word_count.most_common(), 1)
        }
        for sent in self.target_sents:
            sent_tensor = torch.tensor(
                [self.rank[word] for word in sent],
                dtype=torch.long)
            self.target_bits.append(sent_tensor)

    def __len__(self):
        return len(self.source_docs)

    def __getitem__(self, index):
        """
        Arguments:
            index {int} -- element index
        
        Returns:
            torch.Tensor -- (source sent length, hidden size)  e.g., output of BERT
            torch.Tensor -- (target sent length, bit size) bit representation of target sentence
            dict -- meta data concerning the example
        """
        xs = torch.from_numpy(
                self.source_docs[index]._.trf_last_hidden_state)
        ys = self.target_bits[index]
        xs = xs.to(self.device)
        ys = ys.to(self.device)
        return xs, ys


def collate_fun(batch):
    """
    Arguments:
        batch {List[Tuple[torch.Tensor, torch.Tensor]]} -- a list of dataset elements
    
    Returns:
        torch.Tensor -- (source sent length, batch size, hidden size)
        torch.Tensor -- (target sent length, batch size, bit size)
        List[Dict[str, str]] -- list of meta data
    """
    xs, ys, meta_data = zip(*batch)
    xs = torch.nn.utils.rnn.pad_sequence(xs)
    ys = torch.nn.utils.rnn.pad_sequence(ys)
    return xs, ys, meta_data


def save_model(model, file_path):
    with open(file_path, 'wb') as f:
        torch.save(model.state_dict(), f)


def load_model(model, file_path):
    with open(file_path, 'rb') as f:
        model.load_state_dict(torch.load(f))
            

def test(model, dataset, cfg, threshold=None):
    total_correct = total_vars = 0
    for xs, ys, _ in DataLoader(dataset,
            batch_size=cfg.batch_size, collate_fn=collate_fun):
        if threshold is None:
            preds = model.predict(xs, ys.size(0))
        else:
            node_beliefs = model.predict_beliefs(xs, ys.size(0))[:, :, :, 1]
            preds = (node_beliefs > threshold).long()
        total_correct += (preds == ys).float().sum()
        total_vars += ys.numel()
    acc = (total_correct / total_vars).item()
    return acc


def test_with_thresholds(model, dataset, cfg):
    num_vars = 0
    thresholds = np.arange(0.05, 0.80, 0.05)
    total_accs = np.zeros_like(thresholds)
    for xs, ys, _ in DataLoader(dataset,
            batch_size=cfg.batch_size, collate_fn=collate_fun):
        num_vars += ys.numel()
        node_beliefs = model.predict_beliefs(xs, ys.size(0))[:, :, :, 1]
        for i, threshold in enumerate(thresholds):
            preds = (node_beliefs > threshold).long()
            total_accs[i] += (preds == ys).float().sum()
    accs = total_accs / num_vars
    best = accs.argmax()
    return accs[best], thresholds[best]


def train(model, dataset, val_data, cfg, train_logger, val_logger):
    global best_acc, best_epoch, best_threshold
    best_acc, best_epoch, best_threshold = 0., -1, -1

    def validation(epoch):
        global best_acc, best_epoch, best_threshold
        model.eval()
        if cfg.tune_thresholds:
            test_fun = test_with_thresholds
        else:
            test_fun = lambda *args: (test(*args), None)

        acc, threshold = test_fun(model, dataset, cfg)
        logger.info(
            f'train results: {acc}, (threshold: {threshold})')
        train_logger.plot_for_current_epoch('Accuracy', acc)
        if val_data is not None:
            acc, threshold = test_fun(model, val_data, cfg)
            logger.info(
                f'val results: {acc}, (threshold: {threshold})')
            val_logger.plot_for_current_epoch('Accuracy', acc)

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            if cfg.tune_thresholds:
                best_threshold = threshold
            logger.info('new best accuracy found, saving model')
            save_model(model, 'best_model')

        save_model(model, f'model_checkpoint_{epoch}')
        subset = 'val' if val_data is not None else 'train'
        logger.info(
            f'best {subset} results (epoch {best_epoch}, threshold {best_threshold}): {best_acc}')

    train_data_loader = DataLoader(dataset,
                                   batch_size=cfg.batch_size,
                                   shuffle=True,
                                   drop_last=True,
                                   collate_fn=collate_fun)
    opt = model.get_optimizer()
    for epoch in range(cfg.num_epochs):
        logger.info(f'epoch {epoch + 1}')
        train_logger.update_epoch()
        val_logger.update_epoch()
        if epoch % cfg.val_interval == 0:
            validation(epoch)
        avg_loss, count = 0, 0
        for xs, ys, meta_data in train_data_loader:
            ys = onehot(ys)
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


@hydra.main(config_path='configs/spennat.yaml')
def main(cfg: DictConfig) -> None:
    logger.info(cfg.pretty())

    device = get_device(cfg.device)
    if cfg.device >= 0:
        spacy.prefer_gpu(cfg.device)
    spacy_model = spacy.load(cfg.spacy_model)
    dataset = FraEngDataset(
            hydra.utils.to_absolute_path(cfg.dataset),
            spacy_model,
            device=device,
            num_samples=cfg.num_samples)
    hidden_size = spacy_model.get_pipe('trf_tok2vec').token_vector_width
    max_bit_size = dataset.bit_size
    model = SPENModel(hidden_size, max_bit_size, cfg).to(device)

    with TensorBoard('spennat_train') as train_logger, \
            TensorBoard('spennat_val') as val_logger:
        train(model,
              dataset,
              None,
              cfg,
              train_logger,
              val_logger)
            

if __name__ == '__main__':
    main()
