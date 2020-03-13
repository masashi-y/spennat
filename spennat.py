import sys
import math
import hydra
import numpy as np
import logging
import random

import spacy

from omegaconf import DictConfig
from collections import Counter

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

from experiment_utils import TensorBoard
from transformer import (TransformerDecoderLayer,
                         TransformerDecoder,
                         MyDropout,
                         PositionalEncoding)

logger = logging.getLogger(__file__)

EPS = 1e-6


def get_device(gpu_id):
    if gpu_id is not None and gpu_id >= 0:
        return torch.device('cuda', gpu_id)
    return torch.device('cpu')


def onehot(x, num_values):
    x0 = x.view(-1, 1)
    x1 = x.new_zeros(len(x0), num_values, dtype=torch.float)
    x1.scatter_(1, x0, 1)
    return x1.view(x.size() + (num_values,))


def blunt_onehot_with_softmax(x, temperature=1):
    x = torch.softmax(x / temperature, 2)
    rnd = torch.randn_like(x) / 100.  # add some noise
    return torch.clamp(x + rnd, 0., 1.)


optimizers = {
    'adadelta': torch.optim.Adadelta,
    'adagrad': torch.optim.Adagrad,
    'lbfgs': torch.optim.LBFGS,
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW,
    'adamax': torch.optim.Adamax,
    'asgd': torch.optim.ASGD,
    'sgd': torch.optim.SGD,
    'rmsprop': torch.optim.RMSprop,
    'rprop': torch.optim.Rprop,
}


def optimizer_of(string, params):
    """Returns a optimizer object based on input string, e.g., adagrad(lr=0.01, lr_decay=0)
    Arguments:
        string {str} -- string expression of an optimizer
        params {List[torch.Tensor]} -- parameters to learn

    Returns:
        torch.optim.Optimizer -- optimizer
    """
    index = string.find('(')
    assert string[-1] == ')'
    try:
        optim_class = optimizers[string[:index]]
    except KeyError:
        raise KeyError(
            f'Optimizer class "{string[:index]}" does not exist.\n'
            f'Please choose one among: {list(optimizers.keys())}'
        )
    kwargs = eval(f'dict{string[index:]}')
    return optim_class(params, **kwargs)


class FeatureNetwork(nn.Module):
    def __init__(self, hidden_size, cfg):
        super().__init__()
        in_dim, out_dim = hidden_size, cfg.d_model
        layers = []
        for i in range(cfg.num_layers):
            if cfg.dropout is not None:
                layers.append(nn.Dropout(cfg.dropout))
            linear = nn.Linear(in_dim, out_dim)
            linear.weight.data.normal_(std=np.sqrt(2. / in_dim))
            linear.bias.data.fill_(0.)
            layers.append(linear)
            if i + 1 < cfg.num_layers:
                layers.append(nn.ReLU(inplace=True))
            in_dim = out_dim
        self.model = nn.Sequential(*layers)

    def forward(self, xs):
        """
        Arguments:
            xs {torch.Tensor} -- (source sent length, batch size, hidden size)
        
        Returns:
            torch.Tensor -- (source sent length, batch size, d_model)
        """
        xs = self.model(xs)
        return xs


class AttentionalEnergyNetwork(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        self.linear = nn.Linear(vocab_size, cfg.d_model)
        self.dropout = MyDropout(cfg.dropout)
        self.linear_out = nn.Linear(cfg.d_model, 1)
        self.pos_encoder = PositionalEncoding(cfg.d_model, cfg.dropout)
        self.model = TransformerDecoder(
            TransformerDecoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.nhead,
                dim_feedforward=cfg.dim_feedforward,
                dropout=cfg.dropout,
                activation=cfg.activation
            ),
            cfg.num_layers
        )

    def forward(self, ys, potentials):
        """
        Arguments:
            ys {torch.Tensor} -- (target sent length, batch size, vocab size)
            potentials {torch.Tensor} -- (source sent length, batch size, hidden size)

        Returns:
            torch.Tensor -- (batch size,)
        """
        ys = self.dropout(self.linear(ys))
        ys = self.pos_encoder(ys)
        ys = self.model(ys, potentials)
        ys = self.linear_out(ys).squeeze(2).sum(0)
        return ys


class SPENModel(nn.Module):
    def __init__(self, cfg, hidden_size, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.feature_network = FeatureNetwork(self.hidden_size, cfg.feature_network)
        self.global_network = AttentionalEnergyNetwork(self.vocab_size, cfg.global_network)
        self.inference_iterations = cfg.inference.iterations
        self.inference_learning_rate = cfg.inference.learning_rate
        self.inference_eps = cfg.inference.eps
        self.inference_region_eps = cfg.inference.region_eps
        self.use_sqrt_decay = cfg.inference.use_sqrt_decay
        self.entropy_coef = cfg.entropy_coef

    def _random_probabilities(self, batch_size, target_size, device=None):
        """returns a tensor with shape (target sent length, batch size, vocab size)
        that sums to one at the last dimension

        Arguments:
            batch_size {int} -- batch size
            target_size {int} -- target sent length
            device {torch.device} -- torch device

        Returns:
            torch.Tensor -- (target sent length, batch size, vocab size)
        """
        x = torch.rand(target_size, self.vocab_size + 1, device=device)
        x[:, 0] = 0.
        x[:, -1] = 1.
        x, _ = x.sort(1)
        x = x[:, 1:] - x[:, :-1]
        return x[:, None, :].expand(-1, batch_size, -1).contiguous()

    def _gradient_based_inference(self, potentials, max_target_length, init_pred=None):
        """
        Arguments:
            potentials {torch.Tensor} -- (source sent length, batch size, hidden size)

        Returns:
            torch.Tensor -- (target sent length, batch size, vocab size)
        """
        batch_size = potentials.size(1)
        potentials = potentials.detach()
        pred = init_pred or self._random_probabilities(
            batch_size, max_target_length, potentials.device)
        prev = pred
        prev_energy = prev.new_full((batch_size,), -float('inf'))
        for iteration in range(1, self.inference_iterations):
            self.global_network.zero_grad()
            pred = pred.detach().requires_grad_()
            energy = self.global_network(pred, potentials) \
                   - self.entropy_coef * (pred * torch.log(pred + EPS)).sum(dim=(0, 2))
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
        logger.info('number of iterations: %d', iteration)
        return pred

    def loss(self, xs, ys):
        """
        Arguments:
            xs {torch.Tensor} -- (source sent length, batch size, hidden size)
            ys {torch.Tensor} -- (target sent length, batch size, vocab size)

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
             - self.entropy_coef * (preds * torch.log(preds + EPS)).sum(dim=(0, 2))
        loss = torch.max(loss, torch.zeros_like(loss))
        return loss.mean()

    def predict(self, xs, max_target_length, init_pred=None):
        """
        Arguments:
            xs {torch.Tensor} -- (source sent length, batch size, hidden size)
            max_target_length {int} -- max target sentence length

        Returns:
            torch.Tensor -- (max target sent length, batch size)
        """
        potentials = self.feature_network(xs)
        preds = self._gradient_based_inference(
            potentials, max_target_length, init_pred=init_pred)
        return preds.argmax(dim=2)


class BasicVocab(object):
    def __init__(self, counter, count_threshold=1):
        counter['UNK'] = len(counter)
        frequent_words = [
            word for word, count in counter.most_common() \
            if count >= count_threshold
        ]
        self.vocab_size = len(frequent_words)
        self.index2word = {
            rank: word for rank, word in enumerate(frequent_words)
        }
        self.word2index = {
            word: self.onehot_representation(rank) \
            for rank, word in self.index2word.items()
        }

    def onehot_representation(self, rank):
        return tuple(int(i == rank) for i in range(self.vocab_size))

    def __len__(self):
        return len(self.word2index)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.word2index.get(key, self.word2index['UNK'])
        return self.index2word.get(int(key), 'UNK')


class FraEngDataset(Dataset):
    def __init__(self,
                 source_docs,
                 target_bits,
                 meta_data,
                 device=None):
        assert len(source_docs) == len(target_bits) == len(meta_data)
        self.source_docs = source_docs
        self.target_bits = target_bits
        self.meta_data = meta_data
        self.device = device

    def __len__(self):
        return len(self.source_docs)

    def __getitem__(self, index):
        """
        Arguments:
            index {int} -- element index

        Returns:
            torch.Tensor -- (source sent length, hidden size)  e.g., output of BERT
            torch.Tensor -- (target sent length, vocab size) one hot representation of target sentence
            dict -- meta data concerning the example
        """
        xs = torch.from_numpy(
            self.source_docs[index]._.trf_last_hidden_state)
        ys = self.target_bits[index]
        xs = xs.to(self.device)
        ys = ys.to(self.device)
        meta = self.meta_data[index]
        return xs, ys, meta


def load_fra_eng_dataset(file_path, spacy_model, cfg, device=None):
    source_docs = []
    target_sents = []
    meta_data = []
    target_word_count = Counter()
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    for line in lines[: min(cfg.num_samples, len(lines) - 1)]:
        x, y, _ = line.split('\t')
        y = y.lower()
        meta_data.append({ 'source': x, 'target': y })
        source_docs.append(spacy_model.tokenizer(x))
        y = y.split()
        target_word_count.update(word for word in y)
        target_sents.append(y)
    with spacy_model.disable_pipes(['sentencizer']):
        for _, proc in spacy_model.pipeline:
            source_docs = proc.pipe(source_docs, batch_size=32)
    logger.info('max target sentence length: %d', max(map(len, target_sents)))
    source_docs = list(source_docs)
    vocab = BasicVocab(
        target_word_count, count_threshold=cfg.vocab_count_threshold)
    target_bits = [
        torch.tensor(
            [vocab[word] for word in sent],
            dtype=torch.float) for sent in target_sents
    ]
    for meta in meta_data:
        meta['target_unked'] = \
            ' '.join(word if word in vocab.word2index else 'UNK' \
                    for word in meta['target'].split())
    dataset = FraEngDataset(
        source_docs, target_bits, meta_data, device=device)
    return dataset, vocab

    
def collate_fun(batch):
    """
    Arguments:
        batch {List[Tuple[torch.Tensor, torch.Tensor]]} -- a list of dataset elements
    
    Returns:
        torch.Tensor -- (source sent length, batch size, hidden size)
        torch.Tensor -- (target sent length, batch size, vocab size)
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
            

def test(model, dataset, cfg):
    total_correct = total_vars = 0
    for xs, ys, _ in DataLoader(dataset,
                                batch_size=cfg.batch_size,
                                collate_fn=collate_fun):
        preds = model.predict(
            xs,
            ys.size(0),
            init_pred=blunt_onehot_with_softmax(ys) if cfg.test_init_with_gold else None
        )
        ys = ys.argmax(2)
        total_correct += (preds == ys).float().sum()
        total_vars += ys.numel()
    acc = (total_correct / total_vars).item()
    return acc


def train(model, vocab, dataset, val_data, cfg, train_logger, val_logger):
    best_acc, best_epoch = 0., -1

    def validation(epoch):
        nonlocal best_acc, best_epoch
        model.eval()
        acc = test(model, dataset, cfg)
        logger.info('train accuracy: %f', acc)
        train_logger.plot_for_current_epoch('Accuracy', acc)
        if val_data is not None:
            acc = test(model, val_data, cfg)
            logger.info('val accuracy: %f', acc)
            val_logger.plot_for_current_epoch('Accuracy', acc)

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            logger.info('new best accuracy found, saving model')
            save_model(model, 'best_model')

        save_model(model, f'model_checkpoint_{epoch}')
        subset = 'val' if val_data is not None else 'train'
        logger.info('best %s results (epoch %d): %f', subset, best_epoch, best_acc)

    train_data_loader = DataLoader(dataset,
                                   batch_size=cfg.batch_size,
                                   shuffle=True,
                                   drop_last=True,
                                   collate_fn=collate_fun)
    optimizer = optimizer_of(
        cfg.optimizer,
        [param for param in model.parameters() if param.requires_grad]
    )
    for epoch in range(1, cfg.num_epochs + 1):
        logger.info('epoch %d', epoch)
        train_logger.update_epoch()
        val_logger.update_epoch()
        if epoch % cfg.val_interval == 0:
            validation(epoch)
            indices = [random.randint(0, len(dataset) - 1) for _ in range(5)]
            xs, ys, meta_data = collate_fun([dataset[index] for index in indices])
            preds = model.predict(
                xs,
                ys.size(0),
                init_pred=blunt_onehot_with_softmax(ys) if cfg.test_init_with_gold else None
            )
            for index, meta in enumerate(meta_data):
                pred = ' '.join(vocab[word_index] for word_index in preds[:, index])
                logger.info('source: %s', meta["source"])
                logger.info('gold: %s', meta["target"])
                logger.info('gold (unked): %s', meta["target_unked"])
                logger.info('pred: %s', pred)
                logger.info('-----------------------')

        avg_loss, count = 0, 0
        for xs, ys, _ in train_data_loader:
            model.train()
            model.zero_grad()
            loss = model.loss(xs, ys)
            logger.info(
                'loss of batch %d/%d: %f', count + 1, len(train_data_loader), loss.item())
            loss.backward()
            avg_loss += loss
            count += 1
            if cfg.clip_grad:
                nn.utils.clip_grad_value_(model.parameters(), cfg.clip_grad)
            elif cfg.clip_grad_norm:
                nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.clip_grad_norm)
            optimizer.step()
        loss = (avg_loss / count).item()
        logger.info('loss of epoch %d: %f', epoch, loss)
        train_logger.plot_obj_val(loss)
    validation(cfg.num_epochs)


@hydra.main(config_path='configs/spennat.yaml')
def main(cfg: DictConfig) -> None:
    logger.info(cfg.pretty())

    if cfg.device >= 0:
        spacy.prefer_gpu(cfg.device)
    device = get_device(cfg.device)
    spacy_model = spacy.load(cfg.spacy_model)
    dataset, vocab = load_fra_eng_dataset(
            hydra.utils.to_absolute_path(cfg.dataset), spacy_model, cfg, device=device)
    logger.info('target language vocab size: %d', len(vocab))

    hidden_size = spacy_model.get_pipe('trf_tok2vec').token_vector_width
    vocab_size = vocab.vocab_size
    model = SPENModel(cfg, hidden_size, vocab_size).to(device)

    with TensorBoard('spennat_train') as train_logger, \
            TensorBoard('spennat_val') as val_logger:
        train(model,
              vocab,
              dataset,
              None,
              cfg,
              train_logger,
              val_logger)
            

if __name__ == '__main__':
    main()
