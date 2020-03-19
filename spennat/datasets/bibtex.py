import arff
import torch
from torch.utils.data import Dataset
from spennat.utils import onehot

INPUTS = 1836
LABELS = 159

def load_bibtex(train_path, test_path, train_ratio=0.75, device=None):
    def load_(path):
        xs, ys = [], []
        with open(path) as f:
            for sample in arff.load(f)['data']:
                sample = list(map(int, sample))
                xs.append(sample[:-LABELS])
                ys.append(sample[-LABELS:])
        xs = torch.tensor(xs, dtype=torch.float, device=device)
        ys = torch.tensor(ys, dtype=torch.long, device=device)
        return xs, ys

    train_xs, train_ys = load_(train_path)
    index = int(len(train_xs) * train_ratio)
    train_data = BibtexDataset(train_xs[:index, :], train_ys[:index, :])
    val_data = BibtexDataset(train_xs[index:, :], train_ys[index:, :])
    test_data = BibtexDataset(*load_(test_path))
    return train_data, val_data, test_data


class BibtexDataset(Dataset):
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys
        self.onehot_ys = onehot(ys, 2)

    def __len__(self):
        return self.xs.size(0)

    def __getitem__(self, index):
        return self.xs[index, :], self.ys[index, :], self.onehot_ys[index, :]


