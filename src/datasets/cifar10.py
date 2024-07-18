# Dataset CIFAR-10 source: https://www.cs.toronto.edu/~kriz/cifar.html
import os
from pathlib import Path

import numpy as np
import torch

class TorchDataset(torch.utils.data.Dataset):
    _data = None
    _size = 0
    _transform = None

    def __init__(self, data, transform):
        self._data = data
        self._size = len(self._data[1])
        self._transform = transform

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        img, label = self._data[0][index], self._data[1][index]
        if self._transform is not None:
            img = self._transform(img)
        return img, label

class CIFAR10:
    H: int = 32
    W: int = 32
    C: int = 3
    LABELS: list[str] = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    train = None
    dev = None
    test = None

    def __init__(self, path):
        if self.train is None:
            self.load_data(path)

    @classmethod
    def load_data(self, path: Path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        import time
        start = time.perf_counter()
        cifar = np.load(path / "cifar10.npz")
        end = time.perf_counter()
        print(f'Loading took: {end - start}')
        data = cifar['data.npy']
        labels = cifar['labels.npy']

        # Reshape data to NCHW format and transpose to NHWC
        data = data.reshape(-1, self.C, self.H, self.W)#.transpose(0, 2, 3, 1)
        labels = labels.astype(np.int64)

        dev_split = 45000
        test_split = 50000

        self.train = (data[:dev_split], labels[:dev_split])
        self.dev = (data[dev_split:test_split], labels[dev_split:test_split])
        self.test = (data[test_split:], labels[test_split:])

    def dataset(self, type:str, transform) -> torch.utils.data.Dataset:
        if type == 'train':
            return TorchDataset(self.train, transform)
        elif type == 'dev':
            return TorchDataset(self.dev, transform)
        elif type == 'test':
            return TorchDataset(self.test, transform)
        else:
            raise ValueError(f"Unknown dataset type: {type}")

if __name__ == '__main__':
    cifar10 = CIFAR10(Path('data') / 'cifar_10')
    print()