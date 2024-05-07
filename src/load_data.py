from pathlib import Path
import torch
import torch.utils
import torchvision.transforms.v2 as v2
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torchvision.datasets import SVHN

from datasets.multifidelity_sampler import MultifidelitySampler

PTBXL_PATH = Path('data') / 'ptbxl'
KAGGLE_PPG_PATH = Path('data') / 'kaggle_ppg_bp'
CIFAR10_PATH = Path('data') / 'cifar_10'
SVHN_ROOT = Path('data') / 'svhn'

# v2 expects CHW format
transformation = v2.Compose([
v2.RandomResize(28,36),
v2.Pad(4),
v2.RandomCrop(32),
v2.RandomHorizontalFlip(),
v2.ToDtype(torch.float32, scale=True),
])

transformation_test = v2.Compose([
    v2.ToDtype(torch.float32, scale=True),
])

mnist_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

transform_svhn = v2.Compose([
    v2.RandomResizedCrop(size=(32, 32), scale=(0.8, 1.0), interpolation=v2.InterpolationMode.BILINEAR),  # Randomly resize and crop
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
])

transform_svhn_val = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
])

#mnist_transform = v2.Compose([
#    v2.ToTensor()
#])

def cifar10_fn(image):
    return transformation(torch.as_tensor(image) / 255.0)

def cifar10_test_fn(image):
    return transformation_test(torch.as_tensor(image) / 255.0)

def mnist_fn(image):
    return mnist_transform(image)

def load_data(config):
    batch_size = config['data']['batch_size']
    shuffle = config['data']['shuffle']
    pin_memory = True if torch.cuda.is_available() else False
    num_workers = 2
    persistent_workers = True

    if config['data']['name'] == 'PTB-XL':
        from datasets.ptbxl import PTBXLDataset

        train = PTBXLDataset (PTBXL_PATH, train=True)
        dev = PTBXLDataset(PTBXL_PATH, train=False)

    elif config['data']['name'] == 'Kaggle_PPG':
        from datasets.kaggle import KaggleDataset

        train = KaggleDataset(KAGGLE_PPG_PATH, train=True)
        dev = KaggleDataset(KAGGLE_PPG_PATH, train=False)

    elif config['data']['name'] == 'CIFAR10':
        from datasets.cifar10 import CIFAR10
        import torchvision.transforms.v2 as v2
        from functools import partial

        cifar10 = CIFAR10(CIFAR10_PATH)
        train = cifar10.dataset('train', cifar10_fn)
        dev = cifar10.dataset('dev', cifar10_test_fn)
    elif config['data']['name'] == 'MNIST':
        train = MNIST(root='data', train=True, download=True, transform=mnist_fn)
        dev = MNIST(root='data', train=False, download=True, transform=mnist_fn)
    elif config['data']['name'] == 'SVHN':
        train = SVHN(root=SVHN_ROOT, split='train', download=True, transform=transform_svhn)
        dev = SVHN(root=SVHN_ROOT, split='test', download=True, transform=transform_svhn_val)
        # SVHN dev set has 5099 out of 26032 samples with y==1, which is 19.58% of the dataset
    else:
        raise ValueError(f"Unknown dataset: {config['data']['name']}")

    mf_sampler = MultifidelitySampler(train)
    dev_sampler = MultifidelitySampler(dev)
    trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                        sampler=mf_sampler, num_workers=num_workers,
                                        pin_memory=pin_memory, persistent_workers=persistent_workers)
    devloader = torch.utils.data.DataLoader(dev, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers,
                                            pin_memory=pin_memory, persistent_workers=persistent_workers)

    if config['data']['name'] == 'MNIST':
        trainloader.sampler.set_fidelity(0.33)
    return trainloader, devloader

if __name__ == '__main__':
    train = SVHN(root=SVHN_ROOT, split='train', download=True, transform=transforms.ToTensor())
    dev = SVHN(root=SVHN_ROOT, split='test', download=True, transform=transforms.ToTensor())

    print()
