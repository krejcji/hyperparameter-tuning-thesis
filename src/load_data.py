from pathlib import Path
import torch
import torch.utils

from torch.utils.data.sampler import RandomSampler
from datasets.multifidelity_sampler import MultifidelitySampler

PTBXL_PATH = Path('data') / 'ptbxl'
KAGGLE_PPG_PATH = Path('data') / 'kaggle_ppg_bp'
CIFAR10_PATH = Path('data') / 'cifar_10'
SVHN_ROOT = Path('data') / 'svhn'
NIH_XRAY_PATH = Path('data') / 'NIH'

def load_data(config, create=False):
    # Default sampler is RandomSampeler, but MultifidelitySampler is supported as well
    sampler = config['data'].get('sampler', 'RandomSampler')
    if sampler == 'RandomSampler':
        train_sampler = RandomSampler
    elif sampler == 'MultifidelitySampler':
        train_sampler = MultifidelitySampler
    else:
        raise ValueError(f"Unknown sampler: {sampler}")

    # Default number of dataloader workers
    num_workers = 0
    batch_size = config['data']['batch_size']

    # Pick and load the dataset
    if config['data']['name'] == 'PTB-XL':
        from datasets.ptbxl import PTBXLDataset

        train = PTBXLDataset (PTBXL_PATH, train=True, create=create)
        dev = PTBXLDataset(PTBXL_PATH, train=False, create=False)
    elif config['data']['name'] == 'PTB-XL-Shared':
        from datasets.ptbxl_shared import PTBXLDatasetShared
        unique_id = config.get('unique_id', 0)
        train = PTBXLDatasetShared (PTBXL_PATH, train=True, create=create, id=unique_id)
        dev = PTBXLDatasetShared(PTBXL_PATH, train=False, create=create, id=unique_id)
    elif config['data']['name'] == 'Kaggle_PPG':
        from datasets.kaggle import KaggleDataset

        train = KaggleDataset(KAGGLE_PPG_PATH, train=True)
        dev = KaggleDataset(KAGGLE_PPG_PATH, train=False)
    elif config['data']['name'] == 'CIFAR10':
        from datasets.cifar10 import CIFAR10
        import torchvision.transforms.v2 as v2

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

        def cifar10_fn(image):
            return transformation(torch.as_tensor(image) / 255.0)

        def cifar10_test_fn(image):
            return transformation_test(torch.as_tensor(image) / 255.0)

        cifar10 = CIFAR10(CIFAR10_PATH)
        train = cifar10.dataset('train', cifar10_fn)
        dev = cifar10.dataset('dev', cifar10_test_fn)
    elif config['data']['name'] == 'MNIST':
        from torchvision.datasets import MNIST
        import torchvision.transforms.v2 as v2

        mnist_transform = v2.Compose([
            v2.RandomResize(28,36),
            v2.Pad(4),
            v2.RandomCrop(32),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])

        def mnist_fn(image):
            return mnist_transform(image)

        train = MNIST(root='data', train=True, download=True, transform=mnist_fn)
        dev = MNIST(root='data', train=False, download=True, transform=mnist_fn)
    elif config['data']['name'] == 'SVHN':
        from torchvision.datasets import SVHN
        import torchvision.transforms.v2 as v2
        from torchvision.transforms import InterpolationMode

        if 'rotation' in config: # svhn_residual
            num_workers = 2
            rotation = config.get('rotation', 0)
            trans = config.get('translate', 0.1)
            scale_factor = config.get('scale_factor', 0.1)
            scale_offset = config.get('scale_offset', 0.1)
            scale_min = 1 - scale_factor - scale_offset
            scale_max = 1 + scale_factor - scale_offset
            sharpness_factor = config.get('sharpness_factor', 1)

            transform_svhn = v2.Compose([
                v2.ToImage(),
                v2.RandomAffine(degrees=rotation,
                                translate=(trans, trans),
                                scale=(scale_min, scale_max),
                                interpolation=InterpolationMode.BILINEAR),
                v2.RandomAdjustSharpness(sharpness_factor=sharpness_factor),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
            ])
        else: # svhn_simple
            num_workers = 1
            transform_svhn = v2.Compose([
                v2.ToImage(),
                v2.RandomAffine(degrees=10,
                                translate=(0.1, 0.1),
                                scale=(0.8, 1.1),
                                interpolation=InterpolationMode.BILINEAR),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
            ])

        transform_svhn_val = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
        ])

        train = SVHN(root=SVHN_ROOT, split='train', download=True, transform=transform_svhn)
        dev = SVHN(root=SVHN_ROOT, split='test', download=True, transform=transform_svhn_val)
        # SVHN dev set has 5099 out of 26032 samples with y==1, which is 19.58% of the dataset
    elif config['data']['name'] == 'xray':
        from datasets.torchxrayvision import NIH_Dataset
        from datasets.torchxrayvision import SubsetDataset
        from sklearn.model_selection import GroupShuffleSplit
        import torchvision.transforms.v2 as v2
        from torchvision.transforms import InterpolationMode

        num_workers = 2

        imgpath = NIH_XRAY_PATH / "images-224"
        csvpath = NIH_XRAY_PATH / "Data_Entry_2017_v2020.csv.gz"
        bboxpath = NIH_XRAY_PATH / "BBox_List_2017.csv.gz"

        data_aug = None
        unique_patients = True

        rot = config.get('rotation', 5)
        trans = config.get('translate', 0.1)
        scale = config.get('resize_crop', 0.8)

        data_aug = v2.Compose([
            v2.ToImage(),
            v2.RandomAffine(degrees=rot,
                            translate=(trans, trans),
                            scale=(scale, 1.0),
                            interpolation=InterpolationMode.BILINEAR),
            v2.ToDtype(torch.float32, scale=True),
        ])

        data_aug_test = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])

        dataset = NIH_Dataset(imgpath=imgpath, csvpath=csvpath, bbox_list_path=bboxpath, data_aug=data_aug, transform=None, unique_patients=unique_patients)
        dataset_test = NIH_Dataset(imgpath=imgpath, csvpath=csvpath, bbox_list_path=bboxpath, data_aug=data_aug_test, transform=None, unique_patients=unique_patients)
        seed = config.get('seed', 0) # Its not in config
        gss = GroupShuffleSplit(train_size=0.8,test_size=0.2, random_state=seed)
        train_inds, test_inds = next(gss.split(X=range(len(dataset)), groups=dataset.csv.patientid))
        train = SubsetDataset(dataset, train_inds)
        dev = SubsetDataset(dataset_test, test_inds)
    elif config['data']['name'] == 'xray_g':
        from datasets.torchxrayvision import NIH_Google_Dataset
        from datasets.torchxrayvision import SubsetDataset
        from sklearn.model_selection import GroupShuffleSplit
        num_workers = 1
        imgpath = NIH_XRAY_PATH / "images-224"
        csvpath = NIH_XRAY_PATH / "google2019_nih-chest-xray-labels.csv.gz"
        orig_csvpath = NIH_XRAY_PATH / "Data_Entry_2017_v2020.csv.gz"
        dataset = NIH_Google_Dataset(imgpath=imgpath, csvpath=csvpath, orig_csvpath=orig_csvpath, transform=None)
        seed = config.get('seed', 0) # Its not in config
        gss = GroupShuffleSplit(train_size=0.8,test_size=0.2, random_state=seed)
        train_inds, test_inds = next(gss.split(X=range(len(dataset)), groups=dataset.csv.patientid))
        train = SubsetDataset(dataset, train_inds)
        dev = SubsetDataset(dataset, test_inds)

    else:
        raise ValueError(f"Unknown dataset: {config['data']['name']}")

    # Set additional parameters for the dataloader
    pin_memory = True if torch.cuda.is_available() else False
    persistent_workers = True if num_workers > 0 else False

    # Initialize the sampler
    train_sampler = train_sampler(train)

    trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                        sampler=train_sampler, num_workers=num_workers,
                                        pin_memory=pin_memory, persistent_workers=persistent_workers)
    devloader = torch.utils.data.DataLoader(dev, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers,
                                            pin_memory=pin_memory, persistent_workers=persistent_workers)

    if config['data']['name'] == 'MNIST' and config['data'].get('sampler', 'RandomSampler') == 'MultifidelitySampler':
        trainloader.sampler.set_fidelity(0.33)

    return trainloader, devloader

# In main, we benchmark the dataset creation and the dataloader iteration
if __name__ == '__main__':
    import time
    import numpy as np
    exp = "CIFAR10"

    start1 = None

    if exp == "PTB-XL":
        from datasets.ptbxl import PTBXLDataset
        create = False
        start = time.time()
        train = PTBXLDataset (PTBXL_PATH, train=True, create=create)
        dev = PTBXLDataset(PTBXL_PATH, train=False, create=False)
        stop = time.time()
    elif exp == "PTB-XL-Shared":
        from datasets.ptbxl_shared import PTBXLDatasetShared
        create = True
        start = time.time()
        train = PTBXLDatasetShared (PTBXL_PATH, train=True, create=create)
        dev = PTBXLDatasetShared(PTBXL_PATH, train=False, create=False)
        stop = time.time()

        create = False
        start1 = time.time()
        train = PTBXLDatasetShared (PTBXL_PATH, train=True, create=create)
        dev = PTBXLDatasetShared(PTBXL_PATH, train=False, create=False)
        stop1 = time.time()
    elif exp == "CIFAR10":
        from datasets.cifar10 import CIFAR10
        create = False
        start = time.time()
        cifar10 = CIFAR10(CIFAR10_PATH)
        train = cifar10.dataset('train', cifar10_fn)
        dev = cifar10.dataset('dev', cifar10_test_fn)
        stop = time.time()

    batch_size = 128
    num_workers = 2
    pin_memory = True if torch.cuda.is_available() else False
    persistent_workers = True if num_workers > 0 else False
    create = False

    print(f"Exp {exp}, Using {num_workers} workers, pin_memory: {pin_memory}, persistent_workers: {persistent_workers}")

    if start1 is None:
        print(f"Creating dataset took: {stop - start}")
    else:
        print(f"Creating dataset took: 1. {stop - start} and 2. {stop1 - start1}")


    mf_sampler = MultifidelitySampler(train)
    dev_sampler = MultifidelitySampler(dev)
    trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                    sampler=mf_sampler, num_workers=num_workers,
                                    pin_memory=pin_memory, persistent_workers=persistent_workers)
    devloader = torch.utils.data.DataLoader(dev, batch_size=batch_size,
                                        shuffle=False, num_workers=num_workers,
                                        pin_memory=pin_memory, persistent_workers=persistent_workers)

    start = time.time()
    enum = enumerate(trainloader)
    end = time.time()
    print(f"Creating enum took: {end - start}")

    start = time.time()
    for batch in enum:
        x = np.random.rand(1000, 1000)
        y = np.random.rand(1000, 1000)
        z = np.matmul(x, y)
    end = time.time()
    print(f"Iterating over trainloader took: {end - start}")