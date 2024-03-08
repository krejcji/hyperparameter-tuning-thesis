from pathlib import Path
import torch
import torch.utils

def load_data(config):
    batch_size = config['data']['batch_size']
    shuffle = config['data']['shuffle']
    pin_memory = True if torch.cuda.is_available() else False
    num_workers = 2

    if config['data']['name'] == 'PTB-XL':
        import load_ptbxl

        trainset = load_ptbxl.PTBXLDataset (Path('data') / 'ptbxl', train=True)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=shuffle, num_workers=num_workers,
                                            pin_memory=pin_memory)

        testset = load_ptbxl.PTBXLDataset(Path('data') / 'ptbxl', train=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers,
                                            pin_memory=pin_memory)
    elif config['data']['name'] == 'Kaggle_PPG':
        import load_kaggle

        trainset = load_kaggle.KaggleDataset(Path('data') / 'kaggle_ppg_bp', train=True)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=shuffle, num_workers=num_workers,
                                            pin_memory=pin_memory)

        testset = load_kaggle.KaggleDataset(Path('data') / 'kaggle_ppg_bp', train=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers,
                                            pin_memory=pin_memory)
    else:
        raise ValueError(f"Unknown dataset: {config['data']['name']}")

    return trainloader, testloader
