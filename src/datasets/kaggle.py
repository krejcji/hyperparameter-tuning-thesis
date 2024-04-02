import os
from pathlib import Path
import itertools
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scipy.io
from torch.utils.data import Dataset
import torch

def process_file_old(data, num_sequences, seq_length):
    """
    Old version, not used anymore.

    Prepared fixed number of sequences from each record. It was memory efficient,
    but most of the data was discarded.
    """
    data = data.to_numpy().flatten()
    # Split the data into PPG, ECG and BP
    ecg = []
    bp = []
    sbp = [] #Systolic Blood Pressure
    dbp = [] #Diastolic Blood Pressue
    ppg = []

    for i in range(data.shape[0]):
        record = data[i]
        record_samples = record.shape[1]
        max_samples = (int)(record_samples/seq_length)
        real_samples = min(num_sequences,max_samples)
        bounds = np.linspace(0 , record_samples-seq_length, real_samples+1, dtype=int)

        for j in range(real_samples):
            ppg.append(record[0, bounds[j]:bounds[j]+seq_length])
            temp_bp = record[1, bounds[j]:bounds[j]+seq_length]
            # ecg.append(record[2, bounds[j]:bounds[j]+seq_length)

            max_value = max(temp_bp)
            min_value = min(temp_bp)

            sbp.append(max_value)
            dbp.append(min_value)
            # ecg.append(temp_ecg)
            # bp.append(temp_bp)

    ppg = np.array(ppg, dtype=np.single).reshape(-1, 1, seq_length)
    #ecg = np.array(ecg).reshape(-1,1)
    #bp = np.array(bp).reshape(-1,1)
    sbp = np.array(sbp,dtype=np.single).reshape(-1)
    dbp = np.array(dbp,dtype=np.single).reshape(-1)
    bps = np.transpose((sbp, dbp), (1, 0))
    return ppg, bps

def process_file(data, seq_length):
    """
    Process the data from a single file of Kaggle dataset.

    Puts ppgs into a list and preprocesses the blood pressure data,
    esimates systolic and diastolic blood pressure for each moving window
    of size seq_length.
    """
    data = data.to_numpy().flatten()
    # Split the data into PPG, ECG and BP
    ecg = []
    bp = []
    sbp = [] #Systolic Blood Pressure
    dbp = [] #Diastolic Blood Pressue
    ppg = []

    for i in range(data.shape[0]):
        ppg.append(data[i][0])
        samples = data[i][1].shape[0]
        systolic = np.zeros(samples-seq_length+1,dtype=np.single)
        diastolic = np.zeros(samples-seq_length+1, dtype=np.single)
        bps = np.lib.stride_tricks.sliding_window_view(data[i][1], window_shape=(seq_length,), axis=0)

        for k in range(samples-seq_length+1):
            systolic[k] = max(bps[k])
            diastolic[k] = min(bps[k])

        sbp.append(systolic)
        dbp.append(diastolic)

    return ppg, sbp, dbp

def store_processed_data(dir, seq_length, files, train_val_split=0.9):
    """
    Serialize the data using pickle.
    """
    if not dir.exists():
        raise FileNotFoundError(f'The directory with Kaggle dataset doesnt exist: {dir}')

    ppgs, syss, dias = [], [], []
    for i in range(1, files+1):
        path = dir / f'part_{i}.mat'
        if not path.exists():
            raise FileNotFoundError(f'The file of Kaggle dataset doesnt exist: {path}')
        df = pd.DataFrame(scipy.io.loadmat(path)['p'])
        ppg, sys, dia = process_file(df, seq_length)
        ppgs.append(ppg)
        syss.append(sys)
        dias.append(dia)

    # Flatten the lists
    ppg = list(itertools.chain.from_iterable(ppgs))
    sys = list(itertools.chain.from_iterable(syss))
    dia = list(itertools.chain.from_iterable(dias))

    dataset_len = len(ppg)
    split_on = int(dataset_len * train_val_split)

    ppg_train = ppg[:split_on]
    bps_train =  [sys[:split_on], dia[:split_on]]

    ppg_val = ppg[split_on:]
    bps_val = [sys[split_on:], dia[split_on:]]
    # print(f'Dataset loaded, Train: {train}, PPG_shape: {ppg.shape}, bps_shape: {bps.shape}')

    if not os.path.exists(dir / "serialized"):
        os.makedirs(dir / "serialized")

    with open(dir / "serialized" / "bps_train", "wb") as f:
        pickle.dump(bps_train, f)
    with open(dir / "serialized" / "bps_val", "wb") as f:
        pickle.dump(bps_val, f)

    with open(dir / "serialized" / "ppg_train", "wb") as f:
        pickle.dump(ppg_train, f)
    with open(dir / "serialized" / "ppg_val", "wb") as f:
        pickle.dump(ppg_val, f)

def load_kaggle(path, train):
    if train:
        with open(path / "serialized" / "bps_train", "rb") as f:
            bps = pickle.load(f)
        with open(path / "serialized" / "ppg_train", "rb") as f:
            ppg = pickle.load(f)
    else:
        with open(path / "serialized" / "bps_val", "rb") as f:
            bps = pickle.load(f)
        with open(path / "serialized" / "ppg_val", "rb") as f:
            ppg = pickle.load(f)

    return ppg, bps

class KaggleDataset(Dataset):
    def __init__(self, dir, train=True, size=250, transform=None):
        self.transform = transform
        self.seq_length = size
        self.ppg, self.bp = load_kaggle(dir, train)
    def __len__(self):
        return len(self.bp[0])
    def __getitem__(self, index):
        samples = self.ppg[index].shape[0]
        rnd = np.random.randint(0, samples-self.seq_length)
        ppg = self.ppg[index][rnd:rnd+self.seq_length]
        sys = self.bp[0][index][rnd]
        dia = self.bp[1][index][rnd]
        return (torch.from_numpy(ppg.reshape(1,-1)), torch.tensor([sys, dia]))

if __name__== "__main__":
    dir = Path('data') / 'kaggle_ppg_bp'
    store_processed_data(dir, 250, 12)
    #dataset = KaggleDataset(dir)
    #print(dataset.__getitem__(0))
    #print()