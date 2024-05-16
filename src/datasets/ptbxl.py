from pathlib import Path
import os

import pandas as pd
import numpy as np
import wfdb
import ast
import torch
from torch.utils.data import Dataset


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path / f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path / f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

def encode_labels(series, train=True):
    """
    One-hot encode a Series of lists of strings (e.g. diagnostic superclass labels)
    """
    diag_superclass = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    # Get the original length of the Series
    original_length = len(series)
    val_true = 0.9 if train else 1.0
    val_false = 0.1 if train else 0.0

    # Create the DataFrame with original length
    df = pd.DataFrame(columns=diag_superclass, data=np.full((original_length, len(diag_superclass)), val_false, dtype=np.float32))

    # Iterate over the original Series using explicit index access
    for i in range(original_length):
        values = series.iloc[i]
        for string in values:
            df.loc[i, string] = val_true

    return df

def store_processed_data(path):
    if not os.path.exists(path / "serialized"):
        os.makedirs(path / "serialized")
    sampling_rate=100

    # Load and convert annotation data
    Y = pd.read_csv(path/'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load raw signal data
    X = load_raw_data(Y, sampling_rate, path)

    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv(path/'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    # Apply diagnostic superclass
    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)
    Y['lengths'] = Y['diagnostic_superclass'].apply(lambda x: len(x))
    # Y = Y[Y.lengths > 0] # Remove samples without diagnostic superclass

    # Split data into train and test (validation)
    test_fold = 10

    X_train = X[np.where((Y.strat_fold != test_fold) & (Y.lengths > 0))]
    X_train = np.swapaxes(X_train, 1,2).astype(np.float32)

    y_train = Y[(Y.strat_fold != test_fold) & (Y.lengths > 0)].diagnostic_superclass
    y_train = encode_labels(y_train).to_numpy()

    X_test = X[np.where((Y.strat_fold == test_fold) & (Y.lengths > 0))]
    X_test = np.swapaxes(X_test, 1,2).astype(np.float32)

    y_test = Y[(Y.strat_fold == test_fold) & (Y.lengths > 0)].diagnostic_superclass
    y_test = encode_labels(y_test, train=False).to_numpy()

    #
    #y_test = y_test.apply(lambda x: x[0] if len(x) > 0 else 'EMPTY').factorize()[0]
    #y_test = y_test.astype(np.int64)

    torch.save(X_train, path / "serialized"/"X_train.pt")
    torch.save(y_train, path / "serialized"/"y_train.pt")
    torch.save(X_test, path / "serialized"/"X_test.pt")
    torch.save(y_test, path / "serialized"/"y_test.pt")

def load_dataset(path, train, random_samples=False):
    if train:
        X = torch.load(path / "serialized"/"X_train.pt")
        y = torch.load(path / "serialized"/"y_train.pt")
    else:
        X = torch.load(path / "serialized"/"X_test.pt")
        y = torch.load(path / "serialized"/"y_test.pt")

    return X, y

class PTBXLDataset(Dataset):
    def __init__(self, dir, train=True, transform=None, seq_length=256):
        self.transform = transform
        self.seq_length = seq_length
        self.X, self.y = load_dataset(dir, train, random_samples=True)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, index):
        sample_len = self.X[index].shape[1]
        # Sample fixed-length window at random location in the sequence
        rnd = np.random.randint(0, sample_len-self.seq_length)

        return (self.X[index,:,rnd:rnd+self.seq_length], self.y[index])

if __name__== "__main__":
    dir = Path('data') / 'ptbxl'
    store_processed_data(dir)
    dataset = PTBXLDataset(dir,train=True)
    print(dataset.X.shape)
    print(dataset.y.shape)
    print("done.")
