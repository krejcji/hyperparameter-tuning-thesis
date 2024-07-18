# The dataset PTB-XL source: https://physionet.org/content/ptb-xl/1.0.3/
# License for the files: Creative Commons Attribution 4.0 International Public License
from pathlib import Path
import os
import multiprocessing
import multiprocessing.shared_memory
from multiprocessing import resource_tracker

import numpy as np
import torch
from torch.utils.data import Dataset

def load_dataset(path, train, random_samples=False):
    if train:
        X = torch.load(path / "serialized"/"X_train.pt")
        y = torch.load(path / "serialized"/"y_train.pt")
    else:
        X = torch.load(path / "serialized"/"X_test.pt")
        y = torch.load(path / "serialized"/"y_test.pt")

    return X, y

def create_shared_memory(self, path, id, random_samples=False):
    X_train, y_train = load_dataset(path, True, random_samples)
    X_test, y_test = load_dataset(path, False, random_samples)

    arrays = {'X_train': X_train,
              'y_train': y_train,
              'X_test': X_test,
              'y_test': y_test}

    for arr_name in arrays.keys():
        name = f'{arr_name}_{id}'
        array = arrays[arr_name]
        shm_arr = multiprocessing.shared_memory.SharedMemory(create=True, size=array.nbytes, name=name)
        shared = np.ndarray(array.shape, dtype=array.dtype, buffer=shm_arr.buf)
        print(f'Created shared memory for {name} with shape {shared.shape} and dtype {shared.dtype}, on {array.nbytes} bytes.')
        # Do not unregister the tracker here, so the unlink is always called on exit.
        shared[:] = array
        setattr(self, name, shm_arr)
        shm_arr = None

class PTBXLDatasetShared(Dataset):
    def __init__(self, dir, train=True, transform=None, seq_length=256, create=False, id=0):
        train_size = 19230
        test_size = 2158
        self.transform = transform
        self.seq_length = seq_length
        self.create = create
        self.unique_id = id

        # Create should be called before the training to allocate shared memory and hold references
        if create:
            if train:
                create_shared_memory(self, dir, id)
            self.X = np.zeros((1,1,1))
            self.y = np.zeros((1,1))
        else:
            if train:
                mem_X_train = multiprocessing.shared_memory.SharedMemory(name=f'X_train_{id}')
                mem_y_train = multiprocessing.shared_memory.SharedMemory(name=f'y_train_{id}')
                if os.name == 'posix':
                    # Needed to avoid deleting the shared memory prematurely on posix
                    resource_tracker.unregister(mem_X_train._name, 'shared_memory')
                    resource_tracker.unregister(mem_y_train._name, 'shared_memory')
                self.X = np.ndarray((train_size,12,1000), dtype=np.float32, buffer=mem_X_train.buf)
                self.y = np.ndarray((train_size, 5), dtype=np.float32, buffer=mem_y_train.buf)
                self.buf_x = mem_X_train
                self.buf_y = mem_y_train
            else:
                mem_X_test = multiprocessing.shared_memory.SharedMemory(name=f'X_test_{id}')
                mem_y_test = multiprocessing.shared_memory.SharedMemory(name=f'y_test_{id}')
                print(f"Mem: {mem_X_test.name}")
                if os.name == 'posix':
                    resource_tracker.unregister(mem_X_test._name, 'shared_memory')
                    resource_tracker.unregister(mem_y_test._name, 'shared_memory')
                self.X = np.ndarray((test_size,12,1000), dtype=np.float32, buffer=mem_X_test.buf)
                self.y = np.ndarray((test_size, 5), dtype=np.float32, buffer=mem_y_test.buf)
                self.buf_x = mem_X_test
                self.buf_y = mem_y_test
        print(f"Initialized PTBXLDataset with train={train}, transform={transform}, seq_length={seq_length}, create={create}")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        sample_len = self.X[index].shape[1]
        rnd = np.random.randint(0, sample_len-self.seq_length)
        X = self.X[index,:,rnd:rnd+self.seq_length]
        y = self.y[index]
        return (X, y)

    def close(self):
        print("Closing shared memory")
        if not self.create:
            self.buf_x.close()
            self.buf_y.close()
        else:
            print("Unlinking shared memory")
            for name in ['X_train', 'y_train', 'X_test', 'y_test']:
                mem = multiprocessing.shared_memory.SharedMemory(name=f'{name}_{self.unique_id}')
                mem.unlink()

if __name__== "__main__":
    dir = Path('data') / 'ptbxl'
    #store_processed_data(dir)
    print("Create dataset")
    dataset = PTBXLDatasetShared(dir,train=False, create=True)
    print("Make dataset")
    data = PTBXLDatasetShared(dir,train=True)
    trainloader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True, num_workers=1)

    if hasattr(trainloader.dataset, 'close'):
        trainloader.dataset.close()

    print("Iterating over trainloader")
    #for batch in trainloader:
    #    print(batch[0].shape)
    #    print(batch[1].shape)
    print(f"Length: {len(data)}")
    for i in range(len(data)):
        x, y = data[i]
        print(x[0].shape)
    print("Iterating over trainloader")
    for batch in trainloader:
        print(len(batch))
        print(batch[0].shape)

    print(data.X.shape)
    print(data.y.shape)
    print("done.")
