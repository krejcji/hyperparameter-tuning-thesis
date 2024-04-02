import torch
from torch.utils.data import Sampler

class MultifidelitySampler(Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)
        self.set_fidelity(1)
        self.rand_subset = torch.randperm(self.num_samples)

    def set_fidelity(self, fidelity):
        if not 0 <= fidelity <= 1:
            raise ValueError("Fidelity must be between 0 and 1")
        self.num_subset_samples = int(fidelity * self.num_samples)

    def __iter__(self):
        shuffled_indices = torch.randperm(self.num_subset_samples)
        shuffled_indices = [self.rand_subset[i] for i in shuffled_indices]
        return iter(shuffled_indices)

    def __len__(self):
        return self.num_subset_samples
