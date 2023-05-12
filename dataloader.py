# a dataloader for PyTorch:
#
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.data = torch.from_numpy(np.load(path)).float()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

# for a cycleGAN, we need to load in two datasets, one for each domain
# we'll use the noisy data as domain A and the denoised data as domain B
# we'll use the same train/test split for both domains
domain_A_dataset = MyDataset('noisy_data.npy')
domain_B_dataset = MyDataset('denoised_data.npy')

# define a train/test split
train_test_split = 0.8

# define split size for domain_A
domain_A_train_size = int(train_test_split * len(domain_A_dataset))
domain_A_test_size = len(domain_A_dataset) - domain_A_train_size

# define split size for domain_B
domain_B_train_size = int(train_test_split * len(domain_B_dataset))
domain_B_test_size = len(domain_B_dataset) - domain_B_train_size

# split the datasets randomly
domain_A_train_dataset, domain_A_test_dataset = torch.utils.data.random_split(domain_A_dataset, [domain_A_train_size, domain_A_test_size])
domain_B_train_dataset, domain_B_test_dataset = torch.utils.data.random_split(domain_B_dataset, [domain_B_train_size, domain_B_test_size])

BATCH_SIZE = 10

# load the datasets into dataloaders
domain_A_train_dataloader = torch.utils.data.DataLoader(domain_A_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
domain_A_test_dataloader = torch.utils.data.DataLoader(domain_A_test_dataset, batch_size=BATCH_SIZE, shuffle=True)
domain_B_train_dataloader = torch.utils.data.DataLoader(domain_B_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
domain_B_test_dataloader = torch.utils.data.DataLoader(domain_B_test_dataset, batch_size=BATCH_SIZE, shuffle=True)
