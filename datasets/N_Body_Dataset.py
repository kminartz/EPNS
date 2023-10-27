import os
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset

def get_data_loaders(config, additional_loaders=('test_same_init', 'test_long_simulation'),
                     limit_num_data_points_to=np.inf, expand_train_set=True):
    data_directory, batch_size = config['data_directory'], config['batch_size']
    dir_train = os.path.join(data_directory, 'train')
    dir_val = os.path.join(data_directory, 'val')
    dir_test = os.path.join(data_directory, 'test')

    dataset = NBodyDataset(dir_train, is_test=not expand_train_set, num_samples=limit_num_data_points_to)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    val_dataset = NBodyDataset(dir_val, is_test=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = NBodyDataset(dir_test, is_test=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print('Number of train datapoints:', len(dataset.fnames),
          'number of val datapoints:', len(val_dataset.fnames), 'number of test datapoints:',
          len(test_dataset.fnames)
          )

    other_loaders = []
    for dirpath in additional_loaders:
        add_dataset = NBodyDataset(os.path.join(data_directory, dirpath), is_test=True)
        add_loader = torch.utils.data.DataLoader(add_dataset, batch_size=batch_size, shuffle=True)
        other_loaders.append(add_loader)

    return dataloader, val_dataloader, test_dataloader, *other_loaders


class NBodyDataset(Dataset):

    def __init__(self, dir_path, is_test=False, num_samples=np.inf):

        warnings.warn('WARNING: we artificially increase the size of the dataset to slow down the pfw schedule!')
        self.multiplier = 200

        self.dir = dir_path
        all_fnames = os.listdir(dir_path)
        num_samples = min(num_samples, len(all_fnames))
        self.fnames = all_fnames[:num_samples]
        example = np.load(os.path.join(dir_path, self.fnames[0]))
        # shape: (num_bodies, attributes, time), but we permute the dims in getitem to (time, num_bodies, attributes)
        self.num_timesteps = example.shape[2]
        self.is_test = is_test
        self.samples_per_epoch_multiplier = len(all_fnames) / len(self.fnames)


    def __getitem__(self, idx):
        if not self.is_test:
            idx = idx % len(self.fnames)

        datapoint: torch.Tensor = torch.Tensor(np.load(os.path.join(self.dir, self.fnames[idx])))
        # shape: (num_bodies, attributes, time) ->  num_bodies, time, attributes
        datapoint = torch.permute(datapoint, (0, 2, 1))

        return datapoint

    def __len__(self):
        if not self.is_test:
            # slow down multi-step rolllout schedule by having more samples per epoch
            return int(self.samples_per_epoch_multiplier * self.multiplier * len(self.fnames) )
        return len(self.fnames)

