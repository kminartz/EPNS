import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def get_data_loaders(config, additional_loaders=('test_same_init',), limit_num_data_points_to=np.inf):
    data_directory, batch_size = config['data_directory'], config['batch_size'],

    dir_train = os.path.join(data_directory, 'train')
    dir_val = os.path.join(data_directory, 'val')
    dir_test = os.path.join(data_directory, 'test')

    example = np.load(os.path.join(dir_train, os.listdir(dir_train)[0]))

    one_file_shape = example.shape

    dataset = CellCombinedDataset(dir_train, is_test=False, num_samples=limit_num_data_points_to)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    val_dataset = CellCombinedDataset(dir_val, is_test=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = CellCombinedDataset(dir_test, is_test=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print('shape of first datapoint:', one_file_shape, '. Number of train datapoints:', len(dataset.fnames),
          'number of val datapoints:', len(val_dataset.fnames), 'number of test datapoints:',
          len(test_dataset.fnames)
          )

    other_loaders = []
    for dirpath in additional_loaders:
        # get additional loaders to subdirectories with these names: (e.g. test_same_init)
        add_dataset = CellCombinedDataset(os.path.join(data_directory, dirpath), is_test=True)
        add_loader = torch.utils.data.DataLoader(add_dataset, batch_size=batch_size, shuffle=True)
        other_loaders.append(add_loader)

    return dataloader, val_dataloader, test_dataloader, *other_loaders


class CellCombinedDataset(Dataset):
    def __init__(self, dir_path, is_test=False, num_samples=np.inf):

        self.dir = dir_path
        all_fnames = os.listdir(dir_path)
        num_samples = min(num_samples, len(all_fnames))
        self.fnames = all_fnames[:num_samples]
        self.samples_per_epoch_multiplier = len(all_fnames) / len(self.fnames)
        example = np.load(os.path.join(dir_path, self.fnames[0]))
        self.num_timesteps = example.shape[0]
        self.is_test = is_test

    def __getitem__(self, idx):

        idx = idx % len(self.fnames)

        datapoint_temp = torch.Tensor(np.load(os.path.join(self.dir, self.fnames[idx])))  # (t, h, w, 2)
        datapoint = torch.movedim(datapoint_temp, source=(0, 1, 2, 3), destination=(1, 2, 3, 0)) # (2, t, h, w,)
        datapoint = datapoint[:, 1:]  # remove first timestamp as it is heavily out of domain compared to the rest
        # id_to_type_dict = self.get_id_to_type_dict(datapoint[0], datapoint[1])
        return datapoint  # return only id channel

    def __len__(self):
        if not self.is_test:
            return int(self.samples_per_epoch_multiplier * len(self.fnames))
        return int(1 * len(self.fnames))


