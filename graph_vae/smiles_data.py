
import copy

import numpy as np
from torch.utils import data


class SmilesDataset(data.Dataset):
    def __init__(self, filename, transforms=None):
        with open(filename, 'r') as fo:
            data = [x.strip() for x in fo.readlines()]
        self.data = data

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        if self.transforms is not None:
            item = self.transforms(item)
        return item

    def split_dataset(self, train_indcs, valid_indcs, test_indcs):
        def cpy_and_set(line_indcs):
            new_lines = [self.data[i] for i in line_indcs]
            new_ds = copy.copy(self)
            new_ds.data = new_lines
            return new_ds

        new_ds_train = cpy_and_set(train_indcs)
        new_ds_valid = cpy_and_set(valid_indcs)
        new_ds_test = cpy_and_set(test_indcs)

        return new_ds_train, new_ds_valid, new_ds_test

