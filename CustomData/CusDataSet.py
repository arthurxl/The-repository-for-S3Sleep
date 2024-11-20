import numpy as np
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data, labels, seq_len=20):
        super(CustomDataset, self).__init__()
        self.x_list = []
        self.y_list = []
        for i in range(len(data)):
            data_len = data[i].shape[0]
            num_elems = (data_len // seq_len) * seq_len

            self.x_list.append(data[i][:num_elems])
            self.y_list.append(labels[i][:num_elems])

        self.x_list = [np.split(x, x.shape[0] // seq_len) for x in self.x_list]
        self.y_list = [np.split(y, y.shape[0] // seq_len) for y in self.y_list]

        self.x_list = [item for sublist in self.x_list for item in sublist]
        self.y_list = [item for sublist in self.y_list for item in sublist]

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, idx):
        return (torch.FloatTensor(self.x_list[idx]), torch.LongTensor(self.y_list[idx]))
