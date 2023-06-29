from torchvision import datasets
import torch
import os
from torch.utils.data import DataLoader

def read_data(data_dir, batch_size):
    train_data_dir = f'{data_dir}/train'
    val_data_dir = f'{data_dir}/val'
    # test_data_dir = f'{data_dir}/test'
    # 这里可自行修改数据预处理，batch大小也可自行调整
    # 保持本地训练的数据读取和这里一致
    train_dataset = datasets.ImageFolder(train_data_dir)
    val_dataset = datasets.ImageFolder(val_data_dir)
    data_loader_train = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                                    num_workers=os.cpu_count())
    data_loader_val = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True,
                                                  drop_last=True, num_workers=os.cpu_count())
    return train_dataset, val_dataset, data_loader_train, data_loader_val