import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
from PIL import Image
from torch.utils.data import Dataset, Subset, DataLoader


def load_dataset(name, dir, bi=True, class_idx=[0, 1]):
    if name == 'mnist':
        data_dir = dir + '/mnist'
        train_set = datasets.MNIST(data_dir, train=True, download=True, transform=transforms.ToTensor())
        test_set = datasets.MNIST(data_dir, train=False, download=True, transform=transforms.ToTensor())
    elif name == 'fashion_mnist':
        data_dir = dir + '/fashion_mnist'
        train_set = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transforms.ToTensor())
        test_set = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transforms.ToTensor())
    elif name == 'emnist':
        data_dir = dir + '/emnist'
        train_set = datasets.EMNIST(data_dir, train=True, download=True, transform=transforms.ToTensor(), split='digits')
        test_set = datasets.EMNIST(data_dir, train=False, download=True, transform=transforms.ToTensor(), split='digits')

    if name == 'emnist':
        train_y = torch.tensor(train_set.labels)
        test_y = torch.tensor(test_set.labels)
    else:
        train_y = torch.tensor(train_set.targets)
        test_y = torch.tensor(test_set.targets)

    if bi:
        train_idx = torch.where(torch.logical_or(train_y == class_idx[0], train_y == class_idx[1]))[0]
        test_idx = torch.where(torch.logical_or(test_y == class_idx[0], test_y == class_idx[1]))[0]
    else:
        num_class = len(class_idx)
        train_idx = torch.tensor([])
        test_idx = torch.tensor([])
        for i in range(num_class):
            train_idx = torch.cat(train_idx, torch.where(train_y == class_idx[i])[0])
            test_idx = torch.cat(test_idx, torch.where(test_y == class_idx[i])[0])

    print(f'load {name} data for {len(class_idx)} classification, classes: {class_idx}, '
          f'num of training data: {len(train_idx)}, num of testing data: {len(test_idx)}')

    selected_train = DataLoader(Subset(train_set, train_idx), batch_size=64, shuffle=True)
    selected_test = DataLoader(Subset(test_set, test_idx), batch_size=64)

    return selected_train, selected_test
