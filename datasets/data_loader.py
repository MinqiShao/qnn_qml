import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
from PIL import Image
from torch.utils.data import Dataset, Subset, DataLoader
from tools.feature_reduction import feature_reduc


def load_dataset(name, dir, reduction, resize=False, bi=True, class_idx=[0, 1], scale=1.0):
    if name == 'mnist':
        train_set = datasets.MNIST(dir, train=True, download=True, transform=transforms.ToTensor())
        test_set = datasets.MNIST(dir, train=False, download=True, transform=transforms.ToTensor())
    elif name == 'fashion_mnist':
        train_set = datasets.FashionMNIST(dir, train=True, download=True, transform=transforms.ToTensor())
        test_set = datasets.FashionMNIST(dir, train=False, download=True, transform=transforms.ToTensor())
    elif name == 'emnist':
        train_set = datasets.EMNIST(dir, train=True, download=True, transform=transforms.ToTensor(), split='digits')
        test_set = datasets.EMNIST(dir, train=False, download=True, transform=transforms.ToTensor(), split='digits')

    train_y = train_set.targets.clone()
    test_y = test_set.targets.clone()

    if bi:
        train_idx = torch.where(torch.logical_or(train_y == class_idx[0], train_y == class_idx[1]))[0]
        test_idx = torch.where(torch.logical_or(test_y == class_idx[0], test_y == class_idx[1]))[0]
    else:
        num_class = len(class_idx)
        train_idx = torch.tensor([], dtype=torch.long)
        test_idx = torch.tensor([], dtype=torch.long)
        for i in range(num_class):
            train_idx = torch.cat((train_idx, torch.where(train_y == class_idx[i])[0]))
            test_idx = torch.cat((test_idx, torch.where(test_y == class_idx[i])[0]))

    scaled_train_idx = train_idx[:int(len(train_idx) * scale)]
    scaled_test_idx = test_idx[:int(len(test_idx) * scale)]
    print(f'load {name} data for {len(class_idx)} classification, classes: {class_idx}, scale: {scale * 100}%, '
          f'num of training data: {len(scaled_train_idx)}, num of testing data: {len(scaled_test_idx)}')

    if resize:
        train_set.data, test_set.data = feature_reduc(train_set.data.clone().float() / 255.0,
                                                      test_set.data.clone().float() / 255.0, f_type=reduction)
        print(f'feature reduction {reduction} has completed')

    selected_train = DataLoader(Subset(train_set, train_idx), batch_size=64, shuffle=True)
    selected_test = DataLoader(Subset(test_set, test_idx), batch_size=64)

    return selected_train, selected_test
