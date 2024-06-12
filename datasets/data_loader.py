import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
from PIL import Image
from torch.utils.data import Dataset, Subset, DataLoader


def get_transform(resize=True, to_size=14):
    if resize:
        transforms_ = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((to_size, to_size)),
        ])
    else:
        transforms_ = transforms.ToTensor()
    return transforms_


def load_dataset(name, dir, resize=False, bi=True, class_idx=[0, 1], scale=1.0):
    transform_ = get_transform(resize, 14)
    if name == 'mnist':
        train_set = datasets.MNIST(dir, train=True, download=True, transform=transform_)
        test_set = datasets.MNIST(dir, train=False, download=True, transform=transform_)
    elif name == 'fashion_mnist':
        train_set = datasets.FashionMNIST(dir, train=True, download=True, transform=transform_)
        test_set = datasets.FashionMNIST(dir, train=False, download=True, transform=transform_)
    elif name == 'emnist':
        train_set = datasets.EMNIST(dir, train=True, download=True, transform=transform_, split='digits')
        test_set = datasets.EMNIST(dir, train=False, download=True, transform=transform_, split='digits')

    train_y = torch.tensor(train_set.targets)
    test_y = torch.tensor(test_set.targets)

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

    selected_train = DataLoader(Subset(train_set, train_idx), batch_size=64, shuffle=True)
    selected_test = DataLoader(Subset(test_set, test_idx), batch_size=64)

    return selected_train, selected_test
