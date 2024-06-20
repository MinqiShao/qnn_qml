import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, Subset, DataLoader
from tools.feature_reduction import feature_reduc, feature_redc_test
import torchvision.transforms.functional as F


def load_dataset(name, dir, reduction, resize=False, bi=True, class_idx=[0, 1], scale=1.0):
    if name == 'mnist':
        train_set = datasets.MNIST(dir, train=True, download=True, transform=transforms.ToTensor())
        test_set = datasets.MNIST(dir, train=False, download=True, transform=transforms.ToTensor())
    elif name == 'fashion_mnist':
        train_set = datasets.FashionMNIST(dir, train=True, download=True, transform=None if resize else transforms.ToTensor())
        test_set = datasets.FashionMNIST(dir, train=False, download=True, transform=None if resize else transforms.ToTensor())
    elif name == 'emnist':
        train_set = datasets.EMNIST(dir, train=True, download=True, transform=None if resize else transforms.ToTensor(), split='digits')
        test_set = datasets.EMNIST(dir, train=False, download=True, transform=None if resize else transforms.ToTensor(), split='digits')

    scaled_train_idx = filter_data(train_set, bi, class_idx, scale)
    scaled_test_idx = filter_data(test_set, bi, class_idx, scale)
    print(f'load {name} data for {len(class_idx)} classification, classes: {class_idx}, scale: {scale * 100}%, '
          f'num of training data: {len(scaled_train_idx)}, num of testing data: {len(scaled_test_idx)}')

    if resize:
        train_set.data, test_set.data = feature_reduc(train_set.data.clone().float(),
                                                      test_set.data.clone().float(), f_type=reduction)
        print(f'feature reduction {reduction} has completed')

    train_set.targets = transform_labels(train_set.targets.clone(), class_idx)
    test_set.targets = transform_labels(test_set.targets.clone(), class_idx)

    selected_train = Subset(train_set, scaled_train_idx)
    selected_test = Subset(test_set, scaled_test_idx)

    return DataLoader(selected_train, batch_size=64, shuffle=True), DataLoader(selected_test, batch_size=64), \
    train_set.data[scaled_train_idx], train_set.targets[scaled_train_idx], test_set.data[scaled_test_idx], \
    test_set.targets[scaled_test_idx]


def transform_labels(y, class_idx=[0, 1]):
    for idx, i in enumerate(class_idx):
        a = torch.where(y == i)[0]
        y[a] = idx

    return y


def filter_data(data_set, bi=True, class_idx=[0, 1], scale=1.0):
    y = data_set.targets.clone()

    if bi:
        idx = torch.where(torch.logical_or(y == class_idx[0], y == class_idx[1]))[0]
    else:
        num_class = len(class_idx)
        idx = torch.tensor([], dtype=torch.long)
        for i in range(num_class):
            idx = torch.cat((idx, torch.where(y == class_idx[i])[0]))

    scaled_idx = idx[:int(len(idx) * scale)]

    return scaled_idx


def load_test_data(conf, bi=True):
    name = conf.dataset
    dir = conf.data_dir
    resize = conf.resize
    class_idx = conf.class_idx
    scale = conf.data_scale
    if name == 'mnist':
        test_set = datasets.MNIST(dir, train=False, download=True)
    elif name == 'fashion_mnist':
        test_set = datasets.FashionMNIST(dir, train=False, download=True)
    elif name == 'emnist':
        test_set = datasets.EMNIST(dir, train=False, download=True, split='digits')

    scaled_test_idx = filter_data(test_set, bi, class_idx, scale)
    print(f'load {name} data for {len(class_idx)} classification, classes: {class_idx}, scale: {scale * 100}%, '
          f'num of testing data: {len(scaled_test_idx)}')
    if resize:
        test_set.data = feature_redc_test(test_set.data.clone().float() / 255.0, f_type=conf.reduction)
        print(f'feature reduction {conf.reduction} has completed')
    else:
        test_set.data = test_set.data.clone().float() / 255.0

    test_set.targets = transform_labels(test_set.targets.clone(), class_idx)

    return test_set.data[scaled_test_idx], test_set.targets[scaled_test_idx]
