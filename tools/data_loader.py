import os
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, Subset, DataLoader
from tools.feature_reduction import feature_reduc, feature_redc_test
import torchvision.transforms.functional as F


def load_dataset(name, dir, reduction, resize=False, class_idx=[0, 1], scale=1.0):
    if name == 'mnist':
        train_set = datasets.MNIST(dir, train=True, download=True, transform=transforms.ToTensor())
        test_set = datasets.MNIST(dir, train=False, download=True, transform=transforms.ToTensor())
    elif name == 'fashion_mnist':
        train_set = datasets.FashionMNIST(dir, train=True, download=True, transform=transforms.ToTensor())
        test_set = datasets.FashionMNIST(dir, train=False, download=True, transform=transforms.ToTensor())
    elif name == 'emnist':
        train_set = datasets.EMNIST(dir, train=True, download=True, transform=transforms.ToTensor(), split='digits')
        test_set = datasets.EMNIST(dir, train=False, download=True, transform=transforms.ToTensor(), split='digits')

    scaled_train_idx = filter_data(train_set, class_idx, scale)
    scaled_test_idx = filter_data(test_set, class_idx, scale)
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

    return DataLoader(selected_train, batch_size=64, shuffle=True), DataLoader(selected_test, batch_size=64)


def transform_labels(y, class_idx=[0, 1]):
    for idx, i in enumerate(class_idx):
        a = torch.where(y == i)[0]
        y[a] = idx

    return y


def filter_data(data_set, class_idx=[0, 1], scale=1.0):
    y = data_set.targets.clone()

    num_class = len(class_idx)
    idx = torch.tensor([], dtype=torch.long)
    for i in range(num_class):
        c_idx = torch.where(y == class_idx[i])[0]
        if type(scale) is int:
            c_idx = c_idx[:scale]
        else:
            c_idx = c_idx[:int(len(c_idx) * scale)]
        print(f'{len(c_idx)} data from class {class_idx[i]}')
        idx = torch.cat((idx, c_idx))

    return idx

def sample_data(x, y, class_idx=[0, 1], scale=0.1):
    num_class = len(class_idx)
    idx = torch.tensor([], dtype=torch.long)
    for i in range(num_class):
        c_idx = torch.where(y == class_idx[i])[0]
        if type(scale) is int:
            c_idx = c_idx[:scale]
        else:
            c_idx = c_idx[:int(len(c_idx) * scale)]
        print(f'{len(c_idx)} data from class {class_idx[i]}')
        idx = torch.cat((idx, c_idx))
    return x[idx], y[idx]


def load_part_data(conf, train_=False, num_data=100):
    name = conf.dataset
    dir = conf.data_dir
    resize = conf.resize
    class_idx = conf.class_idx
    # scale = conf.data_scale
    if name == 'mnist':
        d_set = datasets.MNIST(dir, train=train_, download=True)
    elif name == 'fashion_mnist':
        d_set = datasets.FashionMNIST(dir, train=train_, download=True)
    elif name == 'emnist':
        d_set = datasets.EMNIST(dir, train=train_, download=True, split='digits')

    scaled_idx = filter_data(d_set, class_idx, num_data)
    n = 'train' if train_ else 'test'
    print(f'load {name} {n} data for {len(class_idx)} classification, classes: {class_idx}, '
          f'num data of each class: {num_data}')
    if resize:
        d_set.data = feature_redc_test(d_set.data.clone().float() / 255.0, f_type=conf.reduction)
        print(f'feature reduction {conf.reduction} has completed')
    else:
        d_set.data = d_set.data.clone().float() / 255.0

    d_set.targets = transform_labels(d_set.targets.clone(), class_idx)

    return d_set.data[scaled_idx], d_set.targets[scaled_idx]


def load_adv_imgs(conf):
    model_n = conf.structure
    if conf.structure == 'hier':
        model_n += '_' + conf.hier_u
    if conf.attack == 'QuanTest':
        p = os.path.join(conf.analysis_dir, 'QuanTest', conf.dataset, model_n, str(conf.class_idx))
    else:
        p = os.path.join(conf.analysis_dir, 'AdvAttack', conf.dataset, model_n, str(conf.class_idx), conf.attack)
    idx_list = []
    img_list = []
    transform = transforms.Compose([transforms.ToTensor()])
    for file in os.listdir(p):
        if file.endswith('.png'):
            idx = int(file.split('_')[0])
            idx_list.append(idx)  # img idx
            img_p = os.path.join(p, file)
            img = Image.open(img_p).convert('L')
            img = transform(img)
            img_list.append(img)
    print(f'load {len(img_list)} adv images from {p}')
    return torch.tensor(idx_list), torch.stack(img_list)


def load_correct_data(conf, model, num_data=100):
    # 挑选被模型预测正确的样本
    name = conf.dataset
    dir = conf.data_dir
    class_idx = conf.class_idx
    if name == 'mnist':
        d_set = datasets.MNIST(dir, train=False, download=True)
    elif name == 'fashion_mnist':
        d_set = datasets.FashionMNIST(dir, train=False, download=True)

    scaled_test_idx = filter_data(d_set, class_idx, 0.2)
    d_set.targets = transform_labels(d_set.targets.clone(), class_idx)

    x = d_set.data[scaled_test_idx].clone().float() / 255.0
    y = d_set.targets[scaled_test_idx].clone()

    y_preds = model.predict(x).argmax(1)
    correct_idx = torch.where(y == y_preds)[0]
    x, y = x[correct_idx], y[correct_idx]

    return sample_data(x, y, class_idx, scale=num_data)
