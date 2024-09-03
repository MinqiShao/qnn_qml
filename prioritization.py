"""
test prioritization
"""
import numpy as np
import torch
import os

from sklearn.metrics import accuracy_score
from tqdm import tqdm

from config import *
from tools.data_loader import load_part_data, load_adv_imgs
from tools.model_loader import load_params_from_path
from models.circuits import qubit_dict, depth_dict
from Coverage import CoverageHandler

conf = get_arguments()
device = torch.device('cpu')
depth = depth_dict[conf.structure]


def pre_data(model, x, y):
    model.eval()
    y_preds = model.predict(x).detach()
    acc = accuracy_score(y.numpy().tolist(), y_preds.numpy().argmax(axis=1).tolist())
    return acc, y_preds


def apfd(y, y_preds):
    y_preds = y_preds.argmax(axis=1)
    f = (y != y_preds).int()
    o_indices = torch.where(f == 1)[0]+1  # 发生错误的位置
    k = torch.sum(f).item()  # k 预测错误的总数
    n = y.shape[0]
    sum_o = o_indices.sum().item()
    return 1 - (sum_o / (k*n)) + 1/(2*n)


def deepgini(test_x, test_y, budget=0.1):
    budget_num = int(budget*test_x.shape[0])
    _, model = load_params_from_path(conf, device)
    _, y_pred_prob = pre_data(model, test_x, test_y)
    metrics = torch.sum(y_pred_prob**2, dim=1)
    rank_lst = torch.argsort(metrics)[:budget_num]
    return rank_lst


def exp(m='ksc', b=0.1):
    test_x, test_y = load_part_data(conf, num_data=conf.num_test)
    if conf.with_adv:
        _, adv_imgs, ori_y = load_adv_imgs(conf)
        adv_imgs = adv_imgs.squeeze(1)
        test_x = torch.cat((test_x, adv_imgs), dim=0)
        test_y = torch.cat((test_y, ori_y))
    indices = torch.randperm(len(test_x))
    test_x = test_x[indices]
    test_y = test_y[indices]

    model_n = 'hier_' + conf.hier_u if conf.structure == 'hier' else conf.structure
    m_params, model = load_params_from_path(conf, device)
    log_dir = os.path.join(conf.analysis_dir, conf.dataset, model_n)
    profile_path = os.path.join(log_dir, conf.cov_cri + '_range_' + str(conf.class_idx) + '_' + str(conf.num_train) + '.pth')
    state_num = 2**qubit_dict[conf.structure]

    if m == 'deepgini':
        subset = deepgini(test_x, test_y, budget=b)
    else:
        handler = CoverageHandler(model, m_params, state_num, profile_path, cri=conf.cri)
        subset, _ = handler.rank(test_x, conf, depth, budget=b)

    acc, preds = pre_data(model, test_x[subset], test_y[subset])

    print(f'criteria {m}, budget: {b}, subset_acc: {acc*100}%')
    return test_y[subset], preds


if __name__ == '__main__':
    # m = 'ksc'  # ksc scc tsc kec deepgini
    b = 1.0  # budget
    gt, preds = exp(conf.cri, b)
    a = apfd(gt, preds)
    print(f'APFD: {a}')
