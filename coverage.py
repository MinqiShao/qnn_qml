"""
test the bucket coverage (probability, entanglement) within and between classes
the internal statistics (whole ent -> run.py)
"""
from tqdm import tqdm

from config import *
from tools.model_loader import load_params_from_path
from tools.data_loader import load_part_data
from tools import Log
from tools.internal import block_prob, kernel_prob, block_exp, kernel_exp
from tools.gragh import dot_graph
from models.circuits import block_dict, qubit_dict
import torch
import os
from datetime import datetime

conf = get_arguments()
device = torch.device('cpu')

n_qubits = qubit_dict[conf.structure]
k = 100  # bucket num
cir = conf.coverage_cri
if cir == 'prob':
    state_num = 2 ** n_qubits
else:
    state_num = n_qubits

log_dir = os.path.join(conf.analysis_dir, conf.dataset, conf.structure)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_path = os.path.join(log_dir, 'log' + str(conf.class_idx) + '.txt')
log = Log(log_path)


def train_range_circuit(train_x, params):
    """
    get min and max boundary of bucket from train data (circuit structure)
    :param train_x:
    :param params:
    :return: save min/max of each state of each block [{'min_l': [state_num], 'max_l': [state_num]}, ...]
    """
    results = []
    save_path = os.path.join(log_dir, 'range_' + str(conf.class_idx) + '.pth')

    for d in block_dict[conf.structure]:
        min_list = torch.ones((state_num, ))  # min/max prob for each state
        max_list = torch.zeros((state_num, ))
        # covered = torch.zeros((state_num, k))  # for topk

        log(f'Block {d}')
        for i, x in enumerate(train_x):
            x = torch.flatten(x, start_dim=0)
            if cir == 'prob':
                p = block_prob(x, conf, params, d)  # (1024)
            else:
                p = block_exp(x, conf, params, d)
            for j, prob in enumerate(p):
                if prob <= min_list[j]:
                    min_list[j] = prob
                if prob >= max_list[j]:
                    max_list[j] = prob
        log(f'example min: {min_list[:10]}, max: {max_list[:10]}')
        dic = {'min_l': min_list, 'max_l': max_list, 'range_len': (max_list-min_list)/k}
        results.append(dic)
    torch.save(results, save_path)


def train_range_kernel(train_x, params):
    state_num_ = state_num * 14 * 14  # todo

    save_path = os.path.join(conf.analysis_dir, conf.dataset, conf.structure)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path, 'range_' + str(conf.class_idx) + '.pth')
    min_list = torch.ones((state_num_, ))
    max_list = torch.zeros((state_num_, ))
    for i, x in enumerate(train_x):
        p = kernel_prob(x, conf, params)  # todo x的p部分相同 -- window内4个像素相同
        for j, prob in enumerate(p):
            if prob <= min_list[j]:
                min_list[j] = prob
            if prob >= max_list[j]:
                max_list[j] = prob
    print(f'example min: {min_list[:10]}, max: {max_list[:10]}')
    dic = {'min_l': min_list, 'max_l': max_list, 'range_len': (max_list-min_list)/k}
    torch.save(dic, save_path)


def test_per_block(test_x, params):
    """
    for QNN of pure circuit (composed of several blocks), i.e., QCL, QCNN, CCQC
    compute coverage of test data on criteria obtained from train data
    :param test_x: sorted by class
    :param params:
    :return:
    """
    range_l = torch.load(os.path.join(conf.analysis_dir, conf.dataset, conf.structure, 'range_' + str(conf.class_idx) + '.pth'))

    log('Compute coverage of testing data...')
    for d in block_dict[conf.structure]:
        bucket_list = torch.zeros((state_num, k), dtype=torch.int)
        feat_list = torch.zeros((test_x.shape[0], state_num))

        min_l, max_l, range_len = range_l[d-1]['min_l'], range_l[d-1]['max_l'], range_l[d-1]['range_len']
        for i, x in enumerate(test_x):
            x = torch.flatten(x, start_dim=0)
            if cir == 'prob':
                p = block_prob(x, conf, params, d)  # (1024)
            else:
                p = block_exp(x, conf, params, d)  # (n_qubits)
            # feat_list[i, :] = p
            for j, prob in enumerate(p):
                if prob < min_l[j] or prob > max_l[j]:
                    continue
                bucket_list[j][int((prob-min_l[j]) // range_len[j])] = 1
        covered_num = torch.sum(bucket_list).item()
        log(f'{d} block: {covered_num}/{state_num * k}={covered_num/(state_num*k)*100}%')

        # print(f'visualize for block {d}...')
        # visual_feature_dis(feat_list, conf.num_test_img, 'block'+str(d))


def test_kernel_feat(test_x, params):
    """
    for QNN layer composed of circuit kernel, i.e., single/multi encoding
    :param test_x:
    :param params:
    :return:
    """
    state_num_ = state_num * 14 * 14  # todo 对于circuit来说只有16个state，但卷积过程得到了14*14个结果，这里简单拼接
    bucket_list = torch.zeros((state_num_, k), dtype=torch.int)
    feat_list = []

    range_l = torch.load(
        os.path.join(conf.analysis_dir, conf.dataset, conf.structure, 'range_' + str(conf.class_idx) + '.pth'))
    min_l, max_l, range_len = range_l['min_l'], range_l['max_l'], range_l['range_len']
    for i, x in enumerate(test_x):
        p = kernel_prob(x, conf, params)
        feat_list.append(p)
        for j, f in enumerate(p):
            if f < min_l[j] or f > max_l[j]:
                continue
            a = int((f-min_l[j]) // range_len[j])
            if a == k:
                a -= 1  # f == max
            bucket_list[j][a] = 1
    covered_num = torch.sum(bucket_list).item()
    print(f'coverage: {covered_num}/{state_num_ * k}={covered_num / (state_num_ * k) * 100}%')

    # print(f'visualize...')
    # visual_feature_dis(torch.stack(feat_list), conf.num_test_img, conf.structure + '_ql')


def visual_feature_dis(data, n_per_c, block_name='block1'):
    s_path = os.path.join(conf.visual_dir, 'feat_dis', conf.dataset, conf.structure, str(conf.class_idx))
    if not os.path.exists(s_path):
        os.makedirs(s_path)
    s_path = os.path.join(s_path, block_name + '.png')
    dot_graph(data, n_per_c, s_path)


def gate_set_within_block():
    # todo how to divide
    pass


if __name__ == '__main__':
    test_x, test_y = load_part_data(conf)
    train_x, train_y = load_part_data(conf=conf, train_=True)
    params = load_params_from_path(conf, device)

    log(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    log(f'Parameter: bucket num: {k}, cir: {conf.coverage_cri}, train/test num: {conf.num_test_img}')
    s_x1 = torch.tensor([])
    s_x2 = torch.tensor([])
    for y in conf.class_idx:
        idx = torch.where(train_y == y)[0]
        x_ = train_x[idx]
        s_x1 = torch.cat((s_x1, x_))

        idx = torch.where(test_y == y)[0]
        x_ = test_x[idx]
        s_x2 = torch.cat((s_x2, x_))

    log('Getting range info from training data...')
    if conf.structure in ['qcl', 'ccqc', 'pure_qcnn']:
        train_range_circuit(s_x1, params)
        log('Completed.')
        test_per_block(s_x2, params)
    elif conf.structure in ['pure_single', 'pure_multi']:
        train_range_kernel(s_x1, params)
        log('Completed.')
        test_kernel_feat(s_x2, params)
