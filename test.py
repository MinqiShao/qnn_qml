"""
test the bucket coverage (probability, entanglement) within and between classes
the internal statistics (whole ent -> run.py)
"""
from tqdm import tqdm

from config import *
from tools.model_loader import load_params_from_path
from tools.data_loader import load_test_data
from tools import Log
from tools.internal import block_prob, kernel_prob
from tools.gragh import dot_graph
from models.circuits import block_dict
import torch
import os

conf = get_arguments()
device = torch.device('cpu')

n_qubits = 10
k = 100  # bucket num
bucket_len = 1/k


def per_block(test_x, params):
    """
    for QNN of pure circuit (composed of several blocks), i.e., QCL, QCNN, CCQC
    :param test_x: sorted by class
    :param params:
    :return:
    """
    state_num = 2 ** n_qubits

    for d in block_dict[conf.structure]:
        bucket_list = [0] * k
        bucket_list = [bucket_list for _ in range(state_num)]  # (1024, k)
        covered = torch.zeros((state_num, k))
        feat_list = torch.zeros((test_x.shape[0], state_num))
        for i, x in enumerate(test_x):
            x = torch.flatten(x, start_dim=0)
            p = block_prob(x, conf, params, d)  # (1024)
            feat_list[i, :] = p
            for j, prob in enumerate(p):
                bucket_list[j][int(prob // bucket_len)] = 1
                covered[j][int(prob // bucket_len)] += 1
        covered_num = torch.sum(torch.tensor(bucket_list)).item()
        # todo 如何更好的显示该数据
        top_covered = torch.topk(covered, dim=1, k=3)
        print(f'{d} block: {covered_num}/{state_num * k}={covered_num/(state_num*k)*100}%')

        print(f'visualize for block {d}...')
        visual_feature_dis(feat_list, conf.num_test_img, 'block'+str(d))


def kernel_feat(test_x, params):
    """
    for QNN layer composed of circuit kernel, i.e., single/multi encoding
    :param test_x:
    :param params:
    :return:
    """
    state_num = (2**4) * 14 * 14  # todo 对于circuit来说只有16个state，但卷积过程得到了14*14个结果，这里简单拼接
    bucket_list = [0] * k
    bucket_list = [bucket_list for _ in range(state_num)]
    feat_list = []
    for i, x in enumerate(test_x):
        p = kernel_prob(x, conf, params)
        feat_list.append(p)
        for j, f in enumerate(p):
            bucket_list[j][int(f // bucket_len)] = 1
    covered_num = torch.sum(torch.tensor(bucket_list)).item()
    print(f'coverage: {covered_num}/{state_num * k}={covered_num / (state_num * k) * 100}%')

    print(f'visualize...')
    visual_feature_dis(torch.stack(feat_list), conf.num_test_img, conf.structure + '_ql')


def visual_feature_dis(data, n_per_c, block_name='block1'):
    s_path = os.path.join(conf.visual_dir, 'feat_dis', conf.dataset, conf.structure, str(conf.class_idx))
    if not os.path.exists(s_path):
        os.makedirs(s_path)
    s_path = os.path.join(s_path, block_name + '.png')
    dot_graph(data, n_per_c, s_path)


def gate_set_within_block():
    pass


if __name__ == '__main__':
    test_x, test_y = load_test_data(conf)
    params = load_params_from_path(conf, device)

    print('Parameter: ')
    print(f'bucket num: {k}')
    s_x = torch.tensor([])
    for y in conf.class_idx:
        idx = torch.where(test_y == y)[0]
        x_ = test_x[idx]
        s_x = torch.cat((s_x, x_))
    # per_block(s_x, params)
    kernel_feat(s_x, params)
