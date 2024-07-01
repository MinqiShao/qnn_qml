"""
test the bucket coverage (probability, entanglement) within and between classes
the internal statistics (whole ent -> run.py)
"""
from tqdm import tqdm

from config import *
from tools.model_loader import load_params_from_path
from tools.data_loader import load_test_data
from tools import Log
from tools.internal import block_prob
from tools.gragh import dot_graph
from models.circuits import block_dict
import torch
import os

conf = get_arguments()
device = torch.device('cpu')

n_qubits = 10
k = 10000  # bucket num
bucket_len = 1/k


def per_block(test_x, params):
    """
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

    s_x = torch.tensor([])
    for y in conf.class_idx:
        idx = torch.where(test_y == y)[0]
        x_ = test_x[idx]
        s_x = torch.cat((s_x, x_))
    per_block(s_x, params)
