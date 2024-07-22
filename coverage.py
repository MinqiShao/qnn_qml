"""
test the bucket coverage (probability, entanglement) within and between classes
the internal statistics (whole ent -> run.py)
"""
from tqdm import tqdm

from config import *
from tools.model_loader import load_params_from_path
from tools.data_loader import load_part_data, load_adv_imgs
from tools import Log
from tools.internal import block_out, kernel_out
from tools.gragh import dot_graph
from tools.entanglement import MW
from models.circuits import block_dict, qubit_block_dict
import torch
import os
from datetime import datetime

conf = get_arguments()
device = torch.device('cpu')

n_qubit_list = qubit_block_dict[conf.structure]
k = 100  # bucket num
cir = conf.cov_cri
if cir == 'prob':
    n_qubit_list = [2**n for n in n_qubit_list]
    exp = False
else:
    exp = True

if conf.structure == 'hier':
    model_n = 'hier_' + conf.hier_u
else:
    model_n = conf.structure
log_dir = os.path.join(conf.analysis_dir, conf.dataset, model_n)
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
    save_path = os.path.join(log_dir, cir + '_range_' + str(conf.class_idx) + '_' + str(conf.num_train) + '.pth')
    if os.path.exists(save_path):
        print(f'{save_path} has existed.')
        return

    for idx, d in enumerate(block_dict[conf.structure]):
        min_list = torch.ones((n_qubit_list[idx], ))  # min/max prob for each state
        max_list = torch.full((n_qubit_list[idx], ), -1.0, dtype=torch.float32)

        log(f'Block {d}')
        for i, x in enumerate(train_x):
            x = torch.flatten(x, start_dim=0)
            p = block_out(x, conf, params, d, exp)
            for j, prob in enumerate(p):
                if prob <= min_list[j]:
                    min_list[j] = prob
                if prob >= max_list[j]:
                    max_list[j] = prob
        log(f'example min: {min_list[:10]}, max: {max_list[:10]}')
        dic = {'min_l': min_list, 'max_l': max_list, 'range_len': (max_list-min_list)/k}
        results.append(dic)
    torch.save(results, save_path)


def train_ent_circuit(train_x, params):
    """
    out entanglement range from train data
    :param train_x:
    :param params:
    :return:
    """
    results = []
    save_path = os.path.join(log_dir, cir + '_range_' + str(conf.class_idx) + '_' + str(conf.num_train) + '.pth')
    if os.path.exists(save_path):
        print(f'{save_path} has existed.')
        return

    for d in block_dict[conf.structure]:
        min = 1
        max = 0
        log(f'Block {d}')
        _, out_list, _ = MW(train_x, params, conf, d)
        i = torch.min(out_list)
        if i < min:
            min = i
        i = torch.max(out_list)
        if i > max:
            max = i
        log(f'min: {min}, max: {max}')
        dic = {'min': min, 'max': max, 'range_len': (max-min)/k}
        results.append(dic)
    torch.save(results, save_path)


def train_range_kernel(train_x, params):
    state_num_ = n_qubit_list[0] * 14 * 14

    save_path = os.path.join(log_dir, 'range_' + str(conf.class_idx) + '_' + str(conf.num_train) + '.pth')
    if os.path.exists(save_path):
        print(f'{save_path} has existed.')
        return

    min_list = torch.ones((state_num_, ))
    max_list = torch.full((state_num_, ), -1.0, dtype=torch.float32)
    for i, x in enumerate(train_x):
        p = kernel_out(x, conf, params, exp)
        for j, prob in enumerate(p):
            if prob <= min_list[j]:
                min_list[j] = prob
            if prob >= max_list[j]:
                max_list[j] = prob
    # todo examine range range==0
    range_l = (max_list-min_list)/k
    r_exist_l = torch.ones((state_num_, ))
    for i, r in enumerate(range_l):
        if r == 0:
            r_exist_l[i] = 0
    dic = {'min_l': min_list, 'max_l': max_list, 'range_len': range_l, 'r_exist_l': r_exist_l}
    torch.save(dic, save_path)
    print(f'example min: {min_list[:10]}, max: {max_list[:10]}, range_exist_num: {torch.sum(r_exist_l).item()}')


def test_per_block_bucket(test_x, params):
    """
    for QNN of pure circuit (composed of several blocks), i.e., QCL, QCNN, CCQC
    compute bucket coverage of test data
    :param test_x: sorted by class
    :param params:
    :return:
    """
    range_l = torch.load(os.path.join(log_dir, cir + '_range_' + str(conf.class_idx) + '_' + str(conf.num_train) +'.pth'))

    log('Compute bucket coverage of testing data...')
    for idx, d in enumerate(block_dict[conf.structure]):
        bucket_list = torch.zeros((n_qubit_list[idx], k), dtype=torch.int)
        feat_list = torch.zeros((test_x.shape[0], n_qubit_list[idx]))

        min_l, max_l, range_len = range_l[d-1]['min_l'], range_l[d-1]['max_l'], range_l[d-1]['range_len']
        for i, x in enumerate(test_x):
            x = torch.flatten(x, start_dim=0)
            p = block_out(x, conf, params, d, exp)
            # feat_list[i, :] = p
            for j, prob in enumerate(p):
                if prob < min_l[j] or prob > max_l[j]:
                    continue
                bucket_list[j][int((prob-min_l[j]) // range_len[j])] = 1
        covered_num = torch.sum(bucket_list).item()
        log(f'{d} block: {covered_num}/{n_qubit_list[idx] * k}={covered_num/(n_qubit_list[idx]*k)*100}%')

        # print(f'visualize for block {d}...')
        # visual_feature_dis(feat_list, conf.num_test_img, 'block'+str(d))


def test_block_corner(test_x, params):
    """
    corner coverage (neurons whose corner region is covered)
    :param test_x:
    :param params:
    :return:
    """
    range_l = torch.load(os.path.join(log_dir, cir + '_range_' + str(conf.class_idx) + '_' + str(conf.num_train) +'.pth'))
    log('Compute corner coverage of testing data...')
    for idx, d in enumerate(block_dict[conf.structure]):
        upper_cover = torch.zeros((n_qubit_list[idx], ))
        lower_cover = torch.zeros((n_qubit_list[idx], ))

        min_l, max_l = range_l[d - 1]['min_l'], range_l[d - 1]['max_l']
        for i, x in enumerate(test_x):
            x = torch.flatten(x, start_dim=0)
            p = block_out(x, conf, params, d, exp)
            for j, v in enumerate(p):
                if v > max_l[j]:
                    upper_cover[j] = 1
                if v < min_l[j]:
                    lower_cover[j] = 1
        u, l = torch.sum(upper_cover).item(), torch.sum(lower_cover).item()
        log(f'Block {d}: #upper cover: {u}, #lower cover: {l}, coverage: {(u+l)/(2*n_qubit_list[idx])*100}%')


def test_block_topk(test_x, params, topk=1):
    # 有多少个neuron曾经成为过某个样本的topk，k=1 2 3
    log('Compute the topk coverage of testing data...')
    for idx, d in enumerate(block_dict[conf.structure]):
        top_l = torch.zeros((n_qubit_list[idx], ))

        for i, x in enumerate(test_x):
            x = torch.flatten(x, start_dim=0)
            p = block_out(x, conf, params, d, exp)
            topk_p = torch.topk(p, k=topk)[1]
            top_l[topk_p] = 1
        log(f'Block {d}: top {topk} coverage: {torch.sum(top_l).item() / n_qubit_list[idx] * 100}%')


def test_block_ent(test_x, params):
    range_l = torch.load(os.path.join(log_dir, cir + '_range_' + str(conf.class_idx) + '_' + str(conf.num_train) +'.pth'))
    log('Compute the entanglement coverage of testing data...')
    for d in block_dict[conf.structure]:
        bucket_list = torch.zeros((k, ), dtype=torch.int)
        min_e, max_e, r_l = range_l[d - 1]['min'], range_l[d - 1]['max'], range_l[d - 1]['range_len']
        _, ent_l, chan_ent = MW(test_x, params, conf, d)  # test_x.shape[0]
        upper_num, lower_num = 0, 0
        for e in ent_l:
            if e < min_e:
                lower_num += 1
                continue
            if e > max_e:
                upper_num += 1
                continue
            bucket_list[int((e - min_e) // r_l)] = 1
        log(f'Block {d}: avg_ent_chan: {torch.mean(chan_ent).item()}'
            f'out entanglement coverage: {torch.sum(bucket_list).item()/k * 100}%, upper: {upper_num}/{test_x.shape[0]}, lower: {lower_num}/{test_x.shape[0]}')


def test_kernel_bucket(test_x, params):
    """
    for QNN layer composed of circuit kernel, i.e., single/multi encoding
    :param test_x:
    :param params:
    :return:
    """
    state_num_ = n_qubit_list[0] * 14 * 14  # todo 对于circuit来说只有16个state，卷积过程得到了14*14个结果，这里简单拼接
    bucket_list = torch.zeros((state_num_, k), dtype=torch.int)
    feat_list = []

    range_l = torch.load(os.path.join(log_dir, cir + '_range_' + str(conf.class_idx) + '_' + str(conf.num_train) +'.pth'))
    min_l, max_l, range_len, r_exist_l = range_l['min_l'], range_l['max_l'], range_l['range_len'], range_l['r_exist_l']
    for i, x in enumerate(test_x):
        p = kernel_out(x, conf, params, exp)
        feat_list.append(p)
        for j, f in enumerate(p):
            if r_exist_l[j] == 0:
                # todo cover single value
                if f == min_l[j]:
                    bucket_list[j][0] = 1
                continue
            if f < min_l[j] or f > max_l[j]:
                continue
            a = int((f-min_l[j]) // range_len[j])
            if a == k:
                a -= 1  # f == max
            bucket_list[j][a] = 1
    covered_num = torch.sum(bucket_list).item()

    r_e_num = torch.sum(r_exist_l).item()
    t_state = (state_num_-r_e_num)*1 + r_e_num*k  # todo bucket总数
    log(f'coverage: {covered_num}/{t_state}={covered_num / t_state * 100}%')

    # print(f'visualize...')
    # visual_feature_dis(torch.stack(feat_list), conf.num_test_img, conf.structure + '_ql')


def test_kernel_corner(test_x, params):
    f_num = n_qubit_list[0] * 14*14
    range_l = torch.load(os.path.join(log_dir, cir + '_range_' + str(conf.class_idx) + '_' + str(conf.num_train) +'.pth'))
    min_l, max_l, range_len, r_exist_l = range_l['min_l'], range_l['max_l'], range_l['range_len'], range_l['r_exist_l']

    upper_cover = torch.zeros((f_num,))
    lower_cover = torch.zeros((f_num,))
    for i, x in enumerate(test_x):
        p = kernel_out(x, conf, params, exp)
        for j, f in enumerate(p):
            if f < min_l[j]:
                lower_cover[j] = 1
            if f > max_l[j]:
                upper_cover[j] = 1
    u, l = torch.sum(upper_cover).item(), torch.sum(lower_cover).item()
    log(f'upper cover: {u}/{f_num}, lower cover: {l}/{f_num}, coverage: {(u+l)/(2*f_num)*100}%')


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
    train_x, train_y = load_part_data(conf=conf, train_=True, num_data=conf.num_train)
    test_x, test_y = load_part_data(conf, num_data=conf.num_test)

    if conf.with_adv:
        _, adv_imgs = load_adv_imgs(conf, log)
        adv_imgs = adv_imgs.squeeze(1)
        test_x = torch.cat((test_x, adv_imgs), dim=0)

    params, _ = load_params_from_path(conf, device)

    log(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    log(f'Parameter: bucket num: {k}, cir: {conf.cov_cri}, train num: {train_x.shape[0]}, test num: {test_x.shape[0]}')

    log('Getting range info from training data...')
    if conf.structure in ['qcl', 'ccqc', 'pure_qcnn', 'hier']:
        if conf.cov_cri == 'ent':
            train_ent_circuit(train_x, params)
            test_block_ent(test_x, params)
        else:
            train_range_circuit(train_x, params)
            log('Completed.')
            test_per_block_bucket(test_x, params)
            test_block_corner(test_x, params)
            test_block_topk(test_x, params)
    elif conf.structure in ['pure_single', 'pure_multi']:
        train_range_kernel(train_x, params)
        log('Completed.')
        test_kernel_bucket(test_x, params)
        test_kernel_corner(test_x, params)
