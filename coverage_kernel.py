"""
the cell coverage for circuit-as-kernel QNN, single/multi encoding
k-cell, corner (upper and lower), topk under probability
k-cell, corner (upper and lower) under entanglement
"""

from config import *
from tools.model_loader import load_params_from_path
from tools.data_loader import load_part_data, load_adv_imgs
from tools import Log
from tools.internal import kernel_out
from tools.entanglement import MW_kernel
from models.circuits import block_dict, qubit_block_dict, depth_dict
import torch
import os
from datetime import datetime

conf = get_arguments()
device = torch.device('cpu')


class Tester:
    def __init__(self):
        self.cri = conf.cov_cri
        self.e_n = 14 * 14  # 卷积次数
        self.k = 100  # cell num
        self.init_param()
        self.init_path()

    def init_param(self):
        self.n_qubit_list = qubit_block_dict[conf.structure]
        if self.cri == 'prob':
            self.n_qubit_list = [2 ** n for n in self.n_qubit_list]
            self.ent = False
        elif self.cri == 'ent':
            self.ent = True
        self.params, _ = load_params_from_path(conf, device)

    def init_path(self):
        log_dir = os.path.join(conf.analysis_dir, conf.dataset, self.model_n)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_path = os.path.join(log_dir, 'log' + str(conf.class_idx) + '.txt')
        self.log = Log(log_path)
        self.save_path = os.path.join(log_dir, self.cri + '_range_' + str(conf.class_idx) + '_' + str(conf.num_train) +'.pth')
        if os.path.exists(self.save_path):
            self.need_train = False
            print(f'{self.save_path} has existed.')

    def train_cell(self, train_x):
        self.log('Compute the cell range of training data...')
        state_num_ = self.n_qubit_list[0] * self.e_n

        min_list = torch.ones((state_num_,))
        max_list = torch.full((state_num_,), -1.0, dtype=torch.float32)
        for i, x in enumerate(train_x):
            p = kernel_out(x, conf, self.params, self.ent)
            for j, prob in enumerate(p):
                if prob <= min_list[j]:
                    min_list[j] = prob
                if prob >= max_list[j]:
                    max_list[j] = prob
        # todo examine range range==0
        range_l = (max_list - min_list) / self.k
        r_exist_l = torch.ones((state_num_,))
        for i, r in enumerate(range_l):
            if r == 0:
                r_exist_l[i] = 0
        dic = {'min_l': min_list, 'max_l': max_list, 'range_len': range_l, 'r_exist_l': r_exist_l}
        torch.save(dic, self.save_path)
        print(f'example min: {min_list[:10]}, max: {max_list[:10]}, range_exist_num: {torch.sum(r_exist_l).item()}')

    def train_ent(self, train_x):
        log('Compute the entanglement range of training data...')
        min_l = torch.ones((self.e_n,))
        max_l = torch.full((self.e_n,), -1.0, dtype=torch.float32)
        for i, x in enumerate(train_x):
            _, out_ent_list = MW_kernel(x, self.params, conf)
            for j, e in enumerate(out_ent_list):  # (169)
                if e <= min_l[j]:
                    min_l[j] = e
                if e >= max_l[j]:
                    max_l[j] = e
        range_l = (max_l - min_l) / self.k
        r_exist_l = torch.ones((self.e_n,))
        for i, r in enumerate(range_l):
            if r == 0:
                r_exist_l[i] = 0
        dic = {'min_l': min_l, 'max_l': max_l, 'range_len': range_l, 'r_exist_l': r_exist_l}
        torch.save(dic, self.save_path)
        print(f'example min: {min_l[:10]}, max: {max_l[:10]}, range_exist_num: {torch.sum(r_exist_l).item()}')

    def test_k_cell(self, test_x):
        log('Compute the k-cell coverage of testing data...')
        state_num_ = self.n_qubit_list[0] * self.e_n  # todo 卷积过程得到了14*14个结果，这里简单拼接
        bucket_list = torch.zeros((state_num_, self.k), dtype=torch.int)

        range_l = torch.load(self.save_path)
        min_l, max_l, range_len, r_exist_l = range_l['min_l'], range_l['max_l'], range_l['range_len'], range_l[
            'r_exist_l']
        for i, x in enumerate(test_x):
            p = kernel_out(x, conf, self.params)
            for j, f in enumerate(p):
                if r_exist_l[j] == 0:
                    # todo cover single value
                    if f == min_l[j]:
                        bucket_list[j][0] = 1
                    continue
                if f < min_l[j] or f > max_l[j]:
                    continue
                a = int((f - min_l[j]) // range_len[j])
                if a == self.k:
                    a -= 1  # f == max
                bucket_list[j][a] = 1
        covered_num = torch.sum(bucket_list).item()
        r_e_num = torch.sum(r_exist_l).item()
        t_state = (state_num_ - r_e_num) * 1 + r_e_num * self.k  # todo bucket总数
        log(f'coverage: {covered_num}/{t_state}={covered_num / t_state * 100}%')

    def test_corner(self, test_x):
        log('Compute the corner coverage of testing data...')
        state_num_ = self.n_qubit_list[0] * self.e_n
        range_l = torch.load(self.save_path)
        min_l, max_l, range_len, r_exist_l = range_l['min_l'], range_l['max_l'], range_l['range_len'], range_l[
            'r_exist_l']

        upper_cover = torch.zeros((state_num_,))
        lower_cover = torch.zeros((state_num_,))
        for i, x in enumerate(test_x):
            p = kernel_out(x, conf, self.params)
            for j, f in enumerate(p):
                if f < min_l[j]:
                    lower_cover[j] = 1
                if f > max_l[j]:
                    upper_cover[j] = 1
        u, l = torch.sum(upper_cover).item(), torch.sum(lower_cover).item()
        log(f'upper cover: {u}/{state_num_}, lower cover: {l}/{state_num_}, coverage: {(u + l) / (2 * state_num_) * 100}%')

    def test_k_ent(self, test_x):
        range_l = torch.load(self.save_path)
        log('Compute the entanglement coverage of testing data...')
        bucket_list = torch.zeros((self.e_n, self.k), dtype=torch.int)
        min_l, max_l, range_len, r_exist_l = range_l['min_l'], range_l['max_l'], range_l['range_len'], range_l[
            'r_exist_l']
        for i, x in enumerate(test_x):
            _, out_ent = MW_kernel(x, self.params, conf)
            for j, e in enumerate(out_ent):
                if r_exist_l[j] == 0:
                    # todo cover single value
                    if e == min_l[j]:
                        bucket_list[j][0] = 1
                    continue
                if e < min_l[j] or e > max_l[j]:
                    continue
                a = int((e - min_l[j]) // range_len[j])
                if a == self.k:
                    a -= 1  # f == max
                bucket_list[j][a] = 1
        covered_num = torch.sum(bucket_list).item()

        r_e_num = torch.sum(r_exist_l).item()
        t_state = (self.e_n - r_e_num) * 1 + r_e_num * self.k
        log(f'coverage: {covered_num}/{t_state}={covered_num / t_state * 100}%')

    def run(self, train_x, test_x):
        log(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        log(f'Parameter: cell num: {self.k}, cri: {self.cri}, train num: {train_x.shape[0]}, test num: {test_x.shape[0]}')
        if self.ent:
            if self.need_train:
                self.train_ent(train_x)
            self.test_k_ent(test_x)
        else:
            if self.need_train:
                self.train_cell(train_x)
            self.test_k_cell(test_x)
            self.test_corner(test_x)



if __name__ == '__main__':
    train_x, train_y = load_part_data(conf=conf, train_=True, num_data=conf.num_train)
    test_x, test_y = load_part_data(conf, num_data=conf.num_test)

    if conf.with_adv:
        _, adv_imgs = load_adv_imgs(conf)
        adv_imgs = adv_imgs.squeeze(1)
        test_x = torch.cat((test_x, adv_imgs), dim=0)

    tester = Tester()
    tester.run(train_x, test_x)


