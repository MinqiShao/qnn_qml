"""
the cell coverage for circuit-as-body QNN, QCL, QCNN, CCQC, Hier
k-cell, corner (upper and lower), topk under probability/expectation
k-cell, corner (upper and lower) under entanglement
"""

from config import *
from tools.model_loader import load_params_from_path
from tools.data_loader import load_part_data, load_adv_imgs
from tools import Log
from tools.internal import block_out
from tools.entanglement import MW
from models.circuits import block_dict, qubit_block_dict
import torch
import os
from datetime import datetime

conf = get_arguments()
device = torch.device('cpu')


class Tester:
    def __init__(self):
        self.model_n = 'hier_' + conf.hier_ue if conf.structure == 'hier' else conf.structure
        self.cri = conf.cov_cri
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
        results = []

        for idx, d in enumerate(block_dict[conf.structure]):
            min_list = torch.ones((self.n_qubit_list[idx],))  # min/max prob for each state
            max_list = torch.full((self.n_qubit_list[idx],), -1.0, dtype=torch.float32)

            self.log(f'Block {d}')
            for i, x in enumerate(train_x):
                x = torch.flatten(x, start_dim=0)
                p = block_out(x, conf, self.params, d)
                for j, prob in enumerate(p):
                    if prob <= min_list[j]:
                        min_list[j] = prob
                    if prob >= max_list[j]:
                        max_list[j] = prob
            self.log(f'example min: {min_list[:10]}, max: {max_list[:10]}')
            dic = {'min_l': min_list, 'max_l': max_list, 'range_len': (max_list - min_list) / self.k}
            results.append(dic)
        torch.save(results, self.save_path)

    def train_ent(self, train_x):
        results = []
        for d in block_dict[conf.structure]:
            min = 1
            max = 0
            self.log(f'Block {d}')
            _, out_list, _ = MW(train_x, self.params, conf, d)
            i = torch.min(out_list)
            if i < min:
                min = i
            i = torch.max(out_list)
            if i > max:
                max = i
            self.log(f'min: {min}, max: {max}')
            dic = {'min': min, 'max': max, 'range_len': (max - min) / self.k}
            results.append(dic)
        torch.save(results, self.save_path)

    def test_k_cell(self, test_x):
        range_l = torch.load(self.save_path)

        for idx, d in enumerate(block_dict[conf.structure]):
            bucket_list = torch.zeros((self.n_qubit_list[idx], self.k), dtype=torch.int)

            min_l, max_l, range_len = range_l[d - 1]['min_l'], range_l[d - 1]['max_l'], range_l[d - 1]['range_len']
            for i, x in enumerate(test_x):
                x = torch.flatten(x, start_dim=0)
                p = block_out(x, conf, self.params, d)
                for j, prob in enumerate(p):
                    if prob < min_l[j] or prob > max_l[j]:
                        continue
                    bucket_list[j][int((prob - min_l[j]) // range_len[j])] = 1
            covered_num = torch.sum(bucket_list).item()
            self.log(f'{d} block: {covered_num}/{self.n_qubit_list[idx] * self.k}={covered_num / (self.n_qubit_list[idx] * self.k) * 100}%')

    def test_corner(self, test_x):
        range_l = torch.load(self.save_path)
        self.log('Compute corner coverage of testing data...')
        for idx, d in enumerate(block_dict[conf.structure]):
            upper_cover = torch.zeros((self.n_qubit_list[idx],))
            lower_cover = torch.zeros((self.n_qubit_list[idx],))

            min_l, max_l = range_l[d - 1]['min_l'], range_l[d - 1]['max_l']
            for i, x in enumerate(test_x):
                x = torch.flatten(x, start_dim=0)
                p = block_out(x, conf, self.params, d)
                for j, v in enumerate(p):
                    if v > max_l[j]:
                        upper_cover[j] = 1
                    if v < min_l[j]:
                        lower_cover[j] = 1
            u, l = torch.sum(upper_cover).item(), torch.sum(lower_cover).item()
            self.log(f'Block {d}: #upper cover: {u}, #lower cover: {l}, coverage: {(u + l) / (2 * self.n_qubit_list[idx]) * 100}%')

    def test_topk(self, test_x, topk=1):
        # 有多少个neuron曾经成为过某个样本的topk，k=1 2 3
        self.log('Compute the topk coverage of testing data...')
        for idx, d in enumerate(block_dict[conf.structure]):
            top_l = torch.zeros((self.n_qubit_list[idx],))

            for i, x in enumerate(test_x):
                x = torch.flatten(x, start_dim=0)
                p = block_out(x, conf, self.params, d)
                topk_p = torch.topk(p, k=topk)[1]
                top_l[topk_p] = 1
            self.log(f'Block {d}: top {topk} coverage: {torch.sum(top_l).item() / self.n_qubit_list[idx] * 100}%')

    def test_ent_cov(self, test_x):
        range_l = torch.load(self.save_path)
        self.log('Compute the entanglement coverage of testing data...')
        for d in block_dict[conf.structure]:
            bucket_list = torch.zeros((self.k,), dtype=torch.int)
            min_e, max_e, r_l = range_l[d - 1]['min'], range_l[d - 1]['max'], range_l[d - 1]['range_len']
            _, ent_l, chan_ent = MW(test_x, self.params, conf, d)  # test_x.shape[0]
            upper_num, lower_num = 0, 0
            for e in ent_l:
                if e < min_e:
                    lower_num += 1
                    continue
                if e > max_e:
                    upper_num += 1
                    continue
                bucket_list[int((e - min_e) // r_l)] = 1
            self.log(f'Block {d}: avg_ent_chan: {torch.mean(chan_ent).item()}'
                     f'out entanglement coverage: {torch.sum(bucket_list).item() / self.k * 100}%, upper: {upper_num}/{test_x.shape[0]}, lower: {lower_num}/{test_x.shape[0]}')

    def test_ent_corner(self, test_x):
        pass

    def run(self, train_x, test_x):
        self.log(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        self.log(f'Parameter: cell num: {self.k}, cir: {self.cri}, train num: {train_x.shape[0]}, test num: {test_x.shape[0]}')

        if self.ent:
            if self.need_train:
                self.train_ent(train_x)
            self.test_ent_cov(test_x)
        else:
            if self.need_train:
                self.train_cell(train_x)
            self.test_k_cell(test_x)
            self.test_corner(test_x)
            self.test_topk(test_x)



if __name__ == '__main__':
    train_x, train_y = load_part_data(conf=conf, train_=True, num_data=conf.num_train)
    test_x, test_y = load_part_data(conf, num_data=conf.num_test)

    if conf.with_adv:
        _, adv_imgs = load_adv_imgs(conf)
        adv_imgs = adv_imgs.squeeze(1)
        test_x = torch.cat((test_x, adv_imgs), dim=0)

    tester = Tester()
    tester.run(train_x, test_x)


