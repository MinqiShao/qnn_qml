"""
the cell coverage for circuit-as-body QNN, QCL, QCNN, CCQC, Hier
k-cell, corner (upper and lower), topk under probability
k-cell(entropy) under entanglement
"""
import time

from config import *
from tools.model_loader import load_params_from_path
from tools.data_loader import load_part_data, load_adv_imgs
from tools import Log
from tools.internal import block_out
from tools.entanglement import MW
from tools.gragh import single_heatmap_graph, multi_heatmap_graph
from models.circuits import depth_dict, cla_qubit_dict, aux_qubit_dict, qubit_dict
from Coverage import Kec, Ksc, Scc, Tsc
import torch
import os
from datetime import datetime

conf = get_arguments()
device = torch.device('cpu')


class Tester:
    def __init__(self, k=100, std_k=0, topk=1, l=-1, u=1):
        self.params = {}
        self.init_param(k, std_k, topk, l, u)
        self.init_path()

    def init_param(self, k, std_k, topk, l, u):
        if conf.cov_cri == 'prob':
            self.params['ent'] = False
        elif conf.cov_cri == 'ent':
            self.params['ent'] = True
        self.params['depth'] = depth_dict[conf.structure]
        self.params['cla_qubits'] = cla_qubit_dict[conf.structure]
        self.params['aux_qubits'] = aux_qubit_dict[conf.structure]
        self.params['state_num'] = 2 ** qubit_dict[conf.structure]
        self.params['cri'] = conf.cov_cri
        self.params['k'] = k
        self.params['std_k'] = std_k
        self.params['topk'] = topk
        self.params['l'] = l
        self.params['u'] = u
        self.params['need_train'] = True
        self.m_params, _ = load_params_from_path(conf, device)

    def init_path(self):
        model_n = 'hier_' + conf.hier_u if conf.structure == 'hier' else conf.structure
        log_dir = os.path.join(conf.analysis_dir, conf.dataset, model_n)
        vis_dir = os.path.join(conf.visual_dir, 'heatmap', conf.dataset, model_n, str(conf.class_idx), conf.cov_cri)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
        self.log = Log(os.path.join(log_dir, 'log' + str(conf.class_idx) + '.txt'))
        self.save_path = os.path.join(log_dir, conf.cov_cri + '_range_' + str(conf.class_idx) + '_' + str(conf.num_train) +'.pth')
        if os.path.exists(self.save_path):
            self.params['need_train'] = False
            print(f'{self.save_path} has existed.')
        self.fig_path = os.path.join(vis_dir, str(self.params['k']) + '_with_' + conf.attack) if conf.with_adv else os.path.join(vis_dir, str(self.params['k']) + '_ori')

    #######   for aux qubits  #######
    def train_cell(self, train_x):
        self.log('Computing cell range from training data...')

        min_list = torch.ones((self.params['state_num'],))  # min/max prob for each state
        max_list = torch.full((self.params['state_num'],), -1.0, dtype=torch.float32)
        whole_list = torch.zeros((train_x.shape[0], self.params['state_num']))  # for computing std

        for i, x in enumerate(train_x):
            x = torch.flatten(x, start_dim=0)
            p = block_out(x, conf, self.m_params, depth=self.params['depth'])
            whole_list[i, :] = p
            for j, prob in enumerate(p):
                if prob <= min_list[j]:
                    min_list[j] = prob
                if prob >= max_list[j]:
                    max_list[j] = prob
        self.log(f'example min: {min_list[:10]}, max: {max_list[:10]}')
        dic = {'min_l': min_list, 'max_l': max_list, 'std_l': torch.std(whole_list, dim=0)}
        torch.save(dic, self.save_path)

    # def train_ent(self, train_x):
    #     self.log('Computing entanglement range from training data...')
    #     _, out_list, _ = MW(train_x, self.params, conf, self.params['depth'])
    #     dic = {'min': torch.min(out_list), 'max': torch.max(out_list), 'std': torch.std(out_list)}
    #     self.log(dic)
    #     torch.save(dic, self.save_path)

    def test_k_cell(self, test_x, range_l):
        state_num = self.params['state_num']
        k = self.params['k']
        cell_list = torch.zeros((state_num, k), dtype=torch.int)

        min_l, max_l = range_l['min_l'], range_l['max_l']
        range_len = (max_l - min_l) / k
        s_time = time.time()
        for i, x in enumerate(test_x):
            x = torch.flatten(x, start_dim=0)
            p = block_out(x, conf, self.m_params, depth=self.params['depth'])
            for j, prob in enumerate(p):
                if prob < min_l[j] or prob > max_l[j]:
                    continue
                cell_list[j][int((prob - min_l[j]) // range_len[j])] = 1
        e_time = time.time()
        covered_num = torch.sum(cell_list).item()
        self.log(f'### {k}-cell coverage: {covered_num}/{state_num * k}={covered_num / (state_num * k) * 100}%')
        self.log(f'### average test time: {(e_time - s_time)/test_x.shape[0]}s')

    def test_corner(self, test_x, range_l, std_k=0):
        self.log('Compute corner coverage of testing data...')
        state_num = self.params['state_num']
        upper_cover = torch.zeros((state_num,))
        lower_cover = torch.zeros((state_num,))

        min_l, max_l, std_l = range_l['min_l'], range_l['max_l'], range_l['std_l']
        s_time = time.time()
        for i, x in enumerate(test_x):
            x = torch.flatten(x, start_dim=0)
            p = block_out(x, conf, self.m_params, depth=self.params['depth'])
            for j, v in enumerate(p):
                if v > max_l[j] + std_k * std_l[j]:
                    upper_cover[j] = 1
                if v < min_l[j] - std_k * std_l[j]:
                    lower_cover[j] = 1
        e_time = time.time()
        u, l = torch.sum(upper_cover).item(), torch.sum(lower_cover).item()
        self.log(f'### upper cover: {u}, #lower cover: {l}, coverage: {(u + l) / (2 * state_num) * 100}%')
        self.log(f'### average test time: {(e_time - s_time) / test_x.shape[0]}s')

    def test_topk(self, test_x, topk=1):
        # 有多少个neuron曾经成为过某个样本的topk，k=1 2
        self.log('Compute the topk coverage of testing data...')
        state_num = self.params['state_num']
        top_l = torch.zeros((state_num,))

        s_time = time.time()
        for i, x in enumerate(test_x):
            x = torch.flatten(x, start_dim=0)
            p = block_out(x, conf, self.m_params, depth=self.params['depth'])
            topk_p = torch.topk(p, k=topk)[1]
            top_l[topk_p] = 1
        e_time = time.time()
        self.log(f'### top {topk} coverage: {torch.sum(top_l).item() / state_num * 100}%')
        self.log(f'### average test time: {(e_time - s_time) / test_x.shape[0]}s')

    #  静态上下限
    #######   for classification qubits  #######
    def test_cla_qubit(self, test_x, l=0, u=1):
        self.log('Plot probability diversity on classification qubits under testing data...')
        self.fig_path += '_' + str(test_x.shape[0]) + '.png'
        state_num = 2 ** len(self.params['cla_qubits'])
        p_l = torch.zeros((state_num, test_x.shape[0]))  # (4, num)
        for i, x in enumerate(test_x):
            x = torch.flatten(x, start_dim=0)
            p = block_out(x, conf, self.m_params, depth=self.params['depth'], qubit_l=self.params['cla_qubits'])
            p_l[:, i] = p
        en = multi_heatmap_graph(p_l, k=self.params['k'], save_path=self.fig_path, l=l, u=u)
        self.log(f'entropy of probability distribution: {en}')

    #######   ent for whole state  #######
    def test_ent(self, test_x, l=0, u=1):
        k = self.params['k']
        self.log('Plot the entanglement diversity of testing data...')
        self.fig_path += '_' + str(test_x.shape[0]) + '.png'
        s_time = time.time()
        in_ent, out_ent, chan_ent = MW(test_x, self.m_params, conf, self.params['depth'])  # test_x.shape[0]
        e_time = time.time()
        ent_up_rate = chan_ent / in_ent
        cell_list = torch.zeros((k,), dtype=int)
        r = (u-l)/k
        for ecr in ent_up_rate:
            if ecr < l or ecr > u:
                continue
            cell_list[int((ecr-l) // r)] = 1
        cov_rate = torch.sum(cell_list).item() / k * 100
        self.log(f'### {k}-cell ent coverage: {cov_rate}%')

        en = single_heatmap_graph(ent_up_rate, k=self.params['k'], save_path=self.fig_path, l=l, u=u)
        # self.log(f'entropy of ent : {en}')
        self.log(f'average test time: {(e_time - s_time) / test_x.shape[0]}s')

    def run(self, train_x, test_x):
        self.log(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        self.log(f'Parameter: {self.params}')

        if self.params['ent']:
            self.test_ent(test_x, l=self.params['l'], u=self.params['u'])
        else:
            # self.test_cla_qubit(test_x, l=0, u=1)
            if self.params['need_train']:
                s_time = time.time()
                self.train_cell(train_x)
                e_time = time.time()
                self.log(f'train time: {e_time - s_time}s')
            range_l = torch.load(self.save_path)
            self.test_k_cell(test_x, range_l)
            self.test_corner(test_x, range_l, std_k=self.params['std_k'])
            self.test_topk(test_x, topk=self.params['topk'])



if __name__ == '__main__':
    train_x, train_y = load_part_data(conf=conf, train_=True, num_data=conf.num_train)
    test_x, test_y = load_part_data(conf, num_data=conf.num_test)

    if conf.with_adv:
        _, adv_imgs = load_adv_imgs(conf)
        # todo 是否保留成功的部分
        adv_imgs = adv_imgs.squeeze(1)
        #test_x = adv_imgs
        test_x = torch.cat((test_x, adv_imgs), dim=0)

    tester = Tester(k=100, std_k=1, topk=2, l=-0.5, u=0.5)
    tester.run(train_x, test_x)


