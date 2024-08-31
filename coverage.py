"""
coverage criteria of QuanXplore
"""
from tqdm import tqdm
import torch
import numpy as np

from tools.entanglement import MW
from tools.internal import block_out


class Ksc:
    def __init__(self, model, m_params, state_num, profile_path, k=100):
        self.state_num = state_num
        self.k = k
        self.range_l = torch.load(profile_path)
        self.m_params, self.model = m_params, model

    def fit(self, test_x, conf, depth):
        """测试集覆盖率"""
        cell_list = torch.zeros((self.state_num, self.k), dtype=torch.int)
        cover_list = torch.zeros((test_x.shape[0], self.state_num, self.k), dtype=torch.int)

        min_l, max_l = self.range_l['min_l'], self.range_l['max_l']
        range_len = (max_l - min_l) / self.k
        for i, x in enumerate(test_x):
            x = torch.flatten(x, start_dim=0)
            p = block_out(x, conf, self.m_params, depth=depth)
            for j, prob in enumerate(p):
                if prob < min_l[j] or prob > max_l[j]:
                    continue
                cell_list[j][int((prob - min_l[j]) // range_len[j])] = 1
                cover_list[i][j][int((prob - min_l[j]) // range_len[j])] = 1
        covered_num = torch.sum(cell_list).item()
        return covered_num / (self.state_num * self.k), cover_list

    def rank(self, conf, depth, test_x, budget=0.1):
        select_num = int(budget * test_x.shape[0])
        _, cover_list = self.fit(test_x, conf, depth)
        subset = []
        lst = list(range(test_x.shape[0]))
        init = np.random.choice(range(test_x.shape[0]))
        lst.remove(init)  # 剩余样本
        subset.append(init)  # 挑选出的样本
        max_cover_num = cover_list[init].sum().item()
        cover_last = cover_list[init]
        while True:
            flag = False
            for idx in tqdm(lst):
                tmp = torch.bitwise_or(cover_last, cover_list[idx])
                now_cover_num = tmp.sum().item()
                if now_cover_num > max_cover_num:
                    max_cover_num = now_cover_num
                    max_idx = idx
                    max_cover = tmp
                    flag = True
            cover_last = max_cover
            if not flag or len(subset) == select_num:
                break
            lst.remove(max_idx)
            subset.append(max_idx)
            print(max_cover_num)
        return subset, max_cover_num / (self.state_num * self.k)*100


class Scc:
    def __init__(self, model, m_params, state_num, profile_path, std_k=0):
        self.state_num = state_num
        self.std_k = std_k
        self.range_l = torch.load(profile_path)
        self.m_params, self.model = m_params, model

    def fit(self, test_x, conf, depth):
        upper_cover = torch.zeros((self.state_num,))
        lower_cover = torch.zeros((self.state_num,))
        cover_list = torch.zeros((test_x.shape[0], self.state_num*2))
        min_l, max_l, std_l = self.range_l['min_l'], self.range_l['max_l'], self.range_l['std_l']
        for i, x in enumerate(test_x):
            x = torch.flatten(x, start_dim=0)
            p = block_out(x, conf, self.m_params, depth=depth)
            for j, v in enumerate(p):
                if v > max_l[j] + self.std_k * std_l[j]:
                    upper_cover[j] = 1
                    cover_list[i][j] = 1
                if v < min_l[j] - self.std_k * std_l[j]:
                    lower_cover[j] = 1
                    cover_list[i][j+self.state_num] = 1
        return (upper_cover.sum().item()+lower_cover.sum().item()) / (self.state_num * 2), cover_list

    def rank(self, test_x, conf, depth, budget=0.1):
        _, cover_list = self.fit(test_x, conf, depth)
        select_num = int(budget * test_x.shape[0])
        subset = []
        lst = list(range(test_x.shape[0]))
        init = np.random.choice(range(test_x.shape[0]))
        lst.remove(init)
        subset.append(init)
        max_cover_num = torch.sum(cover_list[init]).item()
        cover_last = cover_list[init]
        while True:
            flag = False
            for idx in tqdm(lst):
                tmp = torch.bitwise_or(cover_last, cover_list[idx])
                cover = tmp.sum().item()
                if cover > max_cover_num:
                    max_cover_num = cover
                    max_idx = idx
                    flag = True
                    max_cover = tmp
            if not flag or len(subset) == select_num:
                break
            lst.remove(max_idx)
            subset.append(max_idx)
            cover_last = max_cover
            print(max_cover_num)
        return subset, max_cover_num / (2*self.state_num)*100


class Tsc:
    def __init__(self, model, m_params, state_num, topk=1):
        self.topk = topk
        self.state_num = state_num
        self.m_params, self.model = m_params, model

    def fit(self, test_x, conf, depth):
        top_l = torch.zeros((self.state_num,))
        top_list = torch.zeros((test_x.shape[0], self.state_num))
        for i, x in enumerate(test_x):
            x = torch.flatten(x, start_dim=0)
            p = block_out(x, conf, self.m_params, depth=depth)
            topk_p = torch.topk(p, k=self.topk)[1]
            top_l[topk_p] = 1
            top_list[i][topk_p] = 1
        return torch.sum(top_l).item() / self.state_num, top_list

    def rank(self, test_x, conf, depth, budget=0.1):
        budget_num = int(budget*len(test_x))
        _, top = self.fit(test_x, conf, depth)
        init = np.random.choice(range(test_x.shape[0]))
        subset = []
        lst = list(range(len(test_x)))
        lst.remove(init)
        subset.append(init)
        max_cover = torch.sum(top[init]).item()
        cover_now = top[init]
        while True:
            flag = False
            for idx in tqdm(lst):
                tmp = torch.bitwise_or(cover_now, top[idx])
                cover1 = tmp.sum().item()
                if cover1 > max_cover:
                    max_cover = cover1
                    max_idx = idx
                    flag = True
                    max_cover_now = tmp
            if not flag or len(subset) == budget_num:
                break
            lst.remove(max_idx)
            subset.append(max_idx)
            cover_now = max_cover_now
            print(max_cover)
        return subset, max_cover/self.state_num*100


class Kec:
    def __init__(self, model, m_params, k=500, u=0.5, l=-0.5):
        self.u = u
        self.l = l
        self.k = k
        self.m_params, self.model = m_params, model

    def fit(self, test_x, conf, depth):
        in_ent, out_ent, chan_ent = MW(test_x, self.m_params, conf, depth)  # test_x.shape[0]
        ent_up_rate = chan_ent / in_ent
        cell_list = torch.zeros((self.k,), dtype=int)
        r = (self.u - self.l) / self.k
        cover_list = torch.zeros((test_x.shape[0], self.k), dtype=int)
        for i, ecr in enumerate(ent_up_rate):
            if ecr < self.l or ecr > self.u:
                continue
            cell_list[int((ecr - self.l) // r)] = 1
            cover_list[i][int((ecr - self.l) // r)] = 1
        cov_rate = torch.sum(cell_list).item() / self.k * 100
        return cov_rate, cover_list

    def rank(self, test_x, conf, depth, budget=0.1):
        budget_num = int(budget*len(test_x))
        init = np.random.choice(range(test_x.shape[0]))
        subset = []
        lst = list(range(len(test_x)))
        lst.remove(init)
        subset.append(init)
        _, cover_list = self.fit(test_x, conf, depth)
        max_cover_num = 1
        cover_last = cover_list[init]

        while True:
            flag = False
            for idx in tqdm(lst):
                tmp = torch.bitwise_or(cover_last, cover_list[idx])
                now_cover_num = tmp.sum().item()
                if now_cover_num > max_cover_num:
                    max_cover_num = now_cover_num
                    max_idx = idx
                    max_cover = tmp
                    flag = True
            if not flag or len(subset) == budget_num:
                break
            cover_last = max_cover
            lst.remove(max_idx)
            subset.append(max_idx)
            print(max_cover_num)
        return subset, max_cover_num/self.k * 100


class CoverageHandler:
    def __init__(self, model, m_params, state_num, profile_path, cri='ksc'):
        self.handler = None
        if cri == 'ksc':
            self.handler = Ksc(model, m_params, state_num, profile_path, k=100)
        elif cri == 'tsc':
            self.handler = Tsc(model, m_params, state_num, topk=1)
        elif cri == 'scc':
            self.handler = Scc(model, m_params, state_num, profile_path, std_k=0)
        elif cri == 'kec':
            self.handler = Kec(model, m_params, k=500, u=0.5, l=-0.5)

    def fit(self, test_x, conf, depth):
        return self.handler.fit(test_x, conf, depth)

    def rank(self, test_x, conf, depth, budget=0.1):
        return self.handler.rank(test_x, conf, depth, budget)

    def update_cov(self, now_cover, new_m, conf, depth):
        o_num = torch.sum(torch.any(now_cover, dim=0)).item()

        _, m_cover = self.handler.fit(torch.tensor([new_m]), conf, depth)
        m_cover = m_cover.view(-1)
        new_cover = torch.cat((now_cover, m_cover.unsqueeze(0)), dim=0)
        n_num = torch.sum(torch.any(new_cover, dim=0)).item()

        return new_cover, n_num, n_num > o_num

