"""
coverage-guided testing generation
I: initial seeds, b: budget, M: model
Q: test case queue, U: defect queue
"""
import torch
import random
import os
from config import *
from models.circuits import qubit_dict, depth_dict
from tools.model_loader import load_params_from_path
from tools.data_loader import load_correct_data
from tools.mutator import Mutator, Seed
from Coverage import CoverageHandler

conf = get_arguments()
device = torch.device('cpu')


def initial_seeds(test_x, test_y):
    I = []
    for i, x in enumerate(test_x):
        I.append(Seed(x, x, 0, test_y[i], test_y[i], 0, 0, 0))
    return I

def coverage_function(handler, conf, depth):
    def func(now_cover, m):
        return handler.update_cov(now_cover, m, conf, depth)
    return func

def mutation_function(try_num=50):
    def func(seed):
        return Mutator.mutate_one(seed.mutant, seed.cl, seed.l0_ref, seed.linf_ref, try_num)
    return func

def check_fail(model, seed):
    pred = model.predict(seed.mutant).argmax(dim=1)[0]
    if pred != seed.gt:
        return True
    return False

def update_coverage(now_num, now_cover, new_x, cov_fit):
    _, n_cover = cov_fit(torch.tensor([new_x]))
    n_cover = n_cover.view(-1)
    new_cover = torch.any(torch.cat((now_cover, n_cover.unsqueeze(0)), dim=0), dim=0)
    new_num = new_cover.sum().item()
    inc = new_num > now_num
    return new_cover, new_num, inc

def fuzz(I, M, ori_cover, mutate_func, cov_func, budget=50, ran=False):
    Q, U = I, []
    now_cover = ori_cover.clone()
    covered_num = now_cover.sum().item()
    while len(Q) > 0:
        print(f'----current length of test queue: {len(Q)}, length of defect queue: {len(U)}'
              f'----current covered num: {covered_num}')
        t = Q[0]
        Q.remove(0)
        print('dequeued from Q')
        if t.m_times > budget:
            print(f'budget of this sample has merged. dequeue it from Q.')
            continue
        m, cl, changed, l0_ref, linf_ref = mutate_func(t)  # 一次产生一个mutant
        t.m_times += 1
        t.mutant, t.cl, t.l0_ref, t.linf_ref = m, cl, l0_ref, linf_ref
        if changed:
            # check whether tn is a bug or increase coverage
            if check_fail(M, t):
                print('!!!generate a failed test!')
                U.append(t.mutant)
                # append generated mutant into test suite
                now_cover, covered_num, _ = cov_func(now_cover, t.mutant)
                continue
            if ran:  # 随机进入队列
                if random.random() < 0.9:
                    Q.append(t)
                    print('randomly enqueued back to Q')
                    continue
            # only judge whether coverage increases
            _, _, inc = cov_func(now_cover, t.mutant)
            if inc:
                Q.append(t)
                print('not fail but increase cov, enqueued back to Q')
        print(f'cannot generate valid mutants, dequeue from Q')

    gr = len(U) / len(I) * 100
    print(f'Generate Rate: {gr:.2f}%')
    return gr


if __name__ == '__main__':

    model_n = 'hier_' + conf.hier_u if conf.structure == 'hier' else conf.structure
    m_params, model = load_params_from_path(conf, device)
    test_x, test_y = load_correct_data(conf, model, num_data=conf.num_test)

    log_dir = os.path.join(conf.analysis_dir, conf.dataset, model_n)
    profile_path = os.path.join(log_dir, conf.cov_cri + '_range_' + str(conf.class_idx) + '_' + str(conf.num_train) + '.pth')
    state_num = 2**qubit_dict[conf.structure]
    depth = depth_dict[conf.structure]

    coverage_handler = CoverageHandler(model, m_params, state_num, profile_path, conf.cri)

    I = initial_seeds(test_x, test_y)
    _, ori_cover = coverage_handler.fit(test_x, conf, depth)
    ori_cover = torch.any(ori_cover.view(ori_cover.shape[0], -1), dim=0)

    cov_func = coverage_function(coverage_handler, conf, depth)
    mut_func = mutation_function(10)

    fuzz(I, model, ori_cover, mut_func, cov_func, budget=50, ran=False)
