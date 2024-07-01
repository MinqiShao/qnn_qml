import torch

from config import *
from models.circuits import weight_dict, QCL_circuit
from tools.internal import *
from tools.entanglement import *
from tools.data_loader import load_test_data
from tools.model_loader import load_params_from_path
from tools.adv_attack import *
from pennylane.math import fidelity, trace_distance
from pennylane import numpy as np, AmplitudeEmbedding
import time


conf = get_arguments()
device = torch.device('cpu')

test_img_num = conf.num_test_img
lr = 0.05
anti_predict_weight = 1
cov_weight = 1
ent_k = 1


def gen_adv():
    test_x, test_y = load_test_data(conf)
    params = load_params_from_path(conf, device)

    adv_num = 0
    f_list = []
    f_sum = 0
    t_list = []
    t_sum = 0
    QEA_sum = 0
    for i in range(test_img_num):
        x = torch.flatten(test_x[i], start_dim=0)
        x.requires_grad_(True)

        in_state, out_state = in_out_state(x, conf.structure, params)
        now_in_state = in_state.clone()
        ori_outputs, _ = circuit_pred(x, params, conf)
        ori_outputs = torch.tensor(ori_outputs)

        iters = 0
        while True:
            iters += 1

            now_outputs, _ = circuit_pred(x, params, conf)  # list
            now_out_state = circuit_state(x, conf, params)  # tensor
            now_ent_c, _, _ = entQ(now_in_state, now_out_state, ent_k)
            if len(conf.class_idx) == 2:
                obj_orie = DLFuzz2(now_outputs, ori_outputs, anti_predict_weight)
            else:
                obj_orie = DLFuzz3(now_outputs, ori_outputs, anti_predict_weight)

            loss = obj_orie + cov_weight * now_ent_c
            x.retain_grad()
            loss.backward(retain_graph=True)
            perturb = x.grad * lr

            x = torch.clamp((x+perturb), 0, 1)
            # now_in_state = now_in_state / torch.linalg.norm(torch.abs(now_in_state))

            now_in_state, now_out_state = in_out_state(x, conf.structure, params)
            now_ent_c, now_ent_in, now_ent_out = entQ(now_in_state, now_out_state, ent_k)
            # todo log
            print(f'iter {iters}: QEA: {now_ent_c}, ent_in: {now_ent_in}, ent_out: {now_ent_out}')

            _, new_y = circuit_pred(x, params, conf)
            if new_y != test_y[i]:
                print('gen an adv img!')
                print(f'ori_y: {test_y[i]}, new_y: {new_y}')
                f = fidelity(now_in_state, in_state)
                # t = trace_distance(now_in_state, in_state)
                print(f'fidelity: {f}, trace distance: {0}')
                adv_num += 1
                f_list.append(f)
                t_list.append(0)
                f_sum += f
                t_sum += 0
                QEA_sum += now_ent_out - now_ent_in
                # todo save avd img
                break


if __name__ == '__main__':
    print(f'Dataset {conf.dataset}, Model: {conf.structure}, class_idx: {conf.class_idx}')
    print('Parameter info:')
    print(f'test img num: {test_img_num}, lr: {lr}, anti_predict_weight: {anti_predict_weight}, cov_weight: {cov_weight}, '
          f'ent_k: {ent_k}')
    gen_adv()
