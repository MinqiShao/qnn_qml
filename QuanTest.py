import torch
import os
from config import *
from torchvision.utils import save_image
from tools.entanglement import *
from tools.data_loader import load_part_data
from tools.model_loader import load_params_from_path
from tools.adv_attack import *
from models.circuits import depth_dict
from pennylane.math import fidelity, trace_distance, reduce_statevector
from pennylane import numpy as np, AmplitudeEmbedding
import time
from tools.log import Log


conf = get_arguments()
device = torch.device('cpu')

d = depth_dict[conf.structure]
test_img_num = conf.num_test
lr = 0.05
anti_predict_weight = 1
cov_weight = 1
ent_k = 1


def gen_adv():
    test_x, test_y = load_part_data(conf, num_data=conf.num_test)
    params, _ = load_params_from_path(conf, device)

    adv_num = 0
    f_list = []
    f_sum = 0
    t_list = []
    t_sum = 0
    QEA_sum = 0

    p = os.path.join(conf.analysis_dir, 'QuanTest', conf.dataset, conf.structure, str(conf.class_idx))
    if not os.path.exists(p):
        os.makedirs(p)
    p_ = os.path.join(p, 'log.txt')
    log = Log(p_)

    for i in range(test_x.shape[0]):
        log(f'Start for {i}th img...')
        x = torch.flatten(test_x[i], start_dim=0)
        x.requires_grad_(True)

        in_state, out_state = in_out_state(x, conf.structure, params, d)
        ori_outputs, _ = circuit_pred(x, params, conf)
        ori_outputs = torch.tensor(ori_outputs)

        iters = 0
        while True:
            iters += 1
            x.requires_grad_(True)

            now_in_state, now_out_state = in_out_state(x, conf.structure, params, d)
            now_ent_c, now_ent_in, now_ent_out = entQ(now_in_state, now_out_state, ent_k)
            log(f'iter {iters}: QEA: {now_ent_c}, ent_in: {now_ent_in}, ent_out: {now_ent_out}')

            now_outputs, _ = circuit_pred(x, params, conf)  # list
            # now_out_state = circuit_state(x, conf, params)  # tensor
            now_ent_c, _, _ = entQ(now_in_state, now_out_state, ent_k)
            if len(conf.class_idx) == 2:
                obj_orie = DLFuzz2(now_outputs, ori_outputs, anti_predict_weight)
            else:
                obj_orie = DLFuzz3(now_outputs, ori_outputs, anti_predict_weight)

            loss = obj_orie + cov_weight * now_ent_c
            loss.backward()
            perturb = x.grad * lr

            x = torch.clamp((x+perturb), 0, 1).detach()

            _, new_y = circuit_pred(x, params, conf)
            if new_y != test_y[i]:
                log('gen an adv img!')
                log(f'ori_y: {test_y[i]}, new_y: {new_y}')
                idx = list(range(0, int(math.log2(now_in_state.shape[0]))))
                f = fidelity(reduce_statevector(now_in_state, indices=idx), reduce_statevector(in_state, indices=idx))
                t = trace_distance(reduce_statevector(now_in_state, indices=idx), reduce_statevector(in_state, indices=idx))
                log(f'fidelity: {f}, trace distance: {t}')
                adv_num += 1
                f_list.append(f)
                t_list.append(t)
                f_sum += f
                t_sum += t
                QEA_sum += now_ent_out - now_ent_in
                adv_img = x.reshape(1, 28, 28)
                save_image(adv_img, os.path.join(p,   str(i) + '_' + str(test_y[i].item()) + '_' + str(new_y.item()) + '.png'))
                break

            if iters == 500:
                break
    log(f'!!generated {adv_num} adv img out of {test_x.shape[0]} img!')


if __name__ == '__main__':
    print(f'Dataset {conf.dataset}, Model: {conf.structure}, class_idx: {conf.class_idx}')
    print('Parameter info:')
    print(f'test img num: {test_img_num}, lr: {lr}, anti_predict_weight: {anti_predict_weight}, cov_weight: {cov_weight}, '
          f'ent_k: {ent_k}')
    gen_adv()
