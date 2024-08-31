import os
import pennylane as qml
import torch
from torchvision import transforms
from PIL import Image
from tools.model_loader import load_params_from_path
from tools.data_loader import load_part_data, load_adv_imgs
from models.circuits import depth_dict
from config import *
from tools.entanglement import *
from tools.internal import block_out

from tools import Log

conf = get_arguments()
device = torch.device('cpu')


def visualize_circuit():
    test_x, test_y = load_part_data(conf, num_data=conf.num_test)
    test_x, test_y = test_x[:1], test_y[:1]
    test_x = torch.flatten(test_x, start_dim=0)

    fig_save_path = os.path.join(conf.visual_dir, 'circuits')
    if not os.path.exists(fig_save_path):
        os.makedirs(fig_save_path)
    model_n = conf.structure
    if conf.structure == 'hier':
        model_n = conf.structure + '_' + conf.hier_u
    fig_save_path = os.path.join(fig_save_path, model_n + '.png')

    params, model = load_params_from_path(conf, device)
    model.visualize_circuit(test_x, params, fig_save_path)


def analyse_qinfo(c_n='MW'):
    print(f'dataset: {conf.dataset}, model: {conf.structure}, class: {conf.class_idx}')
    test_x, test_y = load_part_data(conf, num_data=conf.num_test)
    params, _ = load_params_from_path(conf, device)
    d = depth_dict[conf.structure]

    if c_n == 'MW':
        in_list, out_list, chan_list = MW(test_x, params, conf, d)
    elif c_n == 'entropy':
        in_list, out_list, chan_list = entropy(test_x, params, conf)
    elif c_n == 'neg':
        in_list, out_list, chan_list = neg(test_x, params, conf)

    log_dir = os.path.join(conf.analysis_dir, conf.dataset, conf.version, conf.structure)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if conf.resize:
        log = Log(os.path.join(log_dir, 'log_' + conf.reduction + '_' + str(conf.class_idx) + '.txt'))
    else:
        log = Log(os.path.join(log_dir, 'log_' + str(conf.class_idx) + '.txt'))
    log(f'{c_n} results for {conf.dataset}+{conf.structure}, class_idx: {conf.class_idx}')
    # log('Input state: mean: {}, min: {}, max: {}'.format(torch.mean(in_list), torch.min(in_list), torch.max(in_list)))
    # log('Output state: mean: {}, min: {}, max: {}'.format(torch.mean(out_list), torch.min(out_list), torch.max(out_list)))
    # log('Change between in and out: mean: {}, min: {}, max: {}'.format(torch.mean(chan_list), torch.min(chan_list), torch.max(chan_list)))
    log('Input state: {}'.format(in_list.tolist()))
    log('Output state: {}'.format(out_list.tolist()))


def compare_adv(t='ent'):
    print('Compare between original, classical adv and QuanTest samples')
    print(f'dataset: {conf.dataset}, model: {conf.structure}, class: {conf.class_idx}, adv: {conf.attack}')
    params, model = load_params_from_path(conf, device)
    # classical adv data
    idx_list, img_list = load_adv_imgs(conf)

    # original data
    test_x, test_y = load_part_data(conf, num_data=conf.num_test)

    # QuanTest imgs
    conf.attack = 'QuanTest'
    quan_idx_list, quan_list = load_adv_imgs(conf)

    common_idx = []
    ori_img = []
    c_img = []
    q_img = []
    for i, idx in enumerate(idx_list):
        a = torch.where(quan_idx_list == idx)[0]
        # if a.shape[0] == 0:
        #     continue
        common_idx.append(idx)
        ori_img.append(test_x[idx])
        # c_img.append(img_list[i])
        # q_img.append(quan_list[a])

    d = depth_dict[conf.structure]
    if t == 'ent':
        # 比较ori、adv、QuanTest的纠缠及纠缠差
        in_o, out_o, chan_o = MW(ori_img, params, conf, d)
        # in_c, out_c, chan_c = MW(c_img, params, conf, d)
        # in_q, out_q, chan_q = MW(q_img, params, conf, d)

        # for imgs that attack classical and qnn successfully
        # now_y_q = model.predict(torch.stack(c_img)).detach().argmax(axis=1)
        # q = torch.where(test_y[torch.tensor(common_idx)] != now_y_q)[0]
        # print(f'{q.shape[0]}/{len(common_idx)} adv imgs attacking qnn successfully')
        for i, idx in enumerate(common_idx):
            # print(f'---{idx} img---')
            # print(f'ori in: {in_o[i]}, adv in: {in_c[i]}, QuanTest: {in_q[i]}')
            # print(f'ori out: {out_o[i]}, adv out: {out_c[i]}, QuanTest: {out_q[i]}')
            # print(f'ori chan(%): {chan_o[i]/in_o[i]}, adv chan: {chan_c[i]/in_o[i]}, QuanTest: {chan_q[i]/in_o[i]}')
            print(f'ori in: {round(in_o[i].item(), 5)}, ori out: {round(out_o[i].item(), 5)}, '
                  f'ori chan(%): {round((chan_o[i] / in_o[i]).item(), 5)*100}')

    if t == 'init_s':
        # 比较ori和adv样本在encoding后的有效基态（prob不为0）的数量对比
        for o, c, q in zip(ori_img, c_img, q_img):
            ori_state = block_out(torch.flatten(o), conf, params, d, exec_=False)
            adv_state = block_out(torch.flatten(c), conf, params, d, exec_=False)
            q_state = block_out(torch.flatten(q), conf, params, d, exec_=False)
            o_n = torch.sum(ori_state > 1e-16).item()
            c_n = torch.sum(adv_state > 1e-16).item()
            q_n = torch.sum(q_state > 1e-16).item()
            print(f'{o_n}/{c_n}/{q_n}, total: {ori_state.shape[0]}')



# analyse_qinfo(c_n='MW')
#visualize_circuit()
compare_adv(t='ent')