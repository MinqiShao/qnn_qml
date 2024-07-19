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

from tools import Log

conf = get_arguments()
device = torch.device('cpu')


def visualize_circuit():
    test_x, test_y = load_part_data(conf, num_data=conf.num_test)
    test_x, test_y = test_x[:1], test_y[:1]
    test_x = torch.flatten(test_x, start_dim=1)

    fig_save_path = os.path.join(conf.visual_dir, 'circuits')
    if not os.path.exists(fig_save_path):
        os.makedirs(fig_save_path)
    fig_save_path = os.path.join(fig_save_path, conf.structure + '.png')

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


def compare_adv():
    print('Compare entanglement between original, classical adv and QuanTest samples')
    print(f'dataset: {conf.dataset}, model: {conf.structure}, class: {conf.class_idx}, adv: {conf.attack}')
    # qcl
    params, model = load_params_from_path(conf, device)
    # classical adv data
    p = os.path.join(conf.analysis_dir, 'AdvAttack', conf.dataset, conf.structure, str(conf.class_idx), conf.attack)
    idx_list = []
    img_list = []
    transform = transforms.Compose([transforms.ToTensor()])
    for file in os.listdir(p):
        if file.endswith('.png'):
            idx = int(file.split('_')[0])
            idx_list.append(idx)  # img idx
            img_p = os.path.join(p, file)
            img = Image.open(img_p).convert('L')
            img = transform(img)
            img_list.append(img)
    idx_list = torch.tensor(idx_list)

    # original data
    test_x, test_y = load_part_data(conf, num_data=conf.num_test)

    # QuanTest imgs
    p = os.path.join(conf.analysis_dir, 'QuanTest', conf.dataset, conf.structure, str(conf.class_idx))
    quan_list = []
    quan_idx_list = []
    transform = transforms.Compose([transforms.ToTensor()])
    for file in os.listdir(p):
        if file.endswith('.png'):
            idx = int(file.split('_')[0])
            quan_idx_list.append(idx)  # img idx
            img_p = os.path.join(p, file)
            img = Image.open(img_p).convert('L')
            img = transform(img)
            quan_list.append(img)
    quan_idx_list = torch.tensor(quan_idx_list)

    common_idx = []
    ori_img = []
    c_img = []
    q_img = []
    for i, idx in enumerate(idx_list):
        a = torch.where(quan_idx_list == idx)[0]
        if a.shape[0] == 0:
            continue
        common_idx.append(idx)
        ori_img.append(test_x[idx])
        c_img.append(img_list[i])
        q_img.append(quan_list[a])

    # for all imgs in img_list (they success attack classical)
    d = depth_dict[conf.structure]
    in_o, out_o, chan_o = MW(ori_img, params, conf, d)
    in_c, out_c, chan_c = MW(c_img, params, conf, d)
    in_q, out_q, chan_q = MW(q_img, params, conf, d)
    # for i in range(len(common_idx)):
    #     print(f'---{common_idx[i]} img---')
    #     print(f'ori in: {in_o[i]}, adv in: {in_c[i]}, QuanTest: {in_q[i]}')
    #     print(f'ori out: {out_o[i]}, adv out: {out_c[i]}, QuanTest: {out_q[i]}')

    # for imgs that attack classical and qnn successfully
    now_y_q = model.predict(torch.stack(c_img)).detach().argmax(axis=1)
    q = torch.where(test_y[torch.tensor(common_idx)] != now_y_q)[0]
    print(f'{q.shape[0]}/{len(common_idx)} adv imgs attacking qnn successfully')
    for i in q:
        print(f'---{common_idx[i]} img---')
        print(f'ori in: {in_o[i]}, adv in: {in_c[i]}, QuanTest: {in_q[i]}')
        print(f'ori out: {out_o[i]}, adv out: {out_c[i]}, QuanTest: {out_q[i]}')
        print(f'ori chan: {chan_o[i]}, adv chan: {chan_c[i]}, QuanTest: {chan_q[i]}')



# analyse_qinfo(c_n='MW')
visualize_circuit()
#compare_adv()