import os
import pennylane as qml
import torch
from torchvision import transforms
from PIL import Image
from tools.model_loader import load_params_from_path
from tools.data_loader import load_part_data
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
    # model.visualize_circuit(test_x, params, fig_save_path)


def analyse_qinfo(c_n='MW'):
    print(f'dataset: {conf.dataset}, model: {conf.structure}, class: {conf.class_idx}')
    test_x, test_y = load_part_data(conf, num_data=conf.num_test)
    params, _ = load_params_from_path(conf, device)

    if c_n == 'MW':
        in_list, out_list, chan_list = MW(test_x, params, conf)
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
    log('Input state: {}'.format(in_list))
    log('Output state: {}'.format(out_list))


def adv_qinfo():
    print(f'dataset: {conf.dataset}, model: {conf.structure}, class: {conf.class_idx}, attack: {conf.attack}')
    if conf.attack == 'QuanTest':
        p = os.path.join(conf.analysis_dir, 'QuanTest', conf.dataset, conf.structure, str(conf.class_idx))
    else:
        p = os.path.join(conf.analysis_dir, 'AdvAttack', conf.dataset, 'qcl', str(conf.class_idx), conf.attack)

    img_list = []
    transform = transforms.Compose([transforms.ToTensor()])
    for file in os.listdir(p):
        if file.endswith('.png'):
            print(file)
            img_p = os.path.join(p, file)
            img = Image.open(img_p).convert('L')
            img = transform(img)
            img_list.append(img.unsqueeze(0))

    # qcl
    params, model = load_params_from_path(conf, device)
    in_list, out_list, chan_list = MW(img_list, params, conf)

    for i, a in enumerate(img_list):
        print(f'-------{i}th img--------')
        print(f'pred label: {model.predict(a).detach().numpy().argmax(axis=1).tolist()}')
        print('in: {}, out: {}'.format(in_list[i], out_list[i]))

analyse_qinfo(c_n='MW')
# visualize_circuit()
adv_qinfo()