import os
import pennylane as qml
import torch
from tools.model_loader import load_params_from_path
from tools.data_loader import load_part_data
from config import *
from tools.entanglement import *

from tools import Log

conf = get_arguments()
device = torch.device('cpu')


def visualize_circuit():
    test_x, test_y = load_part_data(conf)
    test_x, test_y = test_x[:1], test_y[:1]
    test_x = torch.flatten(test_x, start_dim=1)


    fig_save_path = os.path.join(conf.visual_dir, 'circuits')
    if not os.path.exists(fig_save_path):
        os.makedirs(fig_save_path)
    fig_save_path = os.path.join(fig_save_path, conf.structure + '.png')
    # model.visualize_circuit(test_x, params, fig_save_path)


def analyse_qinfo(c_n='MW'):
    print(f'dataset: {conf.dataset}, model: {conf.structure}, class: {conf.class_idx}')
    test_x, test_y = load_part_data(conf)
    params = load_params_from_path(conf, device)

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
    log('Input state: mean: {}, min: {}, max: {}'.format(torch.mean(in_list), torch.min(in_list), torch.max(in_list)))
    log('Output state: mean: {}, min: {}, max: {}'.format(torch.mean(out_list), torch.min(out_list), torch.max(out_list)))
    log('Change between in and out: mean: {}, min: {}, max: {}'.format(torch.mean(chan_list), torch.min(chan_list), torch.max(chan_list)))



analyse_qinfo(c_n='MW')
# visualize_circuit()