import os
import pennylane as qml
import torch
from tqdm import tqdm
from tools.model_loader import load_model_from_path
from tools.data_loader import load_test_data
from config import *
from tools.entanglement import *
from models.circuits import weight_dict
from tools.internal import *
from tools import Log

conf = get_arguments()
device = torch.device('cpu')


def visualize_circuit():
    model = load_model_from_path(conf=conf, device=device)
    test_x, test_y = load_test_data(conf)
    test_x, test_y = test_x[:1], test_y[:1]
    test_x = torch.flatten(test_x, start_dim=1)
    state_dict = model.state_dict()
    weight_name = weight_dict[conf.structure]
    if type(weight_name) is list:
        params = []
        for i in range(len(weight_name)):
            params.append(state_dict[weight_name[i]])
    else:
        params = state_dict[weight_name]

    fig_save_path = os.path.join(conf.visual_dir, 'circuits')
    if not os.path.exists(fig_save_path):
        os.makedirs(fig_save_path)
    fig_save_path = os.path.join(fig_save_path, conf.structure + '.png')
    model.visualize_circuit(test_x, params, fig_save_path)


def analyse_qinfo(c_n='MW'):
    print(f'dataset: {conf.dataset}, model: {conf.structure}, class: {conf.class_idx}')
    model = load_model_from_path(conf=conf, device=device)
    test_x, test_y = load_test_data(conf)

    state_dict = model.state_dict()
    weight_name = weight_dict[conf.structure]
    if type(weight_name) is list:
        params = []
        for i in range(len(weight_name)):
            params.append(state_dict[weight_name[i]])
    else:
        params = state_dict[weight_name]

    if c_n == 'MW':
        in_list, out_list, chan_list = MW(test_x, params)
    elif c_n == 'entropy':
        in_list, out_list, chan_list = entropy(test_x, params)
    elif c_n == 'negativity':
        in_list, out_list, chan_list = negativity(test_x, params)

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


def MW(test_x, params):
    in_list = []
    out_list = []
    for x in tqdm(test_x):
        x = torch.flatten(x, start_dim=0)
        in_state, out_state = in_out_state(x, conf.structure, params)
        _, ent_in, ent_out = entQ(in_state, out_state, 1)
        in_list.append(in_state)
        out_list.append(out_state)
    in_list = torch.tensor(in_list)
    out_list = torch.tensor(out_list)
    return in_list, out_list, out_list-in_list


def entropy(test_x, params):
    in_list = []
    out_list = []
    class_idx = conf.class_idx
    for x in tqdm(test_x):
        x = torch.flatten(x, start_dim=0)
        in_dm, out_dm = whole_density_matrix(x, conf.structure, params)
        in_en, out_en = entanglement_entropy(in_dm, class_idx), entanglement_entropy(out_dm, class_idx)
        in_list.append(in_en)
        out_list.append(out_en)
    in_list = torch.tensor(in_list)
    out_list = torch.tensor(out_list)
    return in_list, out_list, out_list-in_list


def negativity(test_x, params):
    in_list = []
    out_list = []
    for x in tqdm(test_x):
        x = torch.flatten(x, start_dim=0)
        in_state, out_state = in_out_state(x, conf.structure, params)
        in_en, out_en = avg_negativity(in_state), avg_negativity(out_state)
        in_list.append(in_en)
        out_list.append(out_en)
    in_list = torch.tensor(in_list)
    out_list = torch.tensor(out_list)
    return in_list, out_list, out_list - in_list


analyse_qinfo(c_n='MW')
# visualize_circuit()