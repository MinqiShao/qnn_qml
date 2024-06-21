import os
from tqdm import tqdm
from tools.model_loader import load_model_from_path
from tools.data_loader import load_test_data
from config import *
from tools.entanglement import *
from models.circuits import weight_dict
from models.pure import QCL, QCNN_pure, CCQC, single_encoding, multi_encoding
from tools import Log

conf = get_arguments()
device = torch.device('cpu')


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
        if conf.structure == 'qcl':
            in_dm_list = QCL.get_density_matrix(x, params, exec_=False)
            out_dm_list = QCL.get_density_matrix(x, params)
        elif conf.structure == 'pure_qcnn':
            in_dm_list = QCNN_pure.get_density_matrix(x, params[0], params[1], params[2], params[3], params[4], exec_=False)
            out_dm_list = QCNN_pure.get_density_matrix(x, params[0], params[1], params[2], params[3], params[4])
        elif conf.structure == 'ccqc':
            in_dm_list = CCQC.get_density_matrix(x, params[0], params[1], params[2], exec_=False)
            out_dm_list = CCQC.get_density_matrix(x, params[0], params[1], params[2])
        in_list.append(Meyer_Wallach(in_dm_list))
        out_list.append(Meyer_Wallach(out_dm_list))
    in_list = torch.tensor(in_list)
    out_list = torch.tensor(out_list)
    return in_list, out_list, out_list-in_list


def entropy(test_x, params):
    in_list = []
    out_list = []
    for x in tqdm(test_x):
        x = torch.flatten(x, start_dim=0)
        if conf.structure == 'qcl':
            in_dm, out_dm = QCL.whole_dm(x, params)
        elif conf.structure == 'pure_qcnn':
            in_dm, out_dm = QCNN_pure.whole_dm(x, params[0], params[1], params[2], params[3], params[4])
        elif conf.structure == 'ccqc':
            in_dm, out_dm = CCQC.whole_dm(x, params[0], params[1], params[2])
        in_en, out_en = entanglement_entropy(in_dm), entanglement_entropy(out_dm)
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
        if conf.structure == 'qcl':
            in_dm, out_dm = QCL.whole_dm(x, params)
        elif conf.structure == 'pure_qcnn':
            in_dm, out_dm = QCNN_pure.whole_dm(x, params[0], params[1], params[2], params[3], params[4])
        elif conf.structure == 'ccqc':
            in_dm, out_dm = CCQC.whole_dm(x, params[0], params[1], params[2])
        in_en, out_en = avg_negativity(in_dm), avg_negativity(out_dm)
        in_list.append(in_en)
        out_list.append(out_en)
    in_list = torch.tensor(in_list)
    out_list = torch.tensor(out_list)
    return in_list, out_list, out_list - in_list


analyse_qinfo(c_n='entropy')