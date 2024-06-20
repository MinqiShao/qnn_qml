from tqdm import tqdm
from tools.model_loader import load_model_from_path
from tools.data_loader import load_test_data
from config import *
from tools.entanglement import *
from models.circuits import weight_dict
from models.pure import QCL

conf = get_arguments()
device = torch.device('cpu')


def analyse_qinfo(c_n='MW'):
    print(f'dataset: {conf.dataset}, model: {conf.structure}, class: {conf.class_idx}')
    model = load_model_from_path(conf=conf, device=device)
    if len(conf.class_idx) > 2:
        bi = False
    else:
        bi = True
    test_x, test_y = load_test_data(conf, bi)

    state_dict = model.state_dict()
    weight_name = weight_dict[conf.structure]
    if type(weight_name) is list:
        params = []
        for i in range(len(weight_name)):
            params.append(state_dict[weight_name[i]])
    else:
        params = state_dict[weight_name]

    if c_n == 'MW':
        MW(test_x, params)
    elif c_n == 'entropy':
        entropy(test_x, params)
    elif c_n == 'negativity':
        negativity(test_x, params)


def MW(test_x, params):
    in_list = []
    out_list = []
    for x in tqdm(test_x):
        x = torch.flatten(x, start_dim=0)
        if conf.structure == 'qcl':
            in_dm_list = QCL.in_density_matrices(x, params)
            out_dm_list = QCL.out_density_matrices(x, params)
        in_list.append(Meyer_Wallach(in_dm_list))
        out_list.append(Meyer_Wallach(out_dm_list))
    en_list = torch.tensor(in_list)
    print(torch.mean(en_list), torch.min(en_list), torch.max(en_list))
    en_list = torch.tensor(out_list)
    print(torch.mean(en_list), torch.min(en_list), torch.max(en_list))


def entropy(test_x, params):
    in_list = []
    out_list = []
    for x in tqdm(test_x):
        x = torch.flatten(x, start_dim=0)
        if conf.structure == 'qcl':
            in_dm, out_dm = QCL.whole_dm(x, params)
            in_en, out_en = entanglement_entropy(in_dm), entanglement_entropy(out_dm)
            in_list.append(in_en)
            out_list.append(out_en)
    en_list = torch.tensor(in_list)
    print(torch.mean(en_list), torch.min(en_list), torch.max(en_list))
    en_list = torch.tensor(out_list)
    print(torch.mean(en_list), torch.min(en_list), torch.max(en_list))


def negativity(test_x, params):
    in_list = []
    out_list = []
    for x in tqdm(test_x):
        x = torch.flatten(x, start_dim=0)
        if conf.structure == 'qcl':
            in_dm, out_dm = QCL.whole_dm(x, params)
        in_en, out_en = avg_negativity(in_dm), avg_negativity(out_dm)
        in_list.append(in_en)
        out_list.append(out_en)
    ne_list = torch.tensor(in_list)
    print(torch.mean(ne_list), torch.min(ne_list), torch.max(ne_list))
    ne_list = torch.tensor(out_list)
    print(torch.mean(ne_list), torch.min(ne_list), torch.max(ne_list))


analyse_qinfo('entropy')