from models import *
from tq_models import *
from models.circuits import weight_dict
import torch
import os


def load_model(conf, device, data_size=28, e_type='amplitude'):
    model_type = conf.structure
    class_idx = conf.class_idx
    print(f'model: {model_type} of {conf.version}')
    num_classes = len(class_idx)

    if conf.version == 'qml':
        if model_type == 'classical':
            model = Classical(num_classes=num_classes)
        elif model_type == 'pure_single':
            model = SingleEncoding(num_classes=num_classes, img_size=data_size)
        elif model_type == 'pure_multi':
            model = MultiEncoding(num_classes=num_classes, img_size=data_size)
        elif model_type == 'inception':
            model = InceptionNet(num_classes=num_classes)
        elif model_type == 'hier':
            assert num_classes == 2
            model = Hierarchical(embedding_type=e_type, u=conf.hier_u)
        elif model_type == 'hier_qcnn':
            assert num_classes == 2
            model = QCNN_classifier(e=e_type, u=conf.hier_u)
        elif model_type == 'qcl':
            model = QCL_classifier(num_classes=num_classes)
        elif model_type == 'ccqc':
            model = CCQC_classifier(e=e_type, num_classes=num_classes)
        elif model_type == 'pure_qcnn':
            model = QCNN_c(num_classes=num_classes)
    elif conf.version == 'tq':
        if model_type == 'pure_single':
            model = SingleEncoding_(device=device, num_classes=num_classes, img_size=data_size)
        elif model_type == 'pure_multi':
            model = MultiEncoding_(device=device, num_classes=num_classes, img_size=data_size)
        elif model_type == 'ccqc':
            model = CCQC_(device=device)
        elif model_type == 'qcl':
            model = QCL_(device=device)
        elif model_type == 'pure_qcnn':
            model = QCNN_(device=device, num_classes=num_classes)
    return model


def load_params_from_path(conf, device):
    model_n = conf.structure
    if conf.structure == 'hier' or conf.structure == 'hier_qcnn':
        model_n = model_n + '_' + conf.hier_u
    if conf.resize:
        mode_path = os.path.join(conf.model_dir, conf.dataset, conf.version, model_n,
                                 conf.reduction + '_' + str(conf.class_idx) + '.pth')
    else:
        mode_path = os.path.join(conf.model_dir, conf.dataset, conf.version, model_n,
                                 str(conf.class_idx) + '.pth')

    print(f'load model from: {mode_path}...')
    model = load_model(conf=conf,
                       device=device, data_size=28, e_type=conf.encoding)
    model.load_state_dict(torch.load(mode_path))
    model.eval()
    state_dict = model.state_dict()

    weight_name = weight_dict[conf.structure]
    if type(weight_name) is list:
        params = []
        for i in range(len(weight_name)):
            params.append(state_dict[weight_name[i]])
    else:
        params = state_dict[weight_name]
    return params, model


def load_train_params(structure, state_dict):
    weight_name = weight_dict[structure]
    if type(weight_name) is list:
        params = []
        for i in range(len(weight_name)):
            params.append(state_dict[weight_name[i]])
    else:
        params = state_dict[weight_name]
    return params
