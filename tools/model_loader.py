from models import *
from tq_models import *
import torch
import os


def load_model(v, model_type, class_idx, device, data_size=28, e_type='amplitude'):
    print(f'!!loading model {model_type} of {v}')
    num_classes = len(class_idx)

    if v == 'qml':
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
            model = Hierarchical(embedding_type=e_type)
        elif model_type == 'hier_qcnn':
            assert num_classes == 2
            model = QCNN_classifier(e=e_type)
        elif model_type == 'qcl':
            model = QCL_classifier(num_classes=num_classes)
        elif model_type == 'ccqc':
            model = CCQC_classifier(e=e_type, num_classes=num_classes)
        elif model_type == 'pure_qcnn':
            model = QCNN_c()
    elif v == 'tq':
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


def load_model_from_path(conf, device):
    if conf.resize:
        mode_path = os.path.join(conf.result_dir, conf.dataset, conf.version, conf.structure,
                                 conf.reduction + '_' + str(conf.class_idx) + '.pth')
    else:
        mode_path = os.path.join(conf.result_dir, conf.dataset, conf.version, conf.structure,
                                 str(conf.class_idx) + '.pth')
    print(f'load model from: {mode_path}...')
    model = load_model(v=conf.version, model_type=conf.structure, class_idx=conf.class_idx,
                                 device=device, data_size=28, e_type=conf.encoding)
    model.load_state_dict(torch.load(mode_path))
    model.eval()
    return model
