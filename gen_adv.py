"""
generate adversarial examples under constraints
"""
import torch
import os
from sklearn.metrics import accuracy_score
from torchvision.utils import save_image
from config import *
from tools.log import Log
from tools.data_loader import load_part_data
from tools.model_loader import load_params_from_path
from tools.adv_attack import *


conf = get_arguments()
device = torch.device('cpu')

p_c = os.path.join(conf.analysis_dir, 'AdvAttack', conf.dataset, conf.structure, str(conf.class_idx), conf.attack)
p_q = os.path.join(conf.analysis_dir, 'AdvAttack', conf.dataset, 'qcl', str(conf.class_idx), conf.attack)
if not os.path.exists(p_c):
    os.makedirs(p_c)
if not os.path.exists(p_q):
    os.makedirs(p_q)

def gen_adv_classical():
    test_x, test_y = load_part_data(conf, num_data=conf.num_test)
    params, model = load_params_from_path(conf, device)

    print(f'Dataset: {conf.dataset}, model: {conf.structure}, attack: {conf.attack}')
    if conf.attack == 'FGSM':
        adv_imgs = FGSM(model, test_x, test_y, eps=64/255)
    elif conf.attack == 'BIM':
        adv_imgs = BIM(model, test_x, test_y, eps=32/255, alpha=4/255, steps=100)

    now_y_c = model.predict(adv_imgs).detach().numpy().argmax(axis=1).tolist()
    now_acc = accuracy_score(test_y.numpy().tolist(), now_y_c)
    print(f'classical accuracy of adv imgs: {now_acc*100}%')

    # todo qnn
    conf.structure = 'qcl'
    params, model = load_params_from_path(conf, device)
    # qnn predict
    now_y_q = model.predict(adv_imgs).detach().numpy().argmax(axis=1).tolist()
    now_acc = accuracy_score(test_y.numpy().tolist(), now_y_q)
    print(f'qcl accuracy of adv imgs: {now_acc * 100}%')

    gen_num_c = 0
    gen_num_q = 0
    for i, a in enumerate(adv_imgs):
        if now_y_c[i] != test_y[i]:
            gen_num_c += 1
        save_image(a.detach(), os.path.join(p_c, str(i) + '_' + str(test_y[i].item()) + '_' + str(now_y_c[i]) + '.png'))
        if now_y_q[i] != test_y[i]:
            gen_num_q += 1
        save_image(a.detach(), os.path.join(p_q, str(i) + '_' + str(test_y[i].item()) + '_' + str(now_y_q[i]) + '.png'))


gen_adv_classical()
