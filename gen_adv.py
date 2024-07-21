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

model_n = conf.structure
if model_n == 'hier':
    model_n += '_' + conf.hier_u
p_c = os.path.join(conf.analysis_dir, 'AdvAttack', conf.dataset, model_n, str(conf.class_idx), conf.attack)
if not os.path.exists(p_c):
    os.makedirs(p_c)

def gen_adv():
    test_x, test_y = load_part_data(conf, num_data=conf.num_test)
    params, model = load_params_from_path(conf, device)

    print(f'Dataset: {conf.dataset}, model: {model_n}, attack: {conf.attack}')
    if conf.attack == 'FGSM':
        adv_imgs = FGSM(model, test_x, test_y, eps=64/255)
    elif conf.attack == 'BIM':
        adv_imgs = BIM(model, test_x, test_y, eps=64/255, alpha=4/255, steps=100)
    elif conf.attack == 'DLFuzz':
        adv_imgs = DLFuzz(model, test_x, test_y, w=5, steps=500)
    elif conf.attack == 'CW':
        adv_imgs = CW(model, test_x, test_y, c=10, steps=500)
    elif conf.attack == 'DIFGSM':
        adv_imgs = DIFGSM(model, test_x, test_y, eps=32/255, alpha=4/255, steps=100)
    elif conf.attack == 'JSMA':
        adv_imgs = JSMA(model, test_x, test_y, num_class=len(conf.class_idx))

    now_y = model.predict(adv_imgs).detach().numpy().argmax(axis=1).tolist()
    now_acc = accuracy_score(test_y.numpy().tolist(), now_y)
    print(f'accuracy of adv imgs: {now_acc*100}%')

    gen_num_c = 0
    for i, a in enumerate(adv_imgs):
        if now_y[i] != test_y[i]:
            gen_num_c += 1
            save_image(a.detach(), os.path.join(p_c, str(i) + '_' + str(test_y[i].item()) + '_' + str(now_y[i]) + '.png'))
    print(f'generate {gen_num_c} adv imgs!!')


gen_adv()
