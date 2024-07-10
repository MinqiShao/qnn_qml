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

p = os.path.join(conf.analysis_dir, 'AdvAttack', conf.dataset, conf.structure, str(conf.class_idx), conf.attack)
if not os.path.exists(p):
    os.makedirs(p)

def gen_adv():
    test_x, test_y = load_part_data(conf, num_data=conf.num_test)
    params, model = load_params_from_path(conf, device)

    print(f'Dataset: {conf.dataset}, model: {conf.structure}, attack: {conf.attack}')
    if conf.attack == 'FGSM':
        adv_imgs = FGSM(model, test_x, test_y, eps=32/255)
    elif conf.attack == 'BIM':
        adv_imgs = BIM(model, test_x, test_y, eps=32/255, alpha=4/255, steps=100)

    now_y = model.predict(adv_imgs).detach().numpy().argmax(axis=1).tolist()
    now_acc = accuracy_score(test_y.numpy().tolist(), now_y)
    print(f'accuracy of adv imgs: {now_acc*100}%')

    gen_num = 0
    for i, a in enumerate(adv_imgs):
        if now_y[i] != test_y[i]:
            gen_num += 1
            save_image(a.detach(), os.path.join(p, f'{i}.png'))
    print(f'have saved {gen_num} adv imgs to {p}')


gen_adv()
