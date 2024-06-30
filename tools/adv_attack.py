import torch
from pennylane import numpy as np


def DLFuzz2(now_outputs, ori_outputs, w):
    """
    2分类
    :param now_outputs: now outputs of QNN
    :param ori_outputs: original outputs of QNN, (2,)
    :param w: weight to anti
    :return: the decision boundary orientation
    """
    # probability of ori class in now_outputs
    o = torch.tensor(ori_outputs)
    loss1 = now_outputs[torch.argmax(o)]
    # probability of another class in now_outputs
    loss2 = now_outputs[torch.argsort(o)[-2]]
    return w*loss2 - loss1


def DLFuzz3(now_outputs, ori_outputs, w):
    o = torch.tensor(ori_outputs)
    args_ori = torch.argsort(o)
    loss1 = now_outputs[torch.argmax(o)]
    loss2 = now_outputs[args_ori[-2]]
    loss3 = now_outputs[args_ori[-3]]
    return w*(loss2+loss3) - loss1
