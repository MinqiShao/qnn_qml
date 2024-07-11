import torch
from pennylane import numpy as np
from torch import optim


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


def FGSM(model, imgs, labels, eps=8/255):
    """
    Fast Gradient Sign Method, a white-box single-step constraint-based method (untargeted)
    :param label: (N,)
    :param img: shape: (N, C, H, W)
    :param model:
    :param eps: max perturbation range epsilon
    :return: torch.tensor adv images within [0, 1]
    """
    imgs = imgs.clone().detach().requires_grad_(True)
    loss = model(imgs, labels)
    loss.backward()
    grad = imgs.grad.data.sign()
    imgs = torch.clamp((imgs + grad * eps), 0, 1)
    return imgs.detach()


def BIM(model, imgs, labels, eps=8/255, alpha=2/255, steps=10):
    """
    BIM, iterative-FGSM
    distance measure: Linf
    :param model:
    :param img:
    :param label: 步长
    :param eps: adv与ori img之间的变化范围
    :param steps:
    :return:
    """
    ori_imgs = imgs.clone().detach()
    for _ in range(steps):
        imgs.requires_grad = True
        loss = model(imgs, labels)
        loss.backward()
        # grad = torch.autograd.grad(
        #     loss, imgs, retain_graph=False, create_graph=False
        # )[0]
        grad = imgs.grad
        adv_imgs = imgs + alpha*grad.sign()
        a = torch.clamp(ori_imgs - eps, min=0)  # 每个像素位的最低值，变化的下限
        b = (adv_imgs >= a).float() * adv_imgs + (adv_imgs < a).float() * a
        c = (b > ori_imgs + eps).float() * (ori_imgs + eps) + (b<=ori_imgs+eps).float() * b  # 保证变化的上限
        imgs = torch.clamp(c, 0, 1).detach()
    return imgs


def CW(model, imgs, labels, c=1, kappa=0, steps=50, lr=0.01):
    """
    CW
    distance measure: L2
    :param model:
    :param img:
    :param label:
    :param c: for box-constraint
    :param kappa: also written as confidence
    :param steps:
    :param lr:
    :return:
    """
    imgs = imgs.clone().detach().requires_grad_(True)
    w = inverse_tanh_space(imgs).detach()
    w.requires_grad = True

    best_adv_imgs = imgs.clone()
    best_l2 = 1e10 * torch.ones((1,))
    prev_cost = 1e10
    dim = len(imgs.shape)

    MSELoss = torch.nn.MSELoss(reduction='none')
    Flatten = torch.nn.Flatten()
    opt = optim.Adam([w], lr=lr)

    for step in range(steps):
        adv_imgs = tanh_space(w)

        current_L2 = MSELoss(Flatten(adv_imgs), Flatten(imgs)).sum(dim=1)
        L2_loss = current_L2.sum()

        outputs = model.predict(adv_imgs)
        f_loss = f(outputs, labels, kappa).sum()

        cost = L2_loss + c * f_loss

        opt.zero_grad()
        cost.backward()
        opt.step()

        pre = torch.argmax(outputs, dim=1)
        condition = (pre != labels).float()

        # filter imgs that either get correct predictions or non-decreasing loss
        mask = condition * (best_l2>current_L2.detach())
        best_l2 = mask * current_L2.detach() + (1-mask) * best_l2
        mask = mask.view([-1] + [1] * (dim-1))
        best_adv_imgs = mask * adv_imgs.detach() + (1-mask) * best_adv_imgs

        # early stop when loss not converge
        if step % max(steps//10, 1) == 0:
            if cost.item() > prev_cost:
                return best_adv_imgs
            prev_cost = cost.item()
    return best_adv_imgs

def tanh_space(x):
    return 1/2 * (torch.tanh(x)+1)
def inverse_tanh_space(x):
    x = torch.clamp(x*2-1, min=-1, max=1)
    return 0.5 * torch.log((1+x)/(1-x))
def f(outputs, labels, kappa):
    one_hot_labels = torch.eye(outputs.shape[1])[labels]

    # find the max logit other than the target class
    other = torch.max((1 - one_hot_labels) * outputs, dim=1)[0]
    # get the target class's logit
    real = torch.max(one_hot_labels * outputs, dim=1)[0]
    return torch.clamp((real - other), min=-kappa)
