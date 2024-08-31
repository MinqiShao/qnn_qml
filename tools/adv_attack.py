import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm


def random_noise(imgs, k=0.1):
    # noise = torch.rand(imgs.shape[2:])
    noise = torch.randn(imgs.shape[2:])
    norm_noise = (noise - noise.min()) / (noise.max() - noise.min())
    adv_imgs = torch.clamp(imgs + norm_noise * k, 0., 1.)
    return adv_imgs


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


def DLFuzz(model, imgs, labels, w=1, steps=500):
    imgs = imgs.clone()
    for i in tqdm(range(steps)):
        imgs.requires_grad = True
        outputs = model.predict(imgs)  # (num_sample, num_class)
        loss1 = torch.gather(outputs, 1, labels.view(-1, 1))
        loss2 = torch.sum(outputs, dim=1, keepdim=True) - loss1
        loss = torch.mean(w*loss2 - loss1)
        loss.backward()
        imgs = torch.clamp(imgs + imgs.grad, 0., 1.).detach()
    return imgs


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


def DIFGSM(model, imgs, labels, eps=8/255, alpha=2/255, steps=10, decay=0.0, resize_rate=0.9, diversity_prob=0.5, random_start=False):
    """
    DI2-FGSM 生成对抗样本之前，对原始样本随机变换，增加输入的多样性
    distance measure: Linf
    :param model:
    :param imgs:
    :param labels:
    :param eps:
    :param alpha: lr
    :param steps:
    :param decay:
    :param resize_rate: resize factor for input diversity
    :param diversity_prob: the probability of applying input diversity
    :return:
    """
    def input_diversity(x):
        img_size = x.shape[-1]
        img_resize = int(img_size * resize_rate)

        if resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int64)
        rescaled = F.interpolate(
            x, size=[rnd, rnd], mode="bilinear", align_corners=False
        )
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int64)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left
        padded = F.pad(
            rescaled,
            [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()],
            value=0,
        )
        return padded if torch.rand(1) < diversity_prob else x

    imgs = imgs.unsqueeze(1)
    momentum = torch.zeros_like(imgs)
    adv_imgs = imgs.clone()

    if random_start:
        adv_imgs = adv_imgs + torch.empty_like(adv_imgs).uniform_(-eps, eps)
        adv_imgs = torch.clamp(adv_imgs, 0, 1)

    for _ in tqdm(range(steps)):
        adv_imgs.requires_grad = True
        loss = model(input_diversity(adv_imgs), labels)
        loss.backward()
        grad = adv_imgs.grad.data
        grad /= torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
        grad += momentum * decay
        momentum = grad
        adv_imgs = adv_imgs.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_imgs - imgs, min=-eps, max=eps)
        adv_imgs = torch.clamp(imgs + delta, min=0, max=1).detach()
    return adv_imgs


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
    for _ in tqdm(range(steps)):
        imgs.requires_grad = True
        loss = model(imgs, labels)
        loss.backward()
        grad = imgs.grad
        adv_imgs = imgs + alpha*grad.sign()
        a = torch.clamp(ori_imgs - eps, min=0)  # 每个像素位的最低值，变化的下限
        b = (adv_imgs >= a).float() * adv_imgs + (adv_imgs < a).float() * a
        c = (b > ori_imgs + eps).float() * (ori_imgs + eps) + (b<=ori_imgs+eps).float() * b  # 保证变化的上限
        imgs = torch.clamp(c, 0, 1).detach()
    return imgs


def CW(model, imgs, labels, c=1, kappa=0, steps=50, lr=0.01):
    """
    CW 通过L2最小化扰动
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

    for step in tqdm(range(steps)):
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


def JSMA(model, imgs, labels, num_class=2, theta=1.0, gamma=0.1):
    """
    Jacobian Saliency Map Attack 通过雅可比矩阵分析输入特征的敏感度（saliency map），选择对模型影响最大的特征进行扰动
    :param model:
    :param ims:
    :param labels:
    :param theta: perturb length
    :param gamma: highest percentage of pixels can be modified
    :return:
    """
    def compute_jacobian(img):
        var_img = img.clone().detach()
        var_img.requires_grad = True
        output = model.predict(var_img)

        num_features = int(np.prod(var_img.shape[1:]))
        jacobian = torch.zeros([output.shape[1], num_features])
        for i in range(output.shape[1]):
            if var_img.grad is not None:
                var_img.grad.zero_()
            # i-th class logit
            output[0][i].backward(retain_graph=True)
            jacobian[i] = (var_img.grad.squeeze().view(-1, num_features).clone())
        return jacobian

    @torch.no_grad()
    def saliency_map(jacobian, target_label, increasing, search_space, nb_features):
        domain = torch.eq(search_space, 1).float()
        all_sum = torch.sum(jacobian, dim=0, keepdim=True)
        target_grad = jacobian[target_label]
        other_grad = all_sum - target_grad

        if increasing:  # blanks out those not in search domain
            increase_coef = 2 * (torch.eq(domain, 0)).float()
        else:
            increase_coef = -1 * 2 * (torch.eq(domain, 0))
        increase_coef = increase_coef.view(-1, nb_features)

        # sum of target toward derivative of any 2 features
        target_tmp = target_grad.clone()
        target_tmp -= increase_coef * torch.max(torch.abs(target_grad))
        alpha = target_tmp.view(-1, 1, nb_features) + target_tmp.view(-1, nb_features, 1)
        # the sum of other forward derivative of any 2 features
        other_tmp = other_grad.clone()
        other_tmp += increase_coef * torch.max(torch.abs(other_grad))
        beta = other_tmp.view(-1, 1, nb_features) + other_tmp.view(-1, nb_features, 1)

        # zero out the situation where a feature sums with itself
        tmp = np.ones((nb_features, nb_features), int)
        np.fill_diagonal(tmp, 0)
        zero_diagonal = torch.from_numpy(tmp).byte()

        if increasing:
            mask1 = torch.gt(alpha, 0.0)
            mask2 = torch.lt(beta, 0.0)
        else:
            mask1 = torch.lt(alpha, 0.0)
            mask2 = torch.gt(beta, 0.0)

        # mask to saliency map
        mask = torch.mul(torch.mul(mask1, mask2), zero_diagonal.view_as(mask1))
        saliency_map = torch.mul(torch.mul(alpha, torch.abs(beta)), mask.float())
        # the most significant two pixels
        max_idx = torch.argmax(saliency_map.view(-1, nb_features*nb_features), dim=1)
        p = torch.div(max_idx, nb_features, rounding_mode='floor')
        q = max_idx - p*nb_features
        return p, q
    def perturbation_single(img, target_label):
        var_img = img
        var_label = target_label
        if theta > 0:
            increasing = True
        else:
            increasing = False

        num_features = int(np.prod(var_img.shape[1:]))
        shape = var_img.shape
        max_iters = int(np.ceil(num_features * gamma / 2.0))  # perturb 2 pixels in 1 iter
        if increasing:
            search_domain = torch.lt(var_img, 0.99)
        else:
            search_domain = torch.gt(var_img, 0.01)
        search_domain = search_domain.view(num_features)
        output = model.predict(var_img)
        current_pred = torch.argmax(output, 1)

        iter = 0
        while (iter < max_iters) and (current_pred != target_label) and (search_domain.sum() != 0):
            jacobian = compute_jacobian(var_img)
            p1, p2 = saliency_map(jacobian, var_label, increasing, search_domain, num_features)
            var_sample_flatten = var_img.view(-1, num_features)
            var_sample_flatten[0, p1] += theta
            var_sample_flatten[0, p2] += theta

            new_img = torch.clamp(var_sample_flatten, 0, 1)
            new_img = new_img.view(shape)
            search_domain[p1] = 0
            search_domain[p2] = 0
            var_img = new_img

            output = model.predict(var_img)
            current_pred = torch.argmax(output, 1)
            iter += 1
        return var_img

    imgs = imgs.clone().detach()
    imgs = imgs.unsqueeze(1)

    target_labels = (labels + 1) % num_class

    adv_imgs = None
    for im, tl in zip(imgs, target_labels):
        pert_img = perturbation_single(torch.unsqueeze(im, 0), torch.unsqueeze(tl, 0))
        try:
            adv_imgs = torch.cat((adv_imgs, pert_img), 0)
        except Exception:
            adv_imgs = pert_img
        adv_imgs = torch.clamp(adv_imgs, 0, 1)
    return adv_imgs
