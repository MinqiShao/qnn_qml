import os

import torch
from config import *
from tools.data_loader import load_dataset
from tools.model_loader import load_model, load_train_params
from tools.entanglement import *
from tools import Log
from tools.gragh import line_graph, box_graph
import time
from sklearn.metrics import accuracy_score
from tqdm import tqdm

conf = get_arguments()
epochs = 5
batch_size = 64
lr = 0.01
milestones = [5, 10, 15, 20]

if conf.version == 'tq':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')


def train(model_type=conf.structure, class_idx=conf.class_idx, e_type=conf.encoding):
    print(f'training on {device}')

    model = load_model(v=conf.version, model_type=model_type, class_idx=class_idx, device=device,
                       data_size=28, e_type=e_type)

    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)

    train_data, test_data = load_dataset(name=conf.dataset, dir=conf.data_dir, reduction=conf.reduction,
                                                     resize=conf.resize,
                                                     class_idx=class_idx, scale=conf.data_scale)

    model = model.to(device)
    best_acc = 0
    model_save_path = os.path.join(conf.model_dir, conf.dataset, conf.version, model_type)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    if conf.resize:
        log_path = os.path.join(model_save_path, 'log_' + conf.reduction + str(class_idx) + '.txt')
        model_save_path = os.path.join(model_save_path, conf.reduction + '_' + str(class_idx) + '.pth')
    else:
        log_path = os.path.join(model_save_path, 'log_' + str(class_idx) + '.txt')
        model_save_path = os.path.join(model_save_path, str(class_idx) + '.pth')
    if os.path.exists(log_path):
        os.remove(log_path)
    log = Log(log_path)

    epoch_l = []
    ent_c = torch.zeros((len(conf.class_idx), epochs, 100))  # 100: 每个类别取100个测试样本可视化纠缠度
    ent_in = torch.zeros_like(ent_c)
    ent_out = torch.zeros_like(ent_c)
    entro_c = torch.zeros_like(ent_c)
    entro_in = torch.zeros_like(ent_c)
    entro_out = torch.zeros_like(ent_c)
    for epoch in range(epochs):
        log(f'===== Epoch {epoch + 1} =====')
        epoch_l.append(epoch)
        s_time = time.perf_counter()
        model.train()

        y_trues = []
        y_preds = []
        for (images, labels) in tqdm(train_data):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            loss = model(images, labels)
            # loss = criterion(outputs, labels)
            outputs = model.predict(images)

            loss.backward()
            optimizer.step()

            y_trues += labels.cpu().numpy().tolist()
            y_preds += outputs.data.cpu().numpy().argmax(axis=1).tolist()

        e_time = time.perf_counter()
        train_acc = accuracy_score(y_trues, y_preds)
        log('Train: Loss: {:.6f}, Acc: {:.6f}, lr: {:.6f}, Time: {:.2f}s'.format(loss.item(), train_acc,
                                                                                 optimizer.param_groups[0]['lr'],
                                                                                 e_time - s_time))

        scheduler.step()

        model.eval()
        y_trues = []
        y_preds = []
        for i, (images, labels) in enumerate(test_data):
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model.predict(images)
                # loss = criterion(outputs, labels)
            y_trues += labels.cpu().numpy().tolist()
            y_preds += outputs.data.cpu().numpy().argmax(axis=1).tolist()

        test_acc = accuracy_score(y_trues, y_preds)
        log('Test: Loss: {:.6f}, Acc: {:.6f}'.format(loss.item(), test_acc))

        if test_acc > best_acc:
            best_acc = test_acc
            log('save best!!')
            torch.save(model.state_dict(), model_save_path)

        # 按照类别计算纠缠/entropy
        # params = load_train_params(conf.structure, model.state_dict())
        # for c in conf.class_idx:
        #     img_c = test_x[torch.where(test_y == c)[0]]
        #     ent_in_list, ent_out_list, ent_c_list = MW(img_c, params, conf)
        #     ent_c[c, epoch] = ent_c_list[:100]
        #     ent_in[c, epoch] = ent_in_list[:100]
        #     ent_out[c, epoch] = ent_out_list[:100]
        #     entro_in_l, entro_out_l, entro_c_l = entropy(img_c, params, conf)
        #     entro_c[c, epoch] = entro_c_l[:100]
        #     entro_in[c, epoch] = entro_in_l[:100]
        #     entro_out[c, epoch] = entro_out_l[:100]

        # log('Ent: in: {}, out: {}, c: {}'.format(ent_in.tolist(), ent_out.tolist(), ent_c.tolist()))
    # ent变化
    # s_p1 = os.path.join(conf.visual_dir, 'train_ent', conf.dataset, conf.structure, str(conf.class_idx))
    # s_p2 = os.path.join(conf.visual_dir, 'train_entro', conf.dataset, conf.structure, str(conf.class_idx))
    # if not os.path.exists(s_p1):
    #     os.makedirs(s_p1)
    # if not os.path.exists(s_p2):
    #     os.makedirs(s_p2)
    # for c in conf.class_idx:
    #     s_p_ = os.path.join(s_p1, 'class' + str(c) + '_out_plot.png')
    #     line_graph(x=epoch_l, y=torch.mean(ent_out[c], dim=1).tolist(), save_path=s_p_, label_n='ent_out')
    #     s_p_ = os.path.join(s_p1, 'class' + str(c) + '_c_plot.png')
    #     line_graph(x=epoch_l, y=torch.mean(ent_c[c], dim=1).tolist(), save_path=s_p_, label_n='ent_c')
    #     s_p_ = os.path.join(s_p1, 'class' + str(c) + '_box.png')
    #     box_graph(x=epoch_l, y=[ent_in[c].tolist(), ent_out[c].tolist(), ent_c[c].tolist()], data_num=100, save_path=s_p_)
    #
    #     s_p_ = os.path.join(s_p2, 'class' + str(c) + '_out_plot.png')
    #     line_graph(x=epoch_l, y=torch.mean(entro_out[c], dim=1).tolist(), save_path=s_p_, label_n='entro_out')
    #     s_p_ = os.path.join(s_p2, 'class' + str(c) + '_c_plot.png')
    #     line_graph(x=epoch_l, y=torch.mean(entro_c[c], dim=1).tolist(), save_path=s_p_, label_n='entro_c')
    #     s_p_ = os.path.join(s_p2, 'class' + str(c) + '_box.png')
    #     box_graph(x=epoch_l, y=[entro_in[c].tolist(), entro_out[c].tolist(), entro_c[c].tolist()], data_num=100,
    #               save_path=s_p_)


if __name__ == '__main__':
    train()
