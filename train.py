import os
import torch
import torch.nn as nn
from config import *
from datasets.data_loader import load_dataset
from models import *
from tq_models import *
from tools import Log
import time
from sklearn.metrics import accuracy_score
from tqdm import tqdm

conf = get_arguments()
epochs = 20
batch_size = 64
lr = 0.01
milestones = [5, 10, 15, 20]


if conf.version == 'tq':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')


if len(conf.class_idx) > 2:
    conf.binary_cla = False


def load_model(v, model_type, class_idx, data_size=28, e_type='amplitude'):
    print(f'!!loading model {model_type} of {v}')
    # todo: num_classes -> len(class_idx)
    num_classes = max(class_idx) + 1

    if v == 'qml':
        if model_type == 'classical':
            model = Classical(num_classes=num_classes)
        elif model_type == 'pure_single':
            model = SingleEncoding(num_classes=num_classes, img_size=data_size)
        elif model_type == 'pure_multi':
            model = MultiEncoding(num_classes=num_classes, img_size=data_size)
        elif model_type == 'quanv_iswap':
            model = QCNNi()
        elif model_type == 'inception':
            model = InceptionNet(num_classes=num_classes)
        elif model_type == 'hier':
            model = Hierarchical(embedding_type=e_type)
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
        elif model_type == 'quanv_iswap':
            assert num_classes == 2
            model = QCNNi_(device=device)
    return model


def train(model_type=conf.structure, bi=conf.binary_cla, class_idx=conf.class_idx):
    print(f'training on {device}')

    model = load_model(v=conf.version, model_type=model_type, class_idx=class_idx, data_size=28)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)

    train_data, test_data = load_dataset(name=conf.dataset, dir=conf.data_dir, reduction=conf.reduction,
                                         resize=conf.resize,
                                         bi=bi, class_idx=class_idx, scale=conf.data_scale)

    model = model.to(device)
    best_acc = 0
    model_save_path = os.path.join(conf.result_dir, conf.dataset, conf.version, model_type)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    log_path = os.path.join(model_save_path, 'log.txt')
    if os.path.exists(log_path):
        os.remove(log_path)
    log = Log(log_path)
    model_save_path = os.path.join(model_save_path, conf.reduction + '_' + str(class_idx) + '.pth')
    for epoch in range(epochs):
        log(f'===== Epoch {epoch + 1} =====')
        s_time = time.perf_counter()
        model.train()

        y_trues = []
        y_preds = []
        for (images, labels) in tqdm(train_data):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

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
                outputs = model(images)
                loss = criterion(outputs, labels)
            y_trues += labels.cpu().numpy().tolist()
            y_preds += outputs.data.cpu().numpy().argmax(axis=1).tolist()
        test_acc = accuracy_score(y_trues, y_preds)
        log('Test: Loss: {:.6f}, Acc: {:.6f}'.format(loss.item(), test_acc))

        if test_acc > best_acc:
            best_acc = test_acc
            log('save best!!')
            torch.save(model.state_dict(), model_save_path)


if __name__ == '__main__':
    train()
