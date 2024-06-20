import os

import torch
from config import *
from tools.data_loader import load_dataset
from tools.model_loader import load_model
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


def train(model_type=conf.structure, bi=conf.binary_cla, class_idx=conf.class_idx, e_type=conf.encoding):
    print(f'training on {device}')

    model = load_model(v=conf.version, model_type=model_type, class_idx=class_idx, device=device,
                       data_size=28, e_type=e_type)

    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)

    train_data, test_data, _, _, _, _ = load_dataset(name=conf.dataset, dir=conf.data_dir, reduction=conf.reduction,
                                                     resize=conf.resize,
                                                     bi=bi, class_idx=class_idx, scale=conf.data_scale)

    model = model.to(device)
    best_acc = 0
    model_save_path = os.path.join(conf.result_dir, conf.dataset, conf.version, model_type)
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
    for epoch in range(epochs):
        log(f'===== Epoch {epoch + 1} =====')
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
                loss = model(images, labels)
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
