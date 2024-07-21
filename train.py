import os

import torch
from config import *
from tools.data_loader import load_dataset
from tools.model_loader import load_model, load_train_params
from tools.entanglement import *
from tools import Log
from tools.gragh import line_graph, box_graph, train_line
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

    model = load_model(conf=conf, data_size=28, e_type=e_type, device=device)

    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)

    train_data, test_data = load_dataset(name=conf.dataset, dir=conf.data_dir, reduction=conf.reduction,
                                                     resize=conf.resize,
                                                     class_idx=class_idx, scale=conf.data_scale)

    model = model.to(device)
    best_acc = 0
    if conf.structure == 'hier' or conf.structure == 'hier_qcnn':
        model_type += '_' + conf.hier_u
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
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    for epoch in range(epochs):
        log(f'===== Epoch {epoch + 1} =====')
        epoch_l.append(epoch+1)
        s_time = time.perf_counter()
        model.train()

        y_trues = []
        y_preds = []
        total_loss = 0
        for (images, labels) in tqdm(train_data):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            loss = model(images, labels)
            total_loss += loss.item()
            # loss = criterion(outputs, labels)
            outputs = model.predict(images)

            loss.backward()
            optimizer.step()

            y_trues += labels.cpu().numpy().tolist()
            y_preds += outputs.data.cpu().numpy().argmax(axis=1).tolist()

        e_time = time.perf_counter()
        train_acc_ = accuracy_score(y_trues, y_preds)
        train_loss.append(total_loss / len(train_data))
        train_acc.append(train_acc_ * 100)
        log('Train: Loss: {:.6f}, Acc: {:.6f}, lr: {:.6f}, Time: {:.2f}s'.format(loss.item(), train_acc_,
                                                                                 optimizer.param_groups[0]['lr'],
                                                                                 e_time - s_time))

        scheduler.step()

        model.eval()
        y_trues = []
        y_preds = []
        total_loss = 0
        for i, (images, labels) in enumerate(test_data):
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model.predict(images)
                loss = model(images, labels)
                total_loss += loss.item()
            y_trues += labels.cpu().numpy().tolist()
            y_preds += outputs.data.cpu().numpy().argmax(axis=1).tolist()

        test_acc_ = accuracy_score(y_trues, y_preds)
        test_loss.append(total_loss / len(test_data))
        test_acc.append(test_acc_ * 100)
        log('Test: Loss: {:.6f}, Acc: {:.6f}'.format(loss.item(), test_acc_))

        if test_acc_ > best_acc:
            best_acc = test_acc_
            log('save best!!')
            torch.save(model.state_dict(), model_save_path)

    pict_dir = os.path.join(conf.visual_dir, 'train_process', conf.dataset, model_type, str(class_idx))
    if not os.path.exists(pict_dir):
        os.makedirs(pict_dir)
    train_line(e_l=epoch_l, train_l=train_loss, test_l=test_loss, type_='loss',
               save_path=os.path.join(pict_dir, 'train_loss.png'))
    train_line(e_l=epoch_l, train_l=train_acc, test_l=test_acc, type_='accuracy',
               save_path=os.path.join(pict_dir, 'train_acc.png'))
    print(f'training process imgs have saved to {pict_dir}')




if __name__ == '__main__':
    train()
