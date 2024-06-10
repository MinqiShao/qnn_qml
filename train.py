import os
import torch
import torch.nn as nn
from config import *
from datasets.data_loader import load_dataset
from models import *
import time
from sklearn.metrics import accuracy_score

conf = get_arguments()
epochs = 20
batch_size = 64
lr = 0.01
milestones = [10, 20]

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


if len(conf.class_idx) > 2:
    conf.binary_cla = False


def train(model_type=conf.structure, bi=conf.binary_cla, class_idx=conf.class_idx):
    print(f'training on {device}')

    # num_classes = len(class_idx)
    # todo: num_classes -> len(class_idx)
    num_classes = max(class_idx) + 1
    if model_type == 'classical':
        model = Classical(num_classes=num_classes)
    elif model_type == 'pure_single':
        model = SingleEncoding(num_classes=num_classes)
    elif model_type == 'pure_multi':
        model = MultiEncoding(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)

    train_data, test_data = load_dataset(name=conf.dataset, dir=conf.data_dir, bi=bi, class_idx=class_idx)

    model = model.to(device)
    best_acc = 0
    model_save_path = os.path.join(conf.result_dir, conf.dataset, model_type)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    model_save_path = os.path.join(model_save_path, str(class_idx) + '.pth')
    for epoch in range(epochs):
        print(f'===== Epoch {epoch + 1} =====')
        s_time = time.perf_counter()
        model.train()

        y_trues = []
        y_preds = []
        for i, (images, labels) in enumerate(train_data):
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
        print('Train: Loss: {:.6f}, Acc: {:.6f}, lr: {:.6f}, Time: {:.2f}s'.format(loss.item(), train_acc,
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
        print('Test: Loss: {:.6f}, Acc: {:.6f}'.format(loss.item(), test_acc))

        if test_acc > best_acc:
            best_acc = test_acc
            print('save best!!')
            torch.save(model.state_dict(), model_save_path)


if __name__ == '__main__':
    train()
