"""
tq_models implementation for models/pure/pure_single
"""

import torchquantum as tq
from torchquantum.measurement import expval

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from sklearn.metrics import accuracy_score
from datasets.data_loader import load_dataset

n_qubits = 4
depth = 2


class QuantumFilter(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.h = tq.Hadamard()
        self.encoder = tq.GeneralEncoder([
            {"input_idx": [0], "func": "ry", "wires": [0]},
            {"input_idx": [1], "func": "ry", "wires": [1]},
            {"input_idx": [2], "func": "ry", "wires": [2]},
            {"input_idx": [3], "func": "ry", "wires": [3]},
        ])

        self.ry = tq.RY(has_params=True, trainable=True)
        self.crz = tq.CRZ(has_params=True, trainable=True)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, f, qdev):
        for q in range(n_qubits):
            self.h(qdev, wires=q)
        self.encoder(qdev, f)

        for layer in range(depth):
            for i in range(n_qubits):
                self.crz(qdev, wires=[i, (i + 1) % n_qubits])
            for j in range(n_qubits, 2 * n_qubits):
                self.ry(qdev, wires=j % n_qubits)

        return self.measure(qdev)


class Quanv(tq.QuantumModule):
    def __init__(self, kernel_size=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.qfilter = QuantumFilter()

    def forward(self, x, qdev):
        assert len(x.shape) == 4  # (bs, c, w, h)
        x_lst = []
        for i in range(0, x.shape[2]-1, 2):
            for j in range(0, x.shape[3]-1, 2):
                x_lst.append(
                    self.qfilter(torch.flatten(x[:, :, i:i + self.kernel_size, j:j + self.kernel_size], start_dim=1), qdev)
                )
        x = torch.cat(x_lst, dim=1)
        return x


class SingleEncoding_(nn.Module):
    def __init__(self, device, num_classes=2, img_size=28):
        super().__init__()
        self.device = device
        self.qc = Quanv(kernel_size=2)
        img_size = img_size / 2
        self.fc1 = nn.Linear(in_features=n_qubits * img_size * img_size, out_features=20)
        self.fc2 = nn.Linear(in_features=20, out_features=num_classes)

    def forward(self, x):
        qdev = tq.QuantumDevice(n_wires=n_qubits, bsz=x.shape[0], device=self.device, record_op=True)
        x = self.qc(x, qdev)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


# def train():
#     epochs = 20
#     batch_size = 64
#     lr = 0.01
#     milestones = [10, 20]
#
#     print('training on single_encoding...')
#     model = SingleEncoding_(num_classes=2)
#     model = model.to(device)
#
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr)
#     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)
#
#     train_data, test_data = load_dataset(name='mnist', dir='data', resize=True, bi=True, class_idx=[0, 1])
#
#     for epoch in range(epochs):
#         print(f'===== Epoch {epoch + 1} =====')
#         s_time = time.perf_counter()
#         model.train()
#
#         y_trues = []
#         y_preds = []
#         for i, (images, labels) in enumerate(train_data):
#             b_s_time = time.perf_counter()
#             images, labels = images.to(device), labels.to(device)
#             optimizer.zero_grad()
#
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#
#             loss.backward()
#             optimizer.step()
#
#             b_e_time = time.perf_counter()
#             print(f'\tBatch {i+1} : {b_e_time - b_s_time}s')
#
#             y_trues += labels.cpu().numpy().tolist()
#             y_preds += outputs.data.cpu().numpy().argmax(axis=1).tolist()
#
#         e_time = time.perf_counter()
#         train_acc = accuracy_score(y_trues, y_preds)
#         print('Train: Loss: {:.6f}, Acc: {:.6f}, lr: {:.6f}, Time: {:.2f}s'.format(loss.item(), train_acc,
#                                                                             optimizer.param_groups[0]['lr'],
#                                                                             e_time - s_time))
#
#         scheduler.step()
#
#         model.eval()
#         y_trues = []
#         y_preds = []
#         for i, (images, labels) in enumerate(test_data):
#             images, labels = images.to(device), labels.to(device)
#             with torch.no_grad():
#                 outputs = model(images)
#                 loss = criterion(outputs, labels)
#             y_trues += labels.cpu().numpy().tolist()
#             y_preds += outputs.data.cpu().numpy().argmax(axis=1).tolist()
#         test_acc = accuracy_score(y_trues, y_preds)
#         print('Test: Loss: {:.6f}, Acc: {:.6f}'.format(loss.item(), test_acc))
#
#
# train()

