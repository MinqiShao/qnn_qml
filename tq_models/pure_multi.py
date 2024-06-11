"""
tq_models implementation for models/pure/pure_multi
"""

import torchquantum as tq
from torchquantum.measurement import expval

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil


n_qubits = 4
depth = 1
kernel_size = 4


class QuantumFilter(tq.QuantumModule):
    def __init__(self, fn=kernel_size**2):
        self.h = tq.Hadamard()
        self.encoder = self.init_encoder(fn)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def init_encoder(self, fn):
        var_per_qubit = int(fn / n_qubits) + 1
        gates = ['RZ', 'RY'] * ceil(var_per_qubit / 2)
        func_list = []
        for q in range(n_qubits):
            for i in range(var_per_qubit):
                if (q * var_per_qubit + i) < fn:
                    f = {'input_idx: [{}], \'func\': \'{}\', \'wires\': [{}]'
                         .format(q * var_per_qubit + i, gates[i], q)}
                    func_list.append(f)
                else:
                    pass
        encoder = tq.GeneralEncoder(func_list)
        return encoder

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


class MultiEncoding_(nn.Module):
    def __init__(self, device, num_classes=2, img_size=28):
        super().__init__()
        self.device = device
        self.qc = Quanv(kernel_size=2)
        img_size = (img_size-kernel_size) / 2 + 1
        self.fc1 = nn.Linear(in_features=n_qubits * img_size * img_size, out_features=20)
        self.lr1 = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(in_features=20, out_features=num_classes)

    def forward(self, x):
        qdev = tq.QuantumDevice(n_wires=n_qubits, bsz=x.shape[0], device=self.device, record_op=True)
        x = self.qc(x, qdev)
        x = self.fc1(x)
        x = self.lr1(x)
        x = self.fc2(x)
        return x
