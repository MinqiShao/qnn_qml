"""
tq_models implementation for models/hybrid/Quanv_iswap
"""
import math
import torchquantum as tq
import torch
import torch.nn as nn
import numpy as np
from torchquantum.operator import Observable
from abc import ABCMeta


n_qubits = 12
add_noise = False

zero_label, one_label = torch.zeros(size=(2, 2)), torch.zeros(size=(2, 2))
zero_label[1, 1], one_label[0, 0] = 1., 1.

P1 = 0.01
P2 = 0.01


class Y_0_obs(Observable, metaclass=ABCMeta):
    num_params = 0
    num_wires = 1
    eigvals = torch.tensor([1, 0], dtype=torch.complex64)
    matrix = zero_label

    @classmethod
    def _matrix(cls, params):
        return cls.matrix

    @classmethod
    def _eigvals(cls, params):
        return cls.eigvals

    def diagonalizing_gates(self):
        return []


class Y_1_obs(Observable, metaclass=ABCMeta):
    num_params = 0
    num_wires = 1
    eigvals = torch.tensor([1, 0], dtype=torch.complex64)
    matrix = one_label

    @classmethod
    def _matrix(cls, params):
        return cls.matrix

    @classmethod
    def _eigvals(cls, params):
        return cls.eigvals

    def diagonalizing_gates(self):
        return []


class NoiseLayer(tq.QuantumModule):
    def __init__(self, prob):
        super().__init__()
        self.prob = prob
        self.x = tq.PauliX()
        self.y = tq.PauliY()
        self.z = tq.PauliZ()

    def forward(self, qdev):
        if add_noise:
            for i in range(n_qubits):
                if np.random.choice([1, 0], p=[self.prob/3, 1-self.prob/3]):
                    self.x(qdev, wires=i)
                if np.random.choice([1, 0], p=[self.prob / 3, 1 - self.prob / 3]):
                    self.y(qdev, wires=i)
                if np.random.choice([1, 0], p=[self.prob / 3, 1 - self.prob / 3]):
                    self.z(qdev, wires=i)


class IswapLayer(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.iswap = tq.ISWAP()
        
    def forward(self, ascending, qdev):
        for i in range(n_qubits-1):
            if ascending:
                self.iswap(qdev, wires=[n_qubits-i-1, n_qubits-i-2])
            else:
                self.iswap(qdev, wires=[i+1, i])


class Quanv(tq.QuantumModule):
    def __init__(self):
        super().__init__()

        self.encoder1 = self.initial_encoder(fn=6*6, idx=1)
        self.encoder2 = self.initial_encoder(fn=6*6, idx=2)
        self.noise_layer1 = NoiseLayer(P1)
        self.noise_layer2 = NoiseLayer(P2)
        self.iswap_layer = IswapLayer()

        # trainable params
        self.rx = tq.RX(has_params=True, trainable=True, init_params=math.pi/2)
        self.ry = tq.RY(has_params=True, trainable=True, init_params=math.pi/2)
        self.rz = tq.RZ(has_params=True, trainable=True, init_params=math.pi/2)

        self.obs1 = Y_1_obs()
        self.obs0 = Y_0_obs()
        self.measure = tq.MeasureAll(tq.PauliZ)

    def initial_encoder(self, fn, idx=1):
        func_list = []
        var_per_qubit = 3
        if idx == 1:
            gates = ['rx', 'ry', 'rx']
        else:
            gates = ['ry', 'rx', 'ry']
        for q in range(n_qubits):
            for i in range(var_per_qubit):
                if (q*var_per_qubit+i) < fn:
                    f = {'input_idx': [q*var_per_qubit+i], 'func': gates[i], 'wires': q}
                    func_list.append(f)

        encoder = tq.GeneralEncoder(func_list)
        return encoder

    def forward(self, inputs, qdev):
        # inputs: (6*6,)
        self.encoder1(qdev, inputs)

        self.noise_layer1(qdev)
        self.iswap_layer(True, qdev)
        self.noise_layer2(qdev)

        self.encoder2(qdev, inputs)

        self.noise_layer1(qdev)
        self.iswap_layer(False, qdev)
        self.noise_layer2(qdev)

        self.rx(qdev, wires=0)
        self.ry(qdev, wires=0)
        self.rz(qdev, wires=0)
        self.rx(qdev, wires=1)
        self.ry(qdev, wires=1)
        self.rz(qdev, wires=1)

        # f1 = tq.expval(qdev, wires=0, observables=self.obs1)
        # f0 = tq.expval(qdev, wires=0, observables=self.obs0)
        #  return torch.cat((f0, f1), dim=1)
        results = self.measure(qdev)
        return results[:, torch.tensor([0, 1])]


class QCNNi_(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=4, stride=2)
        self.ql = Quanv()

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.uniform_(m.weight, 0, 0.1)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, 0.01)

    def forward(self, x):
        qdev = tq.QuantumDevice(n_wires=n_qubits, bsz=x.shape[0], device=self.device, record_op=True)
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.ql(x, qdev)
        return x
