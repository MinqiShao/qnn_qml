"""
!!只用于2分类!! 需要resize到14
QHCNN: Conv + Quanv
       Quanv: (Rotation+NoiseLayer+SwapLayer+NoiseLayer)*2 + RX RY RZ on 1st qubit
                                                         |-->data reuploading
       量子电路中只读取第一个qubit，可训练参数3个
"""

import pennylane as qml
from torch import nn
import numpy as np
import torch
import math


n_qubits = 12
add_noise = False
# probabilities of applying depolarising channel
# after data uploading (0 0.001 0.01 0.05
P1: float = 0.01
# after ISWAP gate
P2: float = 0.01
# for observable
zero_label, one_label = torch.zeros(size=(2, 2)), torch.zeros(size=(2, 2))
zero_label[1, 1], one_label[0, 0] = 1., 1.

dev = qml.device("default.qubit", wires=n_qubits)


def noise_layer(prob):
    if add_noise:
        for j in range(n_qubits):
            # Depolarising channel
            if np.random.choice([1, 0], p=[prob / 3, 1 - prob / 3]):
                qml.PauliX(wires=j)
            if np.random.choice([1, 0], p=[prob / 3, 1 - prob / 3]):
                qml.PauliY(wires=j)
            if np.random.choice([1, 0], p=[prob / 3, 1 - prob / 3]):
                qml.PauliZ(wires=j)


def iswap_layer(ascending):
    for i in range(n_qubits-1):
        if not ascending:
            qml.ISWAP(wires=[i+1, i])
        else:
            qml.ISWAP(wires=[n_qubits-i-1, n_qubits-i-2])


def circuit(inputs, weights):
    """
    :param inputs: (6*6+1,) conv的输出+标签0/1
    :param weights: (3,) params for the last three rotation gates on 1st qubit
    :return:
    """
    inputs, y = inputs[:-1].reshape(6, 6), inputs[-1]
    y = one_label if y else zero_label
    for i, row in enumerate(inputs):
        qml.RX(row[:3][0], wires=2 * i)
        qml.RY(row[:3][1], wires=2 * i)
        qml.RX(row[:3][2], wires=2 * i)

        qml.RX(row[3:][0], wires=2 * i + 1)
        qml.RY(row[3:][1], wires=2 * i + 1)
        qml.RX(row[3:][2], wires=2 * i + 1)

    noise_layer(P1)
    iswap_layer(True)
    noise_layer(P2)

    qml.Barrier(wires=[0, 11])

    for i, row in enumerate(inputs):
        qml.RY(row[:3][0], wires=2 * i)
        qml.RX(row[:3][1], wires=2 * i)
        qml.RY(row[:3][2], wires=2 * i)

        qml.RY(row[3:][0], wires=2 * i + 1)
        qml.RX(row[3:][1], wires=2 * i + 1)
        qml.RY(row[3:][2], wires=2 * i + 1)

    noise_layer(P1)
    iswap_layer(False)
    noise_layer(P2)

    qml.RX(weights[0], wires=0)
    qml.RY(weights[1], wires=0)
    qml.RX(weights[2], wires=0)

    return qml.expval(qml.Hermitian(y, wires=0))  # (bs, 1)


class QCNNi(nn.Module):
    def __init__(self):
        super(QCNNi, self).__init__()
        weight_shapes = {'weights': (3,)}
        qnode = qml.QNode(circuit, dev, interface='torch', diff_method='best')

        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=4, stride=2)
        self.ql1 = qml.qnn.TorchLayer(qnode, weight_shapes)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.uniform_(m.weight, 0, 0.1)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, 0.01)
            if isinstance(m, qml.qnn.TorchLayer):
                nn.init.uniform_(m.weights, 0, math.pi/2)

    def forward(self, x):
        bs = x.shape[0]
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1) # (bs, 6*6)
        y_one_label = torch.ones((bs, 1), dtype=torch.int)
        y_zero_label = torch.zeros((bs, 1), dtype=torch.int)
        inputs_1 = torch.cat((x, y_one_label), dim=1)
        inputs_0 = torch.cat((x, y_zero_label), dim=1)
        f1, f2 = self.ql1(inputs_0), self.ql1(inputs_1)
        f1, f2 = f1.reshape(bs, 1), f2.reshape(bs, 1)
        return torch.cat((f1, f2), dim=1)
