"""
inception(并行): 2 Conv + 1 Quanv
"""

import torch
import torch.nn as nn
import pennylane as qml
from math import ceil

torch.manual_seed(0)

n_qubits = 4
depth = 1
kernel_size = n_qubits


dev = qml.device("default.qubit", wires=n_qubits)


def circuit(inputs, weights):
    ###### 与multi相同
    var_per_qubit = int(len(inputs) / n_qubits) + 1  # num of param for each qubit
    gates = ['RZ', 'RY'] * ceil(var_per_qubit / 2)
    for q in range(n_qubits):
        qml.Hadamard(wires=q)
        for i in range(var_per_qubit):
            if (q * var_per_qubit + i) < len(inputs):
                exec('qml.{}({}, wires = {})'.format(gates[i], inputs[q * var_per_qubit + i], q))
            else:
                pass
    ######

    for d in range(depth):
        for i in range(n_qubits):
            qml.CRZ(weights[d, i], wires=[i, (i+1)%n_qubits])
        for j in range(n_qubits, n_qubits*2):
            qml.RY(weights[d, j], wires=j%n_qubits)

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


class Quanv2d(nn.Module):
    def __init__(self, kernel_size=2):
        super(Quanv2d, self).__init__()
        weight_shapes = {'weights': (depth, 2*n_qubits)}
        qnode = qml.QNode(circuit, dev, interface='torch', diff_method='best')
        self.ql1 = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.kernel_size = kernel_size

    def forward(self, x):
        # x.shape: (batch_size, c, w, h)
        x_list = []
        for i in range(0, x.shape[2] - 1, 2):
            for j in range(0, x.shape[3] - 1, 2):
                cir_out = self.ql1(torch.flatten(x[:, :, i:i + self.kernel_size, j:j + self.kernel_size], start_dim=1))
                x_list.append(cir_out)
        x = torch.cat(x_list, dim=1)
        return x


class Inception(nn.Module):
    def __init__(self, in_channels):
        super(Inception, self).__init__()

        self.branchClassic_1 = nn.Conv2d(in_channels, 4, kernel_size=1, stride=1)
        self.branchClassic_2 = nn.Conv2d(4, 8, kernel_size=4, stride=2)

        self.branchQuantum = Quanv2d(kernel_size=4)

    def forward(self, x):
        classic = self.branchClassic_1(x)
        classic = self.branchClassic_2(classic)

        quantum = self.branchQuantum(x)

        outputs = [classic, quantum]
        return torch.cat(outputs, dim=1)


class InceptionNet(nn.Module):
    def __init__(self, num_classes, channel=1):
        super(InceptionNet, self).__init__()
        self.incep = Inception(in_channels=channel)
        # classical：8个kernel + quan： 4个qubits
        self.fc1 = nn.Linear(12 * 13 * 13, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.lr = nn.LeakyReLU(0.1)

    def forward(self,x):
        bs = x.shape[0]
        x = x.view(bs,1,28,28)
        x = self.incep(x)
        x = self.lr(x)

        x = x.view(bs,-1)
        x = self.lr(self.fc1(x))
        x = self.fc2(x)
        return x
