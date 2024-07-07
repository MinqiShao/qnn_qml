"""
1 Quanv(kernel) + 2 fc
4 qubits, multi encoding: windows(4*4)->multi [RZ, RY] (more single-qubit gates for each qubit)
"""

import pennylane as qml
import torch.nn as nn
import torch
from models.circuits import pure_multi_circuit, qubit_dict

torch.manual_seed(0)

n_qubits = qubit_dict['pure_multi']
l = []
for q in range(n_qubits):
    l.append(q)
depth = 1
kernel_size = n_qubits

dev = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(dev, interface='torch')
def circuit(inputs, weights):
    pure_multi_circuit(n_qubits, depth, inputs, weights)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


@qml.qnode(dev, interface='torch')
def circuit_state(inputs, weights):
    pure_multi_circuit(n_qubits, depth, inputs, weights)
    return qml.state()


@qml.qnode(dev, interface='torch')
def circuit_prob(inputs, weights, depth_=depth):
    pure_multi_circuit(n_qubits, depth_, inputs, weights)
    return qml.probs(wires=l)


@qml.qnode(dev, interface='torch')
def circuit_dm(inputs, weights, q_idx=0, exec_=True):
    pure_multi_circuit(n_qubits, depth, inputs, weights, exec_)
    return qml.density_matrix(wires=q_idx)


def feat_prob(x, weights):
    if len(x.shape) < 3:
        x = x.unsqueeze(0)
    feat = torch.tensor([])
    for i in range(0, x.shape[1]-1, 2):
        for j in range(0, x.shape[2]-1, 2):
            f = circuit_prob(torch.flatten(x[:, i:i+2, j:j+2]), weights, depth)
            feat = torch.cat((feat, f))
    return feat


class Quan2d(nn.Module):
    def __init__(self, kernel_size):
        super(Quan2d, self).__init__()
        weight_shapes = {'weights': (depth, 2 * n_qubits)}
        self.ql1 = qml.qnn.TorchLayer(circuit, weight_shapes)
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


class MultiEncoding(nn.Module):
    def __init__(self, num_classes, img_size=28):
        super(MultiEncoding, self).__init__()
        self.qc = Quan2d(kernel_size=kernel_size)
        img_size = int(img_size / 2)
        self.fc1 = nn.Linear(in_features=n_qubits * img_size * img_size, out_features=num_classes*2)
        self.lr1 = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(in_features=num_classes*2, out_features=num_classes)

    def forward(self, x, y):
        preds = self.predict(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(preds, y)
        return loss

    def predict(self, x):
        x = self.qc(x)
        x = self.fc1(x)
        x = self.lr1(x)
        x = self.fc2(x)
        return x

    def visualize_circuit(self, x, weights, save_path):
        import matplotlib.pyplot as plt
        x = x.numpy()
        fig, ax = qml.draw_mpl(circuit)(x[0, :16], weights)
        fig.show()
        plt.savefig(save_path)
