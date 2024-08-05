"""
1 Quanv(kernel) + 2 fc
4 qubits, single encoding: windows(2*2)->RY
"""
import pennylane as qml
from pennylane import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from models.circuits import pure_single_circuit, qubit_dict

torch.manual_seed(0)

n_qubits = qubit_dict['pure_single']
l = []
for q in range(n_qubits):
    l.append(q)
depth = 2
dev = qml.device('default.qubit', wires=n_qubits)


@qml.qnode(dev, interface='torch')
def circuit(inputs, weights):
    """
    :param inputs: (4,) single encoding, each param (feature) for each qubit
    :param weights: (i, j): weight for j-th qubit in i-th depth
    :return:
    """
    for qub in range(n_qubits):
        qml.Hadamard(wires=qub)
        qml.RY(inputs[qub], wires=qub)
    pure_single_circuit(n_qubits, depth, weights)

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


@qml.qnode(dev, interface='torch')
def circuit_state(inputs, weights, depth_=depth, exec_=True):
    for qub in range(n_qubits):
        qml.Hadamard(wires=qub)
        qml.RY(inputs[qub], wires=qub)
    if exec_:
        pure_single_circuit(n_qubits, depth_, weights)
    return qml.state()


@qml.qnode(dev, interface='torch')
def circuit_prob(inputs, weights, depth_=depth):
    for qub in range(n_qubits):
        qml.Hadamard(wires=qub)
        qml.RY(inputs[qub], wires=qub)
    pure_single_circuit(n_qubits, depth_, weights)
    # if exp:
    #     return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    return qml.probs(wires=l)


def feat_all(x, weights, ent=False, exec_=True):
    if len(x.shape) == 2:
        x = x.unsqueeze(0)
    elif len(x.shape) == 1:
        x = x.reshape((1, 28, 28))
    feat = []
    for i in range(0, x.shape[1]-1, 2):
        for j in range(0, x.shape[2]-1, 2):
            if ent:
                f = circuit_state(torch.flatten(x[:, i:i + 2, j:j + 2]), weights, depth, exec_=exec_)
            else:
                f = circuit_prob(torch.flatten(x[:, i:i + 2, j:j + 2]), weights, depth)
            feat.append(f)
    return torch.tensor(np.array(feat))


class Quan2d(nn.Module):
    def __init__(self, kernel_size):
        super(Quan2d, self).__init__()
        weight_shapes = {"weights": (depth, 2 * n_qubits)}
        self.ql1 = qml.qnn.TorchLayer(circuit, weight_shapes)
        self.kernel_size = kernel_size

    def forward(self, x):
        assert len(x.shape) == 4  # (bs, c, w, h)
        x_lst = []
        for i in range(0, x.shape[2] - 1, 2):
            for j in range(0, x.shape[3] - 1, 2):
                x_lst.append(
                    self.ql1(torch.flatten(x[:, :, i:i + self.kernel_size, j:j + self.kernel_size], start_dim=1)))
        x = torch.cat(x_lst, dim=1)
        return x


class SingleEncoding(nn.Module):
    def __init__(self, num_classes, img_size=28):
        super(SingleEncoding, self).__init__()
        self.qc = Quan2d(kernel_size=2)
        img_size = int(img_size / 2)
        self.fc1 = nn.Linear(in_features=n_qubits*img_size*img_size, out_features=20)
        self.fc2 = nn.Linear(in_features=20, out_features=num_classes)

    def forward(self, x, y):
        preds = self.predict(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(preds, y)
        return loss

    def predict(self, x):
        if len(x.shape) < 4:
            x = x.reshape(x.shape[0], 1, 28, 28)
        x = self.qc(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def visualize_circuit(self, x, weights, save_path):
        import matplotlib.pyplot as plt
        fig, ax = qml.draw_mpl(circuit)(x[0, :4], weights)
        fig.show()
        plt.savefig(save_path)


