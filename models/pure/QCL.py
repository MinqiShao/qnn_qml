"""
qml implementation for QCL
"""

import pennylane as qml
import torch
import torch.nn as nn
from pennylane import AmplitudeEmbedding
from models.circuits import QCL_circuit, qubit_dict
import matplotlib.pyplot as plt


n_qubits = qubit_dict['qcl']
l = []
for q in range(n_qubits):
    l.append(q)
depth = 5
dev = qml.device('default.qubit', wires=n_qubits)


@qml.qnode(dev, interface='torch')
# @qml.qnode(dev, diff_method="backprop")
def circuit(inputs, weights):
    AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True, pad_with=0)
    QCL_circuit(depth, n_qubits, weights)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


@qml.qnode(dev, interface='torch')
# @qml.qnode(dev, diff_method="backprop")
def circuit_state(inputs, weights, exec_=True):
    AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True, pad_with=0)
    if exec_:
        QCL_circuit(depth, n_qubits, weights)

    return qml.state()


@qml.qnode(dev, interface='torch')
def circuit_prob(inputs, weights, exp=False, depth_=depth):
    AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True, pad_with=0)
    QCL_circuit(depth_, n_qubits, weights)
    if exp:
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    else:
        return qml.probs(wires=l)


@qml.qnode(dev, diff_method="backprop")
def circuit_dm(inputs, weights, q_idx, exec_=True):
    AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True, pad_with=0)
    if exec_:
        QCL_circuit(depth, n_qubits, weights)

    return qml.density_matrix(wires=q_idx)


def get_density_matrix(inputs, weights, exec_=True):
    dm_list = []
    for q in range(n_qubits):
        dm_list.append(circuit_dm(inputs, weights, q, exec_))
    return dm_list


def whole_dm(inputs, weights):
    return circuit_dm(inputs, weights, l, exec_=False), circuit_dm(inputs, weights, l)


class QCL_classifier(nn.Module):
    def __init__(self, num_classes=2):
        super(QCL_classifier, self).__init__()
        self.num_classes = num_classes
        self.depth = depth
        weight_shapes = {'weights': (depth, n_qubits, 3)}
        self.ql = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x, y):
        preds = self.predict(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(preds, y)
        return loss

    def predict(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.ql(x)
        return x[:, :self.num_classes]

    def visualize_circuit(self, x, weights, save_path):
        fig, ax = qml.draw_mpl(circuit)(x, weights)
        fig.show()
        plt.savefig(save_path)
