"""
qml implementation for CCQC
"""

import pennylane as qml
import torch
from pennylane.templates.embeddings import AmplitudeEmbedding
import torch.nn as nn
from models.circuits import ccqc_circuit, qubit_dict


n_qubits = qubit_dict['ccqc']
l = []
for q in range(n_qubits):
    l.append(q)
depth = 5
dev = qml.device('default.qubit', wires=n_qubits)


@qml.qnode(dev, interface='torch')
def circuit(inputs, weights, weights_1, weights_2):
    AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True, pad_with=0)
    ccqc_circuit(n_qubits, depth, weights, weights_1, weights_2)

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


@qml.qnode(dev, interface='torch')
def circuit_state(inputs, weights, weights_1, weights_2, exec_=True, depth_=depth):
    AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True, pad_with=0)
    if exec_:
        ccqc_circuit(n_qubits, depth_, weights, weights_1, weights_2)
    return qml.state()


@qml.qnode(dev, interface='torch')
def circuit_prob(inputs, weights, weights_1, weights_2, depth_=depth):
    AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True, pad_with=0)
    ccqc_circuit(n_qubits, depth_, weights, weights_1, weights_2)
    return qml.probs(wires=l)


@qml.qnode(dev, interface='torch')
def circuit_dm(inputs, weights, weights_1, weights_2, q_idx, exec_=True):
    AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True, pad_with=0)
    if exec_:
        ccqc_circuit(n_qubits, depth, weights, weights_1, weights_2)
    return qml.density_matrix(wires=q_idx)


def get_density_matrix(inputs, weights, weights_1, weights_2, exec_=True):
    dm_list = []
    for q in range(n_qubits):
        dm_list.append(circuit_dm(inputs, weights, weights_1, weights_2, q, exec_))
    return dm_list


def whole_dm(inputs, weights, weights_1, weights_2):
    l = []
    for q in range(n_qubits):
        l.append(q)
    return circuit_dm(inputs, weights, weights_1, weights_2, l, exec_=False), circuit_dm(inputs, weights, weights_1, weights_2, l)


class CCQC_classifier(nn.Module):
    def __init__(self, e='amplitude', num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.depth = depth
        weight_shapes = {'weights': (depth, n_qubits, 5), 'weights_1': (depth,), 'weights_2': (depth,)}
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
        import matplotlib.pyplot as plt
        fig, ax = qml.draw_mpl(circuit)(x, weights[0], weights[1], weights[2])
        fig.show()
        plt.savefig(save_path)
