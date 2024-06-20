"""
qml implementation for QCL
"""

import pennylane as qml
import torch
import torch.nn as nn
from pennylane import AmplitudeEmbedding
from models.circuits import QCL_circuit

from tools.embedding import data_embedding_qml

n_qubits = 10
depth = 5
dev = qml.device('default.qubit', wires=n_qubits)


@qml.qnode(dev, interface='torch')
def circuit(inputs, weights):
    AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True, pad_with=0)
    QCL_circuit(depth, n_qubits, weights)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


# 单个样本
@qml.qnode(dev, interface='torch')
def circuit_state(inputs, weights):
    AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True, pad_with=0)
    QCL_circuit(depth, n_qubits, weights)

    return qml.state()


@qml.qnode(dev, interface='torch')
def circuit_dm_out(inputs, weights, q_idx):
    AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True, pad_with=0)
    QCL_circuit(depth, n_qubits, weights)

    return qml.density_matrix(wires=q_idx)


@qml.qnode(dev, interface='torch')
def circuit_dm_in(inputs, weights, q_idx):
    AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True, pad_with=0)

    return qml.density_matrix(wires=q_idx)


def out_density_matrices(inputs, weights):
    dm_list = []
    for q in range(n_qubits):
        dm_list.append(circuit_dm_out(inputs, weights, q))
    return dm_list


def in_density_matrices(inputs, weights):
    dm_list = []
    for q in range(n_qubits):
        dm_list.append(circuit_dm_in(inputs, weights, q))
    return dm_list


def whole_dm(inputs, weights):
    l = []
    for q in range(n_qubits):
        l.append(q)
    return circuit_dm_in(inputs, weights, l), circuit_dm_out(inputs, weights, l)


class QCL_classifier(nn.Module):
    def __init__(self):
        super(QCL_classifier, self).__init__()
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
        return x
