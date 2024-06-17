"""
qml implementation for CCQC
"""

import pennylane as qml
import torch
from pennylane.templates.embeddings import AmplitudeEmbedding
import torch.nn as nn


n_qubits = 10
depth = 5
dev = qml.device('default.qubit', wires=n_qubits)


@qml.qnode(dev, interface='torch')
def circuit(inputs, weights, weights_1, weights_2):
    AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True, pad_with=0)
    for d in range(0, depth):
        if d % 2:
            for i in range(n_qubits):
                qml.RX(weights[d, i, 0], wires=i)
                qml.RZ(weights[d, i, 1], wires=i)
                qml.RX(weights[d, i, 2], wires=i)
            qml.CPhase(weights_1[d], wires=[0, n_qubits-1])
            qml.RX(weights_2[d], wires=n_qubits-1)
            for i in range(1, n_qubits):
                qml.CPhase(weights[d, i, 3], wires=[n_qubits-i, n_qubits-i-1])
                qml.RX(weights[d, i, 4], wires=n_qubits-i-1)
        else:
            for i in range(n_qubits):
                qml.RX(weights[d, i, 0], wires=i)
                qml.RZ(weights[d, i, 1], wires=i)
                qml.RX(weights[d, i, 2], wires=i)
            j = 0
            for i in range(n_qubits):
                nj = (j+(n_qubits-3)) % n_qubits
                qml.CPhase(weights[d, i, 3], wires=[j, nj])
                qml.RX(weights[d, i, 4], wires=nj)
                j = nj

        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


class CCQC_classifier(nn.Module):
    def __init__(self):
        super().__init__()
        weight_shapes = {'weights': (depth, n_qubits, 5), 'weights_1': (depth,), 'weights_2': (depth,)}
        self.ql = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.ql(x)
        return x
