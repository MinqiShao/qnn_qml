"""
qml implementation for QCL
"""

import pennylane as qml
import torch
import torch.nn as nn
from pennylane.templates.embeddings import AmplitudeEmbedding

n_qubits = 10
depth = 5
dev = qml.device('default.qubit', wires=n_qubits)


@qml.qnode(dev, interface='torch')
def circuit(inputs, weights):
    AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True, pad_with=0)
    for d in range(depth):
        for i in range(n_qubits-1):
            qml.CNOT(wires=[i, i+1])
        qml.CNOT(wires=[n_qubits-1, 0])
        for i in range(n_qubits):
            qml.RX(weights[d, i, 0], wires=i)
            qml.RZ(weights[d, i, 1], wires=i)
            qml.RX(weights[d, i, 2], wires=i)

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


class QCL_classifier(nn.Module):
    def __init__(self):
        super(QCL_classifier, self).__init__()
        weight_shapes = {'weights': (depth, n_qubits, 3)}
        self.ql = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.ql(x)
        return x
