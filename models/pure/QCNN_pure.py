"""
qml implementation for QCNN (conv1, conv2, pool1, pool2, fc)
resize
"""

import pennylane as qml
import torch
import torch.nn as nn
from pennylane.templates.embeddings import AmplitudeEmbedding
import math
import numpy as np


n_qubits = 10
dev = qml.device('default.qubit', wires=n_qubits)


@qml.qnode(dev, interface='torch')
def circuit(inputs, weights_conv1, weights_conv2, weights_pool1, weights_pool2, weights_fc):
    AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True, pad_with=0)
    # conv1
    for i in range(0, n_qubits, 2):
        qml.U3(weights_conv1[i, 0], weights_conv1[i, 1], weights_conv1[i, 2], wires=i)
        qml.U3(weights_conv1[i, 3], weights_conv1[i, 4], weights_conv1[i, 5], wires=i+1)
        qml.CNOT(wires=[i, i+1])
        qml.RY(weights_conv1[i, 6], wires=i)
        qml.RZ(weights_conv1[i, 7], wires=i+1)
        qml.CNOT(wires=[i+1, i])
        qml.RY(weights_conv1[i, 8], wires=i)
        qml.CNOT(wires=[i, i+1])
        qml.U3(weights_conv1[i, 9], weights_conv1[i, 10], weights_conv1[i, 11], wires=i)
        qml.U3(weights_conv1[i, 12], weights_conv1[i, 13], weights_conv1[i, 14], wires=i + 1)
    for i in range(1, n_qubits-1, 2):
        qml.U3(weights_conv1[i, 0], weights_conv1[i, 1], weights_conv1[i, 2], wires=i)
        qml.U3(weights_conv1[i, 3], weights_conv1[i, 4], weights_conv1[i, 5], wires=i + 1)
        qml.CNOT(wires=[i, i + 1])
        qml.RY(weights_conv1[i, 6], wires=i)
        qml.RZ(weights_conv1[i, 7], wires=i + 1)
        qml.CNOT(wires=[i + 1, i])
        qml.RY(weights_conv1[i, 8], wires=i)
        qml.CNOT(wires=[i, i + 1])
        qml.U3(weights_conv1[i, 9], weights_conv1[i, 10], weights_conv1[i, 11], wires=i)
        qml.U3(weights_conv1[i, 12], weights_conv1[i, 13], weights_conv1[i, 14], wires=i + 1)
    qml.U3(weights_conv1[7, 0], weights_conv1[7, 1], weights_conv1[7, 2], wires=0)
    qml.U3(weights_conv1[7, 3], weights_conv1[7, 4], weights_conv1[7, 5], wires=n_qubits-1)
    qml.CNOT(wires=[0, n_qubits-1])
    qml.RY(weights_conv1[7, 6], wires=0)
    qml.RZ(weights_conv1[7, 7], wires=n_qubits-1)
    qml.CNOT(wires=[n_qubits-1, 0])
    qml.RY(weights_conv1[7, 8], wires=0)
    qml.CNOT(wires=[0, n_qubits-1])
    qml.U3(weights_conv1[7, 9], weights_conv1[7, 10], weights_conv1[7, 11], wires=0)
    qml.U3(weights_conv1[7, 12], weights_conv1[7, 13], weights_conv1[7, 14], wires=n_qubits-1)

    # pool1
    for idx, i in enumerate(range(0, n_qubits, 2)):
        qml.CRZ(weights_pool1[idx, 0], wires=[i + 1, i])
        qml.X(wires=i + 1)
        qml.CRX(weights_pool1[idx, 1], wires=[i + 1, i])

    # conv2
    for idx, i in enumerate(range(0, n_qubits-2, 2)):
        qml.U3(weights_conv2[idx, 0], weights_conv2[idx, 1], weights_conv2[idx, 2], wires=i)
        qml.U3(weights_conv2[idx, 3], weights_conv2[idx, 4], weights_conv2[idx, 5], wires=i + 2)
        qml.CNOT(wires=[i, i + 2])
        qml.RY(weights_conv2[idx, 6], wires=i)
        qml.RZ(weights_conv2[idx, 7], wires=i + 2)
        qml.CNOT(wires=[i + 2, i])
        qml.RY(weights_conv2[idx, 8], wires=i)
        qml.CNOT(wires=[i, i + 2])
        qml.U3(weights_conv2[idx, 9], weights_conv2[idx, 10], weights_conv2[idx, 11], wires=i)
        qml.U3(weights_conv2[idx, 12], weights_conv2[idx, 13], weights_conv2[idx, 14], wires=i + 2)

    # pool2
    for idx, i in enumerate(range(0, n_qubits-2, 4)):
        qml.CRZ(weights_pool2[idx, 0], wires=[i+2, i])
        qml.X(wires=i+2)
        qml.CRX(weights_pool2[idx, 1], wires=[i+2, i])

    # fc
    qml.CNOT(wires=[0, 4])
    qml.CNOT(wires=[2, 4])
    qml.CNOT(wires=[4, 0])
    qml.RX(weights_fc[0], wires=0)
    qml.RX(weights_fc[1], wires=2)
    qml.RX(weights_fc[2], wires=4)

    return [qml.expval(qml.PauliZ(i)) for i in range(0, 4, 2)]


class QCNN_classifier(nn.Module):
    def __init__(self, num_classes=2):
        super(QCNN_classifier, self).__init__()
        self.num_classes = num_classes

        self.cir = qml.qnn.TorchLayer(circuit, {'weights_conv1': (n_qubits, 15),
                                                'weights_conv2': (math.ceil((n_qubits-2)/2), 15),
                                                'weights_pool1': (math.ceil(n_qubits/2), 2),
                                                'weights_pool2': (math.ceil((n_qubits-2)/4), 2),
                                                'weights_fc': (3,)})

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.cir(x)
        return x

