"""
weighted probability of conv and circuit output
"""


import pennylane as qml
import torch.nn as nn
import torch.nn.functional as F
import torch
from math import ceil, log2

torch.manual_seed(0)

global n_qubits
depth = 2
dev = qml.device('default.qubit', wires=n_qubits)


@qml.qnode(dev, interface='torch')
def circuit(inputs, weights):
    """
    :param inputs:
    :param weights: (2,) theta, phi
    :return: amplitude of quantum state
    """
    feature_dim = inputs.shapes[0]
    n_qubits = int(ceil(log2(feature_dim)))

    for i in range(n_qubits):
        if i % 2 == 0:
            qml.RY(weights[0], wires=i)
        else:
            qml.RY(weights[1], wires=i)

    for i in range(n_qubits):
        qml.CNOT(wires=[i, i+1])

    return qml.state()
