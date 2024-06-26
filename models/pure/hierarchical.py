"""
Hierarchical circuit quantum classifier
8 qubits, 2分类
resize
"""

import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
from tools.embedding import *
import autograd.numpy as anp

n_qubits = 8
dev = qml.device('default.qubit', wires=8)
U = 'U_SU4'
U_params=15


# Unitary ansatz for conv layer
def U_TTN(params, wires):  # 2 params 2 qubits
    qml.RY(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])


def U_5(params, wires):  # 10 params 2 qubits
    qml.RX(params[0], wires=wires[0])
    qml.RX(params[1], wires=wires[1])
    qml.RZ(params[2], wires=wires[0])
    qml.RZ(params[3], wires=wires[1])
    qml.CRZ(params[4], wires=[wires[1], wires[0]])
    qml.CRZ(params[5], wires=[wires[0], wires[1]])
    qml.RX(params[6], wires=wires[0])
    qml.RX(params[7], wires=wires[1])
    qml.RZ(params[8], wires=wires[0])
    qml.RZ(params[9], wires=wires[1])


def U_6(params, wires):  # 10 params 2 qubits
    qml.RX(params[0], wires=wires[0])
    qml.RX(params[1], wires=wires[1])
    qml.RZ(params[2], wires=wires[0])
    qml.RZ(params[3], wires=wires[1])
    qml.CRX(params[4], wires=[wires[1], wires[0]])
    qml.CRX(params[5], wires=[wires[0], wires[1]])
    qml.RX(params[6], wires=wires[0])
    qml.RX(params[7], wires=wires[1])
    qml.RZ(params[8], wires=wires[0])
    qml.RZ(params[9], wires=wires[1])


def U_9(params, wires): # 2 params 2 qubits
    qml.Hadamard(wires=wires[0])
    qml.Hadamard(wires=wires[1])
    qml.CZ(wires=[wires[0], wires[1]])
    qml.RX(params[0], wires=wires[0])
    qml.RX(params[1], wires=wires[1])


def U_13(params, wires):  # 6 params 2qubits
    qml.RY(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CRZ(params[2], wires=[wires[0], wires[1]])
    qml.RY(params[3], wires=wires[0])
    qml.RY(params[4], wires=wires[1])
    qml.CRZ(params[5], wires=[wires[0], wires[1]])


def U_14(params, wires):  # 6 params
    qml.RY(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CRX(params[2], wires=[wires[1], wires[0]])
    qml.RY(params[3], wires=wires[0])
    qml.RY(params[4], wires=wires[1])
    qml.CRX(params[5], wires=[wires[0], wires[1]])


def U_15(params, wires):  # 4 params
    qml.RY(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RY(params[2], wires=wires[0])
    qml.RY(params[3], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])


def U_SO4(params, wires):  # 6 params
    qml.RY(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[2], wires=wires[0])
    qml.RY(params[3], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[4], wires=wires[0])
    qml.RY(params[5], wires=wires[1])


def U_SU4(params, wires): # 15 params
    qml.U3(params[0], params[1], params[2], wires=wires[0])
    qml.U3(params[3], params[4], params[5], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[6], wires=wires[0])
    qml.RZ(params[7], wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RY(params[8], wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.U3(params[9], params[10], params[11], wires=wires[0])
    qml.U3(params[12], params[13], params[14], wires=wires[1])

# Pooling Layer

def Pooling_ansatz1(params, wires): #2 params
    qml.CRZ(params[0], wires=[wires[0], wires[1]])
    qml.PauliX(wires=wires[0])
    qml.CRX(params[1], wires=[wires[0], wires[1]])

def Pooling_ansatz2(wires): #0 params
    qml.CRZ(wires=[wires[0], wires[1]])

def Pooling_ansatz3(*params, wires): #3 params
    qml.CRot(*params, wires=[wires[0], wires[1]])



def Hierarchical_structure(U, params, U_params):
    param1 = params[0 * U_params:1 * U_params]
    param2 = params[1 * U_params:2 * U_params]
    param3 = params[2 * U_params:3 * U_params]
    param4 = params[3 * U_params:4 * U_params]
    param5 = params[4 * U_params:5 * U_params]
    param6 = params[5 * U_params:6 * U_params]
    param7 = params[6 * U_params:7 * U_params]

    # 1st Layer
    U(param1, wires=[0, 1])
    U(param2, wires=[2, 3])
    U(param3, wires=[4, 5])
    U(param4, wires=[6, 7])
    # 2nd Layer
    U(param5, wires=[1, 3])
    U(param6, wires=[5, 7])
    # 3rd Layer
    U(param7, wires=[3, 7])


@qml.qnode(dev, interface='torch')
def Hierarchical_circuit(inputs, weights):
    """
    :param weights: (U_params*7, )
    :return:
    """
    AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True, pad_with=0)
    if U == 'U_TTN':
        Hierarchical_structure(U_TTN, weights, 2)
    elif U == 'U_5':
        Hierarchical_structure(U_5, weights, 10)
    elif U == 'U_6':
        Hierarchical_structure(U_6, weights, 10)
    elif U == 'U_9':
        Hierarchical_structure(U_9, weights, 2)
    elif U == 'U_13':
        Hierarchical_structure(U_13, weights, 6)
    elif U == 'U_14':
        Hierarchical_structure(U_14, weights, 6)
    elif U == 'U_15':
        Hierarchical_structure(U_15, weights, 4)
    elif U == 'U_SO4':
        Hierarchical_structure(U_SO4, weights, 6)
    elif U == 'U_SU4':
        Hierarchical_structure(U_SU4, weights, 15)
    else:
        print("Invalid Unitary Ansatz")
        return False

    # return qml.expval(qml.PauliZ(7))
    return qml.probs(wires=7)  # (bs, 2)


class Hierarchical(nn.Module):
    def __init__(self, embedding_type='amplitude'):
        super(Hierarchical, self).__init__()
        self.embedding_type = embedding_type

        total_params = U_params * 7
        weight_shapes = {'weights': (total_params, )}
        self.ql = qml.qnn.TorchLayer(Hierarchical_circuit, weight_shapes)

    def forward(self, x, y):
        preds = self.predict(x)
        loss = torch.FloatTensor([0])
        for l, p in zip(y, preds):
            c_e = l * (torch.log(p[l])) + (1 - l) * torch.log(1 - p[1 - l])
            loss = loss + c_e
        return -1 * loss

    def predict(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.ql(x)
        x = x.float()
        return x


# quantum circuits for conv layers and pool layers
def conv1(U, params):
    U(params, wires=[0, 7])
    for i in range(0, n_qubits, 2):
        U(params, wires=[i, i+1])
    for i in range(1, n_qubits-1, 2):
        U(params, wires=[i, i+1])

def conv2(U, params):
    U(params, wires=[0, 6])
    U(params, wires=[0, 2])
    U(params, wires=[4, 6])
    U(params, wires=[2, 4])

def conv3(U, params):
    U(params, wires=[0, 4])

def pool1(V, params):
    for i in range(0, n_qubits, 2):
        V(params, wires=[i, i+1])

def pool2(V, params):
    V(params, wires=[2, 0])
    V(params, wires=[6, 4])

def pool3(V, params):
    V(params, wires=[0, 4])

def QCNN_structure(U, params, U_params):
    param1 = params[0:U_params]
    param2 = params[U_params: 2 * U_params]
    param3 = params[2 * U_params: 3 * U_params]
    param4 = params[3 * U_params: 3 * U_params + 2]
    param5 = params[3 * U_params + 2: 3 * U_params + 4]
    param6 = params[3 * U_params + 4: 3 * U_params + 6]

    conv1(U, param1)
    pool1(Pooling_ansatz1, param4)
    conv2(U, param2)
    pool2(Pooling_ansatz1, param5)
    conv3(U, param3)
    pool3(Pooling_ansatz1, param6)


@qml.qnode(dev, interface='torch')
def QCNN_circuit(inputs, weights):
    AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True, pad_with=0)
    if U == 'U_TTN':
        QCNN_structure(U_TTN, weights, 2)
    elif U == 'U_5':
        QCNN_structure(U_5, weights, 10)
    elif U == 'U_6':
        QCNN_structure(U_6, weights, 10)
    elif U == 'U_9':
        QCNN_structure(U_9, weights, 2)
    elif U == 'U_13':
        QCNN_structure(U_13, weights, 6)
    elif U == 'U_14':
        QCNN_structure(U_14, weights, 6)
    elif U == 'U_15':
        QCNN_structure(U_15, weights, 4)
    elif U == 'U_SO4':
        QCNN_structure(U_SO4, weights, 6)
    elif U == 'U_SU4':
        QCNN_structure(U_SU4, weights, 15)
    else:
        print("Invalid Unitary Ansatze")
        return False

    return qml.probs(wires=4)


class QCNN_classifier(nn.Module):
    def __init__(self, e='amplitude'):
        super().__init__()
        self.e_type = e
        total_params = U_params * 3 + 6
        weight_shapes = {'weights': (total_params,)}
        self.ql = qml.qnn.TorchLayer(QCNN_circuit, weight_shapes)

    def forward(self, x, y):
        x = torch.flatten(x, start_dim=1)
        x = self.ql(x)
        x = x.float()
        return x

    def predict(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.ql(x)
        x = x.float()
        return x
