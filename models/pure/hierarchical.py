"""
Hierarchical circuit quantum classifier (2分类)
https://arxiv.org/abs/2108.00661
"""

import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt

from tools.embedding import *
from models.circuits import qubit_dict, Hierarchical_circuit

n_qubits = qubit_dict['hier']
l = []
for q in range(n_qubits):
    l.append(q)
dev = qml.device('default.qubit', wires=n_qubits)
U = 'U_SU4'

def param_num(u):
    num = 0
    if u == 'U_TTN':
        num = 2
    elif u == 'U_5':
        num = 10
    elif u == 'U_6':
        num = 10
    elif u == 'U_9':
        num = 2
    elif u == 'U_13':
        num = 6
    elif u == 'U_14':
        num = 6
    elif u == 'U_15':
        num = 4
    elif u == 'U_SO4':
        num = 6
    elif u == 'U_SU4':
        num = 15
    return num



# Pooling Layer

def Pooling_ansatz1(params, wires): #2 params
    qml.CRZ(params[0], wires=[wires[0], wires[1]])
    qml.PauliX(wires=wires[0])
    qml.CRX(params[1], wires=[wires[0], wires[1]])

def Pooling_ansatz2(wires): #0 params
    qml.CRZ(wires=[wires[0], wires[1]])

def Pooling_ansatz3(*params, wires): #3 params
    qml.CRot(*params, wires=[wires[0], wires[1]])


@qml.qnode(dev, interface='torch')
def circuit(inputs, weights):
    AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True, pad_with=0)
    Hierarchical_circuit(U, weights)

    return qml.probs(wires=9)  # (bs, 2)


@qml.qnode(dev, interface='torch')
def circuit_state(inputs, weights, u=U, depth_=3, exec_=True):
    AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True, pad_with=0)
    if exec_:
        Hierarchical_circuit(u, weights, depth_)
    return qml.state()


@qml.qnode(dev, interface='torch')
def circuit_prob(inputs, weights, u=U, depth_=3, qubit_l=l):
    AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True, pad_with=0)
    Hierarchical_circuit(u, weights, depth_)

    # if depth_ == 1:
    #     return qml.probs(wires=l)
    # if depth_ == 2:
    #     return qml.probs(wires=[1, 3, 5, 7])
    # if depth_ == 3:
    #     return qml.probs(wires=[5, 9])

    return qml.probs(wires=qubit_l)


class Hierarchical(nn.Module):
    def __init__(self, u='U_SU4', embedding_type='amplitude'):
        super(Hierarchical, self).__init__()
        global U
        U = u
        self.embedding_type = embedding_type

        U_params = param_num(u)
        total_params = U_params * 10
        weight_shapes = {'weights': (total_params, )}
        self.ql = qml.qnn.TorchLayer(circuit, weight_shapes)

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

    def visualize_circuit(self, x, weights, save_path):
        fig, ax = qml.draw_mpl(Hierarchical_circuit)(x, weights)
        fig.show()
        plt.savefig(save_path)


# quantum circuits for conv layers and pool layers
def conv1(U, params):
    U(params, wires=[0, 9])
    for i in range(0, n_qubits, 2):
        U(params, wires=[i, i+1])
    for i in range(1, n_qubits-1, 2):
        U(params, wires=[i, i+1])

def conv2(U, params):
    U(params, wires=[0, 8])
    U(params, wires=[0, 2])
    U(params, wires=[2, 4])
    U(params, wires=[4, 6])
    U(params, wires=[6, 8])

def conv3(U, params):
    U(params, wires=[0, 4])
    U(params, wires=[4, 8])

def pool1(V, params):
    for i in range(0, n_qubits, 2):
        V(params, wires=[i, i+1])

def pool2(V, params):
    V(params, wires=[2, 0])
    V(params, wires=[6, 4])
    V(params, wires=[0, 8])

def pool3(V, params):
    V(params, wires=[0, 4])

# def QCNN_structure(U, params, U_params):
#     param1 = params[0:U_params]
#     param2 = params[U_params: 2 * U_params]
#     param3 = params[2 * U_params: 3 * U_params]
#     param4 = params[3 * U_params: 3 * U_params + 2]
#     param5 = params[3 * U_params + 2: 3 * U_params + 4]
#     param6 = params[3 * U_params + 4: 3 * U_params + 6]
#
#     conv1(U, param1)
#     pool1(Pooling_ansatz1, param4)
#     conv2(U, param2)
#     pool2(Pooling_ansatz1, param5)
#     conv3(U, param3)
#     pool3(Pooling_ansatz1, param6)
#
#
# @qml.qnode(dev, interface='torch')
# def QCNN_circuit(inputs, weights):
#     AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True, pad_with=0)
#     if U == 'U_TTN':
#         QCNN_structure(U_TTN, weights, 2)
#     elif U == 'U_5':
#         QCNN_structure(U_5, weights, 10)
#     elif U == 'U_6':
#         QCNN_structure(U_6, weights, 10)
#     elif U == 'U_9':
#         QCNN_structure(U_9, weights, 2)
#     elif U == 'U_13':
#         QCNN_structure(U_13, weights, 6)
#     elif U == 'U_14':
#         QCNN_structure(U_14, weights, 6)
#     elif U == 'U_15':
#         QCNN_structure(U_15, weights, 4)
#     elif U == 'U_SO4':
#         QCNN_structure(U_SO4, weights, 6)
#     elif U == 'U_SU4':
#         QCNN_structure(U_SU4, weights, 15)
#     else:
#         print("Invalid Unitary Ansatze")
#         return False
#
#     return qml.probs(wires=4)
#
#
# class QCNN_classifier(nn.Module):
#     def __init__(self, u='U_SU4', e='amplitude'):
#         super().__init__()
#         global U
#         U = u
#         self.e_type = e
#         total_params = param_num(u) * 3 + 6
#         weight_shapes = {'weights': (total_params,)}
#         self.ql = qml.qnn.TorchLayer(QCNN_circuit, weight_shapes)
#
#     def forward(self, x, y):
#         preds = self.predict(x)
#         loss = torch.FloatTensor([0])
#         for l, p in zip(y, preds):
#             c_e = l * (torch.log(p[l])) + (1 - l) * torch.log(1 - p[1 - l])
#             loss = loss + c_e
#         return -1 * loss
#
#     def predict(self, x):
#         x = torch.flatten(x, start_dim=1)
#         x = self.ql(x)
#         x = x.float()
#         return x
