"""
circuit structures, sub-structure
dict key names for saving weights
"""
import pennylane as qml
from math import ceil
from models.layers import *


########## QCL ##########
def QCL_circuit(depth, n_qubits, weights):
    for d in range(depth):
        QCL_block(d, n_qubits, weights)


########## QCNN ##########
def pure_qcnn_circuit(n_qubits, weights_conv1, weights_conv2, weights_pool1, weights_pool2, weights_fc):
    # conv1
    QCNN_conv1(n_qubits, weights_conv1)

    # pool1
    QCNN_pool1(n_qubits, weights_pool1)

    # conv2
    QCNN_conv2(n_qubits, weights_conv2)

    # pool2
    QCNN_pool2(n_qubits, weights_pool2)

    # fc
    QCNN_fc(weights_fc)

def pure_qcnn_block1(n_qubits, weights_conv1, weights_pool1):
    QCNN_conv1(n_qubits, weights_conv1)
    QCNN_pool1(n_qubits, weights_pool1)

def pure_qcnn_block2(n_qubits, weights_conv1, weights_conv2, weights_pool1, weights_pool2):
    QCNN_conv1(n_qubits, weights_conv1)
    QCNN_pool1(n_qubits, weights_pool1)
    QCNN_conv2(n_qubits, weights_conv2)
    QCNN_pool2(n_qubits, weights_pool2)


########## CCQC ##########
def ccqc_circuit(n_qubits, depth, weights, weights_1, weights_2):
    for d in range(1, depth+1):
        CCQC_block(d, n_qubits, weights, weights_1, weights_2)


########## Single/Multi Encoding ##########
def pure_single_circuit(n_qubits, depth, weights):
    for layer in range(depth):
        for i in range(n_qubits):
            qml.CRZ(weights[layer, i], wires=[i, (i + 1) % n_qubits])
        for j in range(n_qubits, 2 * n_qubits):
            qml.RY(weights[layer, j], wires=j % n_qubits)


def pure_multi_circuit(n_qubits, depth, inputs, weights, exec_=True):
    ###### 与single不同
    var_per_qubit = int(len(inputs) / n_qubits) + 1  # num of param for each qubit
    gates = ['RZ', 'RY'] * ceil(var_per_qubit / 2)
    for q in range(n_qubits):
        qml.Hadamard(wires=q)
        for i in range(var_per_qubit):
            if (q * var_per_qubit + i) < len(inputs):
                if gates[i] == 'RZ':
                    qml.RZ(inputs[q * var_per_qubit + i], wires=q)
                elif gates[i] == 'RY':
                    qml.RY(inputs[q * var_per_qubit + i], wires=q)
                # exec('qml.{}({}, wires = {})'.format(gates[i], inputs[q * var_per_qubit + i], q))
            else:
                pass
    ######
    if exec_:
        for d in range(depth):
            for i in range(n_qubits):
                qml.CRZ(weights[d, i], wires=[i, (i + 1) % n_qubits])
            for j in range(n_qubits, n_qubits * 2):
                qml.RY(weights[d, j], wires=j % n_qubits)


########## dict ##########
weight_dict = {'classical': ['conv.weight', 'conv.bias', 'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'],
               'qcl': 'ql.weights',
               'pure_qcnn': ['cir.weights_conv1', 'cir.weights_conv2', 'cir.weights_pool1', 'cir.weights_pool2', 'cir.weights_fc'],
               'ccqc': ['ql.weights', 'ql.weights_1', 'ql.weights_2'],
               'pure_single': 'qc.ql1.weights',
               'pure_multi': 'qc.ql1.weights'}
block_dict = {'qcl': [1, 2, 3, 4, 5],  # 'block1', 'block2', 'block3', 'block4', 'block5'
              'ccqc': [1, 2, 3, 4, 5],
              'pure_qcnn': [0, 1, 2],  # 0: whole, 1: conv1+pool1, 2: conv1+pool1+conv2+pool2
              'pure_single': [0, 1],  # depth for each kernel
              'pure_multi': [0]}
depth_dict = {'qcl': 5,  # 'block1', 'block2', 'block3', 'block4', 'block5'
              'ccqc': 5,
              'pure_qcnn': 2}
qubit_dict = {'qcl': 10, 'ccqc': 10, 'pure_qcnn': 10, 'pure_single': 4, 'pure_multi': 4}


