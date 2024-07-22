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


########## Hierarchical ##########
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

def Hierarchical_circuit(U, weights, depth_=3):
    if U == 'U_TTN':
        Hierarchical_structure(U_TTN, weights, 2, depth_)
    elif U == 'U_5':
        Hierarchical_structure(U_5, weights, 10, depth_)
    elif U == 'U_6':
        Hierarchical_structure(U_6, weights, 10, depth_)
    elif U == 'U_9':
        Hierarchical_structure(U_9, weights, 2, depth_)
    elif U == 'U_13':
        Hierarchical_structure(U_13, weights, 6, depth_)
    elif U == 'U_14':
        Hierarchical_structure(U_14, weights, 6, depth_)
    elif U == 'U_15':
        Hierarchical_structure(U_15, weights, 4, depth_)
    elif U == 'U_SO4':
        Hierarchical_structure(U_SO4, weights, 6, depth_)
    elif U == 'U_SU4':
        Hierarchical_structure(U_SU4, weights, 15, depth_)
    else:
        print("Invalid Unitary Ansatz")
        return False

def Hierarchical_structure(U, params, U_params, depth_=3):
    param1 = params[0 * U_params:1 * U_params]
    param2 = params[1 * U_params:2 * U_params]
    param3 = params[2 * U_params:3 * U_params]
    param4 = params[3 * U_params:4 * U_params]
    param5 = params[4 * U_params:5 * U_params]
    param6 = params[5 * U_params:6 * U_params]
    param7 = params[6 * U_params:7 * U_params]
    param8 = params[7 * U_params:8 * U_params]
    param9 = params[8 * U_params:9 * U_params]
    param10 = params[9 * U_params:10 * U_params]
    if depth_ >= 1:
        # 1st Layer
        U(param1, wires=[0, 1])
        U(param2, wires=[2, 3])
        U(param3, wires=[4, 5])
        U(param4, wires=[6, 7])
        U(param5, wires=[8, 9])
    if depth_ > 1:  # depth 2
        # 2nd Layer
        U(param6, wires=[1, 3])
        U(param7, wires=[3, 5])
        U(param8, wires=[5, 7])
        U(param9, wires=[7, 9])
    if depth_ > 2:  # depth 3
        # 3rd Layer
        U(param10, wires=[3, 7])


########## dict ##########
weight_dict = {'classical': ['conv.weight', 'conv.bias', 'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'],
               'qcl': 'ql.weights',
               'pure_qcnn': ['cir.weights_conv1', 'cir.weights_conv2', 'cir.weights_pool1', 'cir.weights_pool2', 'cir.weights_fc'],
               'ccqc': ['ql.weights', 'ql.weights_1', 'ql.weights_2'],
               'pure_single': 'qc.ql1.weights',
               'pure_multi': 'qc.ql1.weights',
               'hier': 'ql.weights'}
block_dict = {'qcl': [1, 2, 3, 4, 5],  # 'block1', 'block2', 'block3', 'block4', 'block5'
              'ccqc': [1, 2, 3, 4, 5],
              'pure_qcnn': [0, 1, 2],  # 0: whole, 1: conv1+pool1, 2: conv1+pool1+conv2+pool2
              'pure_single': [0, 1],  # depth for each kernel
              'pure_multi': [0],
              'hier': [1, 2, 3]}
depth_dict = {'qcl': 5,  # 'block1', 'block2', 'block3', 'block4', 'block5'
              'ccqc': 5,
              'pure_qcnn': 2,  # max value in block_dict
              'hier': 3}
qubit_dict = {'qcl': 10,
              'ccqc': 10,
              'pure_qcnn': 10,
              'pure_single': 4,
              'pure_multi': 4,
              'hier': 10}
qubit_block_dict = {'qcl': [10]*5,
              'ccqc': [10]*5,
              'pure_qcnn': [10]*3,
              'pure_single': [4]*2,
              'pure_multi': [4],
              'hier': [10, 4, 2]}


