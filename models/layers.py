"""
block, components of different circuits
"""

import pennylane as qml


def QCL_block(d, n_qubits, weights):
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
    qml.CNOT(wires=[n_qubits - 1, 0])
    for i in range(n_qubits):
        qml.RX(weights[d, i, 0], wires=i)
        qml.RZ(weights[d, i, 1], wires=i)
        qml.RX(weights[d, i, 2], wires=i)


def QCNN_conv1(n_qubits, weights_conv1):
    for i in range(0, n_qubits, 2):
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
    for i in range(1, n_qubits - 1, 2):
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
    qml.U3(weights_conv1[n_qubits - 1, 0], weights_conv1[n_qubits - 1, 1], weights_conv1[n_qubits - 1, 2], wires=0)
    qml.U3(weights_conv1[n_qubits - 1, 3], weights_conv1[n_qubits - 1, 4], weights_conv1[n_qubits - 1, 5], wires=n_qubits - 1)
    qml.CNOT(wires=[0, n_qubits - 1])
    qml.RY(weights_conv1[n_qubits - 1, 6], wires=0)
    qml.RZ(weights_conv1[n_qubits - 1, 7], wires=n_qubits - 1)
    qml.CNOT(wires=[n_qubits - 1, 0])
    qml.RY(weights_conv1[n_qubits - 1, 8], wires=0)
    qml.CNOT(wires=[0, n_qubits - 1])
    qml.U3(weights_conv1[n_qubits - 1, 9], weights_conv1[n_qubits - 1, 10], weights_conv1[n_qubits - 1, 11], wires=0)
    qml.U3(weights_conv1[n_qubits - 1, 12], weights_conv1[n_qubits - 1, 13], weights_conv1[n_qubits - 1, 14], wires=n_qubits - 1)

def QCNN_pool1(n_qubits, weights_pool1):
    for idx, i in enumerate(range(0, n_qubits, 2)):
        qml.CRZ(weights_pool1[idx, 0], wires=[i + 1, i])
        qml.PauliX(wires=i + 1)
        qml.CRX(weights_pool1[idx, 1], wires=[i + 1, i])

def QCNN_conv2(n_qubits, weights_conv2):
    for idx, i in enumerate(range(0, n_qubits - 2, 2)):
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

def QCNN_pool2(n_qubits, weights_pool2):
    for idx, i in enumerate(range(0, n_qubits - 2, 4)):
        qml.CRZ(weights_pool2[idx, 0], wires=[i + 2, i])
        qml.PauliX(wires=i + 2)
        qml.CRX(weights_pool2[idx, 1], wires=[i + 2, i])
    qml.CRZ(weights_pool2[(n_qubits - 2)//4, 0], wires=[n_qubits - 2, 0])
    qml.PauliX(wires=n_qubits - 2)
    qml.CRX(weights_pool2[(n_qubits - 2)//4, 1], wires=[n_qubits - 2, 0])

def QCNN_fc(weights_fc):
    qml.CNOT(wires=[0, 4])
    qml.CNOT(wires=[4, 8])
    qml.CNOT(wires=[8, 0])
    qml.RX(weights_fc[0], wires=0)
    qml.RX(weights_fc[1], wires=4)
    qml.RX(weights_fc[2], wires=8)


def CCQC_block(d, n_qubits, weights, weights_1, weights_2):
    if d % 2:
        for i in range(n_qubits):
            qml.RX(weights[d - 1, i, 0], wires=i)
            qml.RZ(weights[d - 1, i, 1], wires=i)
            qml.RX(weights[d - 1, i, 2], wires=i)
        qml.CPhase(weights_1[d - 1], wires=[0, n_qubits - 1])
        qml.RX(weights_2[d - 1], wires=n_qubits - 1)
        for i in range(1, n_qubits):
            qml.CPhase(weights[d - 1, i, 3], wires=[n_qubits - i, n_qubits - i - 1])
            qml.RX(weights[d - 1, i, 4], wires=n_qubits - i - 1)
    else:
        for i in range(n_qubits):
            qml.RX(weights[d - 1, i, 0], wires=i)
            qml.RZ(weights[d - 1, i, 1], wires=i)
            qml.RX(weights[d - 1, i, 2], wires=i)
        j = 0
        for i in range(n_qubits):
            nj = (j + (n_qubits - 3)) % n_qubits
            qml.CPhase(weights[d - 1, i, 3], wires=[j, nj])
            qml.RX(weights[d - 1, i, 4], wires=nj)
            j = nj
