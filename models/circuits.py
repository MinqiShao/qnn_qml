"""
circuit structures
dict key names for saving weights
"""
import pennylane as qml
from math import ceil


##### circuit structures
def QCL_circuit(depth, n_qubits, weights):
    for d in range(depth):
        for i in range(n_qubits-1):
            qml.CNOT(wires=[i, i+1])
        qml.CNOT(wires=[n_qubits-1, 0])
        for i in range(n_qubits):
            qml.RX(weights[d, i, 0], wires=i)
            qml.RZ(weights[d, i, 1], wires=i)
            qml.RX(weights[d, i, 2], wires=i)


def pure_qcnn_circuit(n_qubits, weights_conv1, weights_conv2, weights_pool1, weights_pool2, weights_fc):
    # conv1
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
    qml.U3(weights_conv1[7, 0], weights_conv1[7, 1], weights_conv1[7, 2], wires=0)
    qml.U3(weights_conv1[7, 3], weights_conv1[7, 4], weights_conv1[7, 5], wires=n_qubits - 1)
    qml.CNOT(wires=[0, n_qubits - 1])
    qml.RY(weights_conv1[7, 6], wires=0)
    qml.RZ(weights_conv1[7, 7], wires=n_qubits - 1)
    qml.CNOT(wires=[n_qubits - 1, 0])
    qml.RY(weights_conv1[7, 8], wires=0)
    qml.CNOT(wires=[0, n_qubits - 1])
    qml.U3(weights_conv1[7, 9], weights_conv1[7, 10], weights_conv1[7, 11], wires=0)
    qml.U3(weights_conv1[7, 12], weights_conv1[7, 13], weights_conv1[7, 14], wires=n_qubits - 1)

    # pool1
    for idx, i in enumerate(range(0, n_qubits, 2)):
        qml.CRZ(weights_pool1[idx, 0], wires=[i + 1, i])
        qml.PauliX(wires=i + 1)
        qml.CRX(weights_pool1[idx, 1], wires=[i + 1, i])

    # conv2
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

    # pool2
    for idx, i in enumerate(range(0, n_qubits - 2, 4)):
        qml.CRZ(weights_pool2[idx, 0], wires=[i + 2, i])
        qml.PauliX(wires=i + 2)
        qml.CRX(weights_pool2[idx, 1], wires=[i + 2, i])

    # fc
    qml.CNOT(wires=[0, 4])
    qml.CNOT(wires=[2, 4])
    qml.CNOT(wires=[4, 0])
    qml.RX(weights_fc[0], wires=0)
    qml.RX(weights_fc[1], wires=2)
    qml.RX(weights_fc[2], wires=4)


def ccqc_circuit(n_qubits, depth, weights, weights_1, weights_2):
    for d in range(1, depth+1):
        if d % 2:
            for i in range(n_qubits):
                qml.RX(weights[d-1, i, 0], wires=i)
                qml.RZ(weights[d-1, i, 1], wires=i)
                qml.RX(weights[d-1, i, 2], wires=i)
            qml.CPhase(weights_1[d-1], wires=[0, n_qubits-1])
            qml.RX(weights_2[d-1], wires=n_qubits-1)
            for i in range(1, n_qubits):
                qml.CPhase(weights[d-1, i, 3], wires=[n_qubits-i, n_qubits-i-1])
                qml.RX(weights[d-1, i, 4], wires=n_qubits-i-1)
        else:
            for i in range(n_qubits):
                qml.RX(weights[d-1, i, 0], wires=i)
                qml.RZ(weights[d-1, i, 1], wires=i)
                qml.RX(weights[d-1, i, 2], wires=i)
            j = 0
            for i in range(n_qubits):
                nj = (j+(n_qubits-3)) % n_qubits
                qml.CPhase(weights[d-1, i, 3], wires=[j, nj])
                qml.RX(weights[d-1, i, 4], wires=nj)
                j = nj


def pure_single_circuit(n_qubits, depth, weights):
    for layer in range(depth):
        for i in range(n_qubits):
            qml.CRZ(weights[layer, i], wires=[i, (i + 1) % n_qubits])
        for j in range(n_qubits, 2 * n_qubits):
            qml.RY(weights[layer, j], wires=j % n_qubits)


def pure_multi_circuit(n_qubits, depth, inputs, weights):
    ###### 与single不同
    var_per_qubit = int(len(inputs) / n_qubits) + 1  # num of param for each qubit
    gates = ['RZ', 'RY'] * ceil(var_per_qubit / 2)
    for q in range(n_qubits):
        qml.Hadamard(wires=q)
        for i in range(var_per_qubit):
            if (q * var_per_qubit + i) < len(inputs):
                exec('qml.{}({}, wires = {})'.format(gates[i], inputs[q * var_per_qubit + i], q))
            else:
                pass
    ######

    for d in range(depth):
        for i in range(n_qubits):
            qml.CRZ(weights[d, i], wires=[i, (i + 1) % n_qubits])
        for j in range(n_qubits, n_qubits * 2):
            qml.RY(weights[d, j], wires=j % n_qubits)


##### weight dict
weight_dict = {'qcl': 'ql.weights',
               'pure_qcnn': ['cir.weights_conv1', 'cir.weights_conv2', 'cir.weights_pool1', 'cir.weights_pool2', 'cir.weights_fc'],
               'ccqc': ['ql.weights', 'ql.weights_1', 'ql.weights_2'],
               'pure_single': 'qc.ql1.weights',
               'pure_multi': 'qc.ql1.weights'}


