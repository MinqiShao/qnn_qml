"""
Different encoding/embedding methods
(1) Basic: Amplitude, Angle
(2) Hybrid: hybrid direct embedding(HDE), hybrid angle embedding(HAE)
"""

import torchquantum as tq
from torchquantum.encoding import encoder_op_list_name_dict
import pennylane as qml
from pennylane.templates.embeddings import AmplitudeEmbedding, AngleEmbedding
from pennylane.templates.state_preparations import MottonenStatePreparation
import numpy as np


def data_embedding_tq(n_qubits, e_type='amplitude'):
    if e_type == 'amplitude':
        e = tq.AmplitudeEncoder()
    elif e_type == 'angle_y':
        # 4 16 10 25 8x2
        e_name = f'{n_qubits}_ry' if n_qubits != 8 else f'{n_qubits}x2_ry'
        e = tq.GeneralEncoder(encoder_op_list_name_dict[e_name])
    elif e_type == 'angle_xyz':
        # 10
        e_name = f'6x6_ryzx'
        e = tq.GeneralEncoder(encoder_op_list_name_dict[e_name])

    elif e_type == 'hde':  # 将amplitude分解为两个block分别encoding
        # m = int(n_qubits/2)  # 每个block中的qubit数
        e = [tq.AmplitudeEncoder(), tq.AmplitudeEncoder()]

    elif e_type == 'hae':
        e = Angular_Hybrid_tq()

    return e


def data_embedding_qml(x, n_qubits, e_type='amplitude'):
    if e_type == 'amplitude':
        AmplitudeEmbedding(x, wires=range(n_qubits), normalize=True, pad_with=0)
    elif e_type == 'angle':
        AngleEmbedding(x, wires=range(n_qubits), rotation='Y')
    elif e_type == 'angle_xyz':
        AngleEmbedding(x[:n_qubits], wires=range(n_qubits), rotation='X')
        AngleEmbedding(x[n_qubits:n_qubits*2], wires=range(n_qubits), rotation='Y')

    elif e_type == 'hde':
        # 2 blocks, each of m qubits
        m = int(n_qubits / 2)
        x1 = x[:2**m]
        x2 = x[2**m:2*(2**m)]
        norm_x1, norm_x2 = np.linalg.norm(x1), np.linalg.norm(x2)
        x1, x2 = x1/norm_x1, x2/norm_x2

        wires1, wires2 = [], []
        for i in range(m):
            wires1.append(i)
            wires2.append(i+m)

        MottonenStatePreparation(x1, wires=wires1)
        MottonenStatePreparation(x2, wires=wires2)

    # n_qubits = 8, features = 30
    elif e_type == 'hae':
        assert n_qubits == 8
        x1 = x[:15]
        x2 = x[15:2*15]
        Angular_Hybrid_qml(x1, wires=[0, 1, 2, 3])
        Angular_Hybrid_qml(x2, wires=[4, 5, 6, 7])

"""
alternative mottonen state preparation to avoid normalization problem
"""
def Angular_Hybrid_qml(X, wires):
    # 15 features -> 4 qubits
    qml.RY(X[0], wires=wires[0])

    qml.PauliX(wires=wires[0])
    qml.CRY(X[1], wires=[wires[0], wires[1]])
    qml.PauliX(wires=wires[0])
    qml.CRY(X[2], wires=[wires[0], wires[1]])

    qml.RY(X[3], wires=wires[2])
    qml.CNOT(wires=[wires[1], wires[2]])
    qml.RY(X[4], wires=wires[2])
    qml.CNOT(wires=[wires[0], wires[2]])
    qml.RY(X[5], wires=wires[2])
    qml.CNOT(wires=[wires[1], wires[2]])
    qml.RY(X[6], wires=wires[2])
    qml.CNOT(wires=[wires[0], wires[2]])

    qml.RY(X[7], wires=wires[3])
    qml.CNOT(wires=[wires[2], wires[3]])
    qml.RY(X[8], wires=wires[3])
    qml.CNOT(wires=[wires[1], wires[3]])
    qml.RY(X[9], wires=wires[3])
    qml.CNOT(wires=[wires[2], wires[3]])
    qml.RY(X[10], wires=wires[3])
    qml.CNOT(wires=[wires[0], wires[3]])
    qml.RY(X[11], wires=wires[3])
    qml.CNOT(wires=[wires[2], wires[3]])
    qml.RY(X[12], wires=wires[3])
    qml.CNOT(wires=[wires[1], wires[3]])
    qml.RY(X[13], wires=wires[3])
    qml.CNOT(wires=[wires[2], wires[3]])
    qml.RY(X[14], wires=wires[3])
    qml.CNOT(wires=[wires[0], wires[3]])


def Angular_Hybrid_tq():
    func_list = []
    func_list.append({'input_idx': [0], 'func': 'ry', 'wires': [0]})

    func_list.append({'input_idx': None, 'func': 'sx', 'wires': [0]})
    func_list.append({'input_idx': [1], 'func': 'cry', 'wires': [0, 1]})
    func_list.append({'input_idx': None, 'func': 'sx', 'wires': [0]})
    func_list.append({'input_idx': [2], 'func': 'cry', 'wires': [0, 1]})

    func_list.append({'input_idx': [3], 'func': 'ry', 'wires': [2]})
    func_list.append({'input_idx': None, 'func': 'cnot', 'wires': [1, 2]})
    func_list.append({'input_idx': [4], 'func': 'ry', 'wires': [2]})
    func_list.append({'input_idx': None, 'func': 'cnot', 'wires': [0, 2]})
    func_list.append({'input_idx': [5], 'func': 'ry', 'wires': [2]})
    func_list.append({'input_idx': None, 'func': 'cnot', 'wires': [1, 2]})
    func_list.append({'input_idx': [6], 'func': 'ry', 'wires': [2]})
    func_list.append({'input_idx': None, 'func': 'cnot', 'wires': [0, 2]})

    func_list.append({'input_idx': [7], 'func': 'ry', 'wires': [3]})
    func_list.append({'input_idx': None, 'func': 'cnot', 'wires': [2, 3]})
    func_list.append({'input_idx': [8], 'func': 'ry', 'wires': [3]})
    func_list.append({'input_idx': None, 'func': 'cnot', 'wires': [1, 3]})
    func_list.append({'input_idx': [9], 'func': 'ry', 'wires': [3]})
    func_list.append({'input_idx': None, 'func': 'cnot', 'wires': [2, 3]})
    func_list.append({'input_idx': [10], 'func': 'ry', 'wires': [3]})
    func_list.append({'input_idx': None, 'func': 'cnot', 'wires': [0, 3]})
    func_list.append({'input_idx': [11], 'func': 'ry', 'wires': [3]})
    func_list.append({'input_idx': None, 'func': 'cnot', 'wires': [2, 3]})
    func_list.append({'input_idx': [12], 'func': 'ry', 'wires': [3]})
    func_list.append({'input_idx': None, 'func': 'cnot', 'wires': [1, 3]})
    func_list.append({'input_idx': [13], 'func': 'ry', 'wires': [3]})
    func_list.append({'input_idx': None, 'func': 'cnot', 'wires': [2, 3]})
    func_list.append({'input_idx': [14], 'func': 'ry', 'wires': [3]})
    func_list.append({'input_idx': None, 'func': 'cnot', 'wires': [0, 3]})

    return tq.GeneralEncoder(func_list)