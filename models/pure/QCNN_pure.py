"""
qml implementation for QCNN (conv1, conv2, pool1, pool2, fc)
2/3分类
"""

import pennylane as qml
import torch
import torch.nn as nn
from pennylane.templates.embeddings import AmplitudeEmbedding
import math
from models.circuits import pure_qcnn_circuit
from tools.embedding import data_embedding_qml


n_qubits = 10
dev = qml.device('default.qubit', wires=n_qubits)


@qml.qnode(dev, interface='torch')
def circuit(inputs, weights_conv1, weights_conv2, weights_pool1, weights_pool2, weights_fc):
    AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True, pad_with=0)
    # data_embedding_qml(inputs, n_qubits, e_type)
    pure_qcnn_circuit(n_qubits, weights_conv1, weights_conv2, weights_pool1, weights_pool2, weights_fc)

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


@qml.qnode(dev, interface='torch')
def circuit_state(inputs, weights_conv1, weights_conv2, weights_pool1, weights_pool2, weights_fc, exec_=True):
    AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True, pad_with=0)
    if exec_:
        pure_qcnn_circuit(n_qubits, weights_conv1, weights_conv2, weights_pool1, weights_pool2, weights_fc)
    return qml.state()


@qml.qnode(dev, interface='torch')
def circuit_dm(q_idx, inputs, weights_conv1, weights_conv2, weights_pool1, weights_pool2, weights_fc, exec_=True):
    AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True, pad_with=0)
    if exec_:
        pure_qcnn_circuit(n_qubits, weights_conv1, weights_conv2, weights_pool1, weights_pool2, weights_fc)
    return qml.density_matrix(wires=q_idx)


def get_density_matrix(inputs, weights_conv1, weights_conv2, weights_pool1, weights_pool2, weights_fc, exec_=True):
    dm_list = []
    for q in range(n_qubits):
        dm_list.append(circuit_dm(q, inputs, weights_conv1, weights_conv2, weights_pool1, weights_pool2, weights_fc, exec_))
    return dm_list


def whole_dm(inputs, weights_conv1, weights_conv2, weights_pool1, weights_pool2, weights_fc):
    l = []
    for q in range(n_qubits):
        l.append(q)
    return (circuit_dm(l, inputs, weights_conv1, weights_conv2, weights_pool1, weights_pool2, weights_fc, exec_=False),
            circuit_dm(l, inputs, weights_conv1, weights_conv2, weights_pool1, weights_pool2, weights_fc))


class QCNN_classifier(nn.Module):
    def __init__(self, num_classes=2, e='amplitude'):
        super(QCNN_classifier, self).__init__()
        self.num_classes = num_classes

        self.cir = qml.qnn.TorchLayer(circuit, {'weights_conv1': (n_qubits, 15),
                                                'weights_conv2': (math.ceil((n_qubits-2)/2), 15),
                                                'weights_pool1': (math.ceil(n_qubits/2), 2),
                                                'weights_pool2': (math.ceil((n_qubits-2)/4), 2),
                                                'weights_fc': (3,)})

    def forward(self, x, y):
        preds = self.predict(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(preds, y)
        return loss

    def predict(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.cir(x)
        if self.num_classes == 2:
            return x[:, torch.tensor([0, 2])]
        else:
            return x[:, torch.tensor([0, 2, 4])]

    def visualize_circuit(self, x, weights, save_path):
        import matplotlib.pyplot as plt
        fig, ax = qml.draw_mpl(circuit)(x, weights[0], weights[1], weights[2], weights[3], weights[4])
        fig.show()
        plt.savefig(save_path)


