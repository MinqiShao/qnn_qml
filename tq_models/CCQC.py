"""
tq_models implementation for models/pure/CCQC
"""

import torchquantum as tq
from torchquantum.measurement import expval

import torch


n_qubits = 10
depth = 5


class CCQC_(tq.QuantumModule):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.encoder = tq.AmplitudeEncoder()

        self.rx = tq.RX(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)
        # todo 原来是CP门
        self.crot = tq.CRot(has_params=True, trainable=True)

        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, input):
        input = torch.flatten(input, start_dim=1)
        qdev = tq.QuantumDevice(n_wires=n_qubits, bsz=input.shape[0], device=self.device, record_op=True)
        self.encoder(qdev, input)
        for d in range(1, depth + 1):
            if d % 2:
                for i in range(n_qubits):
                    self.rx(qdev, wires=i)
                    self.rz(qdev, wires=i)
                    self.rx(qdev, wires=i)
                self.crot(qdev, wires=[0, n_qubits-1])
                self.rx(qdev, wires=n_qubits-1)
                for i in range(1, n_qubits):
                    self.crot(qdev, wires=[n_qubits-i, n_qubits-i-1])
                    self.rx(qdev, wires=n_qubits-i-1)
            else:
                for i in range(n_qubits):
                    self.rx(qdev, wires=i)
                    self.rz(qdev, wires=i)
                    self.rx(qdev, wires=i)
                j = 0
                for i in range(n_qubits):
                    nj = (j+(n_qubits-3)) % n_qubits
                    self.crot(qdev, wires=[j, nj])
                    self.rx(qdev, wires=nj)
                    j = nj
        result = self.measure(qdev)  # (bs, n_qubits)  n_qubits > num_class
        return result
