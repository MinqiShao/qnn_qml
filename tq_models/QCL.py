"""
tq_models implementation for models/pure/QCL
"""
import torch
import torchquantum as tq
from torchquantum.measurement import expval


n_qubits = 10
depth = 5

class QCL_(tq.QuantumModule):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.encoder = tq.AmplitudeEncoder()

        self.cnot = tq.CNOT()
        self.rx = tq.RX(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)

        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, input):
        input = torch.flatten(input, start_dim=1)
        qdev = tq.QuantumDevice(n_wires=n_qubits, bsz=input.shape[0], device=self.device, record_op=True)
        self.encoder(qdev, input)

        for d in range(depth):
            for i in range(n_qubits-1):
                self.cnot(qdev, wires=[i, i+1])
            self.cnot(qdev, wires=[n_qubits-1, 0])
            for i in range(n_qubits):
                self.rx(qdev, wires=i)
                self.rz(qdev, wires=i)
                self.rx(qdev, wires=i)

        return self.measure(qdev)

