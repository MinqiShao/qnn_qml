"""
tq_models implementation for models/pure/QCNN
--> 目前只支持2/3分类，需要resize
"""

import torchquantum as tq
import torch


n_qubits = 8


class Conv1(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.u3 = tq.U3(has_params=True, trainable=True)
        self.cnot = tq.CNOT()
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)


    def forward(self, qdev):
        for i in range(0, 8, 2):
            self.u3(qdev, wires=i)
            self.u3(qdev, wires=i + 1)
            self.cnot(qdev, wires=[i, i+1])
            self.ry(qdev, wires=i)
            self.rz(qdev, wires=i + 1)
            self.cnot(qdev, wires=[i+1, i])
            self.ry(qdev, wires=i)
            self.cnot(qdev, wires=[i, i+1])
            self.u3(qdev, wires=i)
            self.u3(qdev, wires=i + 1)
        for i in range(1, 7, 2):
            self.u3(qdev, wires=i)
            self.u3(qdev, wires=i + 1)
            self.cnot(qdev, wires=[i, i + 1])
            self.ry(qdev, wires=i)
            self.rz(qdev, wires=i + 1)
            self.cnot(qdev, wires=[i + 1, i])
            self.ry(qdev, wires=i)
            self.cnot(qdev, wires=[i, i + 1])
            self.u3(qdev, wires=i)
            self.u3(qdev, wires=i + 1)
        self.u3(qdev, wires=0)
        self.u3(qdev, wires=7)
        self.cnot(qdev, wires=[0, 7])
        self.ry(qdev, wires=0)
        self.rz(qdev, wires=7)
        self.cnot(qdev, wires=[7, 0])
        self.ry(qdev, wires=0)
        self.cnot(qdev, wires=[0, 7])
        self.u3(qdev, wires=0)
        self.u3(qdev, wires=7)


class Conv2(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.u3 = tq.U3(has_params=True, trainable=True)
        self.cnot = tq.CNOT()
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)


    def forward(self, qdev):
        for i in range(0, 6, 2):
            self.u3(qdev, wires=i)
            self.u3(qdev, wires=i + 2)
            self.cnot(qdev, wires=[i, i + 2])
            self.ry(qdev, wires=i)
            self.rz(qdev, wires=i + 2)
            self.cnot(qdev, wires=[i + 2, i])
            self.ry(qdev, wires=i)
            self.cnot(qdev, wires=[i, i + 2])
            self.u3(qdev, wires=i)
            self.u3(qdev, wires=i + 2)


class Pool1(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.x = tq.PauliX()
        self.crz = tq.CRZ(has_params=True, trainable=True)
        self.crx = tq.CRX(has_params=True, trainable=True)

    def forward(self, qdev):
        # 保留偶数序的qubits
        for i in range(0, 8, 2):
            self.crz(qdev, wires=[i+1, i])
            self.x(qdev, wires=i+1)
            self.crx(qdev, wires=[i+1, i])


class Pool2(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.x = tq.PauliX()
        self.crz = tq.CRZ(has_params=True, trainable=True)
        self.crx = tq.CRX(has_params=True, trainable=True)

    def forward(self, qdev):
        for i in range(0, 6, 4):
            self.crz(qdev, wires=[i+2, i])
            self.x(qdev, wires=i+2)
            self.crx(qdev, wires=[i+2, i])


class FC(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.cnot = tq.CNOT()
        self.rx = tq.RX(has_params=True, trainable=True)

    def forward(self, qdev):
        self.cnot(qdev, wires=[0, 4])
        self.cnot(qdev, wires=[2, 4])
        self.cnot(qdev, wires=[4, 0])
        self.rx(qdev, wires=0)
        self.rx(qdev, wires=2)
        self.rx(qdev, wires=4)


class QCNN_(tq.QuantumModule):
    def __init__(self, device, num_classes=2):
        super().__init__()
        self.device = device
        self.num_class = num_classes
        self.encoder = tq.AmplitudeEncoder()
        self.conv1 = Conv1()
        self.pool1 = Pool1()
        self.conv2 = Conv2()
        self.pool2 = Pool2()
        self.fc = FC()

        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, inputs):
        inputs = torch.flatten(inputs, start_dim=1)
        qdev = tq.QuantumDevice(n_wires=n_qubits, bsz=inputs.shape[0], device=self.device, record_op=True)
        self.encoder(qdev, inputs)
        self.conv1(qdev)
        self.pool1(qdev)
        self.conv2(qdev)
        self.pool2(qdev)
        self.fc(qdev)

        results = self.measure(qdev)
        # 只取第0, 2, 4个qubit的结果
        if self.num_class == 2:
            return results[:, torch.tensor([0, 2])]
        else:
            return results[:, torch.tensor([0, 2, 4])]

