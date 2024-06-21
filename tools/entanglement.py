"""
compute entanglement
Meyer_Wallach
"""
import math
import numpy as np
import torch
import scipy.linalg as la
from qiskit.quantum_info import DensityMatrix, negativity, entropy


#### Entanglement for single sample
def Meyer_Wallach(partial_traces):
    """
    [0, 1]
    Meyer_Wallach entanglement under single sample
    :param partial_traces: list, for each qubit (its density matrix)
    :return:
    """
    measure = 0
    n_qubits = len(partial_traces)
    for j in range(n_qubits):
        rho = partial_traces[j]
        rho_squared = rho ** 2
        rho_squared_trace = torch.trace(rho_squared)
        measure += 1/2 * (1-rho_squared_trace)

    en = measure * (4/n_qubits)
    return en.real

### Implement from QuanTest
def gener_distance(u, v):
    uvmat = torch.kron(u, v) - torch.kron(v, u)
    return 0.5 * (torch.linalg.norm(torch.abs(uvmat)) ** 2)

def liner_map(b, j, psi):
    newpsi = []
    num_qubits = math.ceil(math.log2(psi.size(0)))
    for i in range(num_qubits):
        delta_i2bin = ((i >> (num_qubits - 1 - j)) & 1) ^ b ^ 1
        if (delta_i2bin):
            newpsi.append(psi[i])
    return torch.tensor(newpsi)

def ent_state(psi):
    num_qubits = int(math.log2(psi.size(0)))
    res = 0.0
    for j in range(num_qubits):
        res += gener_distance(liner_map(0, j, psi), liner_map(1, j, psi))
    return res * 4 / num_qubits


def entanglement_entropy(density_matrix):
    """
    [0, log(n_qubits)]
    :param density_matrix: dm of all qubits
    :return:
    """
    rho = density_matrix.numpy()
    evals = np.maximum(np.real(la.eigvals(rho)), 0.0)
    h_val = 0.0
    for x in evals:
        if 0 < x < 1:
            h_val += -x * math.log2(x)
    # rho = DensityMatrix(rho)
    # h_val = entropy(rho)
    return h_val


def avg_negativity(density_matrix):
    """
    [0, 1]
    :param density_matrix: dm of all qubits
    :return: 
    """
    nev = 0.0
    n_qubits = int(math.log2(density_matrix.shape[0]))
    dm = DensityMatrix(density_matrix.numpy())
    for q in range(n_qubits):
        qargs = [q]
        negv = negativity(dm, qargs)
        nev += negv
    return nev / n_qubits
