"""
compute entanglement
Meyer_Wallach
"""
import math
import torch
from tqdm import tqdm
from qiskit.quantum_info import DensityMatrix, negativity, Statevector
from pennylane.math import vn_entropy, max_entropy
from tools.internal import *

def MW(test_x, params, conf, depth=1):
    in_list = []
    out_list = []
    for x in tqdm(test_x):
        x = torch.flatten(x, start_dim=0)
        in_state, out_state = in_out_state(x, conf, params, depth)
        _, ent_in, ent_out = entQ(in_state, out_state, 1)
        in_list.append(ent_in)
        out_list.append(ent_out)
    in_list = torch.tensor(in_list)
    out_list = torch.tensor(out_list)
    return in_list, out_list, out_list-in_list

def MW_kernel(test_x, params, conf):
    # for single test_x
    in_list, out_list = [], []
    x = torch.flatten(test_x, start_dim=0)
    in_state, out_state = in_out_state(x, conf, params)  # (169*16)
    for in_, out_ in zip(in_state, out_state):
        _, ent_in, ent_out = entQ(in_, out_, 1)
        in_list.append(ent_in)
        out_list.append(ent_out)
    return in_list, out_list



def entropy(test_x, params, conf):
    in_list = []
    out_list = []
    class_idx = conf.class_idx
    for x in tqdm(test_x):
        x = torch.flatten(x, start_dim=0)
        in_dm, out_dm = whole_density_matrix(x, conf.structure, params)
        in_en, out_en = entanglement_entropy(in_dm, class_idx), entanglement_entropy(out_dm, class_idx)
        in_list.append(in_en)
        out_list.append(out_en)
    in_list = torch.tensor(in_list)
    out_list = torch.tensor(out_list)
    return in_list, out_list, out_list-in_list


def neg(test_x, params, conf):
    in_list = []
    out_list = []
    for x in tqdm(test_x):
        x = torch.flatten(x, start_dim=0)
        in_state, out_state = in_out_state(x, conf.structure, params)
        in_en, out_en = avg_negativity(in_state), avg_negativity(out_state)
        in_list.append(in_en)
        out_list.append(out_en)
    in_list = torch.tensor(in_list)
    out_list = torch.tensor(out_list)
    return in_list, out_list, out_list - in_list


#### Entanglement for single sample
def Meyer_Wallach(partial_traces):
    """
    [0, 1] 越大纠缠越大
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

### QuanTest
def gener_distance(u, v):
    uvmat = torch.kron(u, v) - torch.kron(v, u)
    return 0.5 * (torch.linalg.norm(torch.abs(uvmat)) ** 2)

def liner_map(b, j, psi):
    newpsi = []
    num_qubits = math.ceil(math.log2(psi.size(0)))
    for i in range(psi.size(0)):
        delta_i2bin = ((i >> (num_qubits - 1 - j)) & 1) ^ b ^ 1
        if (delta_i2bin):
            newpsi.append(psi[i].unsqueeze(0))
    return torch.cat(newpsi)

def ent_state(psi):
    num_qubits = math.ceil(math.log2(psi.size(0)))
    res = 0.0
    for j in range(num_qubits):
        res += gener_distance(liner_map(0, j, psi), liner_map(1, j, psi))
    return res * 4 / num_qubits

def entQ(in_state, out_state, k):
    ent_in = ent_state(in_state)
    ent_out = ent_state(out_state)
    return k*(ent_out - ent_in), ent_in, ent_out


def entanglement_entropy(density_matrix, class_idx=[0, 1]):
    """
    [0, log(n_qubits)]
    :param density_matrix: dm of all qubits
    :return:
    """
    # rho = density_matrix.numpy()
    # evals = np.maximum(np.real(la.eigvals(rho)), 0.0)
    # h_val = 0.0
    # for x in evals:
    #     if 0 < x < 1:
    #         h_val += -x * math.log2(x)
    # rho = DensityMatrix(rho)
    # h_val = entropy(rho)
    # todo sub_sys如何选取
    sub_sys = [2]
    h_val = vn_entropy(density_matrix, indices=sub_sys, base=2)
    max_en = max_entropy(density_matrix, indices=sub_sys, base=2)
    return h_val / max_en


def avg_negativity(state_vector):
    """
    [0, 1]
    :param state_vector
    :return: 
    """
    nev = 0.0
    n_qubits = int(math.log2(state_vector.shape[0]))
    state_vector = Statevector(state_vector.numpy())
    for q in range(n_qubits):
        qargs = [q]
        negv = negativity(state_vector, qargs)
        nev += negv
    return nev / n_qubits
