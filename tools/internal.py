from models.pure import QCL, QCNN_pure, CCQC, single_encoding, multi_encoding
import torch
import pennylane as qml
from pennylane import numpy as np

##### obtain state, density matrix
def in_out_state(x, structure, params):
    if structure == 'qcl':
        in_state = QCL.circuit_state(x, params, exec_=False)
        out_state = QCL.circuit_state(x, params)
    elif structure == 'pure_qcnn':
        in_state = QCNN_pure.circuit_state(x, params[0], params[1], params[2], params[3], params[4], exec_=False)
        out_state = QCNN_pure.circuit_state(x, params[0], params[1], params[2], params[3], params[4])
    elif structure == 'ccqc':
        in_state = CCQC.circuit_state(x, params[0], params[1], params[2], exec_=False)
        out_state = CCQC.circuit_state(x, params[0], params[1], params[2])
    return in_state, out_state


def whole_density_matrix(x, structure, params):
    if structure == 'qcl':
        in_dm, out_dm = QCL.whole_dm(x, params)
    elif structure == 'pure_qcnn':
        in_dm, out_dm = QCNN_pure.whole_dm(x, params[0], params[1], params[2], params[3], params[4])
    elif structure == 'ccqc':
        in_dm, out_dm = CCQC.whole_dm(x, params[0], params[1], params[2])
    return in_dm, out_dm


def partial_density_matrix(x, structure, params):
    if structure == 'qcl':
        in_dm_list = QCL.get_density_matrix(x, params, exec_=False)
        out_dm_list = QCL.get_density_matrix(x, params)
    elif structure == 'pure_qcnn':
        in_dm_list = QCNN_pure.get_density_matrix(x, params[0], params[1], params[2], params[3], params[4], exec_=False)
        out_dm_list = QCNN_pure.get_density_matrix(x, params[0], params[1], params[2], params[3], params[4])
    elif structure == 'ccqc':
        in_dm_list = CCQC.get_density_matrix(x, params[0], params[1], params[2], exec_=False)
        out_dm_list = CCQC.get_density_matrix(x, params[0], params[1], params[2])
    return in_dm_list, out_dm_list


def circuit_pred(x, params, conf):
    if conf.structure == 'qcl':
        o = QCL.circuit(x, params)
    elif conf.structure == 'pure_qcnn':
        o = QCNN_pure.circuit(x, params[0], params[1], params[2], params[3], params[4])
    elif conf.structure == 'ccqc':
        o = CCQC.circuit(x, params[0], params[1], params[2])
    o = o[:len(conf.class_idx)]
    pred = torch.argmax(torch.tensor(o))
    return o, pred


def circuit_state(in_state, conf, params):
    if conf.structure == 'qcl':
        out_state = QCL.circuit_state(in_state, params)
    elif conf.structure == 'pure_qcnn':
        out_state = QCNN_pure.circuit_state(in_state, params[0], params[1], params[2], params[3], params[4])
    elif conf.structure == 'ccqc':
        out_state = CCQC.circuit_state(in_state, params[0], params[1], params[2])
    return out_state


def block_out(x, conf, params, depth=1, exp=False):
    # 单个样本
    if conf.structure == 'qcl':
        prob = QCL.circuit_prob(x, params, depth_=depth, exp=exp)
    elif conf.structure == 'pure_qcnn':
        prob = QCNN_pure.circuit_prob(x, params[0], params[1], params[2], params[3], params[4], depth=depth, exp=exp)
    elif conf.structure == 'ccqc':
        prob = CCQC.circuit_prob(x, params[0], params[1], params[2], depth_=depth, exp=exp)

    if exp:
        return torch.tensor(prob)
    return prob


def kernel_out(x, conf, params, exp=False):
    if conf.structure == 'pure_single':
        prob = single_encoding.feat_prob(x, params, exp=exp)
    elif conf.structure == 'pure_multi':
        prob = multi_encoding.feat_prob(x, params, exp=exp)

    if exp:
        return torch.tensor(prob)
    return prob

