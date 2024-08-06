from models.pure import QCL, QCNN_pure, CCQC, single_encoding, multi_encoding, hierarchical
import torch
import pennylane as qml
from pennylane import numpy as np

##### obtain state, density matrix
def in_out_state(x, conf, params, depth=1):
    structure = conf.structure
    if structure == 'qcl':
        in_state = QCL.circuit_state(x, params, exec_=False, depth_=depth)
        out_state = QCL.circuit_state(x, params, depth_=depth)
    elif structure == 'pure_qcnn':
        in_state = QCNN_pure.circuit_state(x, params[0], params[1], params[2], params[3], params[4], exec_=False, depth_=depth)
        out_state = QCNN_pure.circuit_state(x, params[0], params[1], params[2], params[3], params[4], depth_=depth)
    elif structure == 'ccqc':
        in_state = CCQC.circuit_state(x, params[0], params[1], params[2], exec_=False, depth_=depth)
        out_state = CCQC.circuit_state(x, params[0], params[1], params[2], depth_=depth)
    elif structure == 'hier':
        in_state = hierarchical.circuit_state(x, params, u=conf.hier_u, depth_=depth, exec_=False)
        out_state = hierarchical.circuit_state(x, params, u=conf.hier_u, depth_=depth)
    elif structure == 'pure_single':
        in_state = single_encoding.feat_all(x, params, ent=True, exec_=False)
        out_state = single_encoding.feat_all(x, params, ent=True)
    elif structure == 'pure_multi':
        in_state = multi_encoding.feat_all(x, params, ent=True, exec_=False)
        out_state = multi_encoding.feat_all(x, params, ent=True)

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
    elif conf.structure == 'hier':
        o = hierarchical.circuit(x, params)
    o = o[:len(conf.class_idx)]
    pred = torch.argmax(torch.tensor(o))
    return o, pred


def block_out(x, conf, params, depth=1, exec_=True):
    # 单个样本
    if conf.structure == 'qcl':
        out = QCL.circuit_prob(x, params, depth_=depth, exec_=exec_)
    elif conf.structure == 'pure_qcnn':
        out = QCNN_pure.circuit_prob(x, params[0], params[1], params[2], params[3], params[4], depth=depth)
    elif conf.structure == 'ccqc':
        out = CCQC.circuit_prob(x, params[0], params[1], params[2], depth_=depth)
    elif conf.structure == 'hier':
        out = hierarchical.circuit_prob(x, params, u=conf.hier_u, depth_=depth)

    return out


def kernel_out(x, conf, params, ent=False):
    if conf.structure == 'pure_single':
        out = single_encoding.feat_all(x, params, ent=ent)
    elif conf.structure == 'pure_multi':
        out = multi_encoding.feat_all(x, params, ent=ent)

    if not ent:
        out = out.flatten()
    return out

