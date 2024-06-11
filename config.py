
import argparse


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--version', type=str, default='tq', choices=['tq', 'qml'])

    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--result_dir', type=str, default='tmp_results')

    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion_mnist', 'emnist'])
    parser.add_argument('--structure', type=str, default='classical', choices=['classical',
                                                                        'qcl', 'ccqc', 'pure_qcnn',
                                                                        'pure_single', 'pure_multi', 'quanv', 'inception',
                                                                        'quanv_iswap', 'hnn'])
    parser.add_argument('--binary_cla', type=bool, default=True)
    parser.add_argument('--class_idx', nargs='+', type=int, default=[0, 1])
    return parser.parse_args()


