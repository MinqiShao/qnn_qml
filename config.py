
import argparse


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--version', type=str, default='tq', choices=['tq', 'qml'])

    parser.add_argument('--data_dir', type=str, default='../../shaominqi/ns219x/data')
    parser.add_argument('--result_dir', type=str, default='../../shaominqi/ns219x/tmp_results')

    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion_mnist', 'emnist'])
    parser.add_argument('--structure', type=str, default='classical', choices=['classical',
                                                                        'qcl', 'ccqc', 'pure_qcnn',
                                                                        'pure_single', 'pure_multi', 'quanv', 'inception',
                                                                        'quanv_iswap', 'hnn', 'hier'])
    parser.add_argument('--encoding', type=str, default='amplitude', choices=['amplitude', 'angle_y', 'angle_xyz',
                                                                              'hde', 'hae'])
    parser.add_argument('--reduction', type=str, default='resize', choices=['resize', 'pca', 'encoder'])
    parser.add_argument('--binary_cla', type=bool, default=True)
    parser.add_argument('--class_idx', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--resize', action='store_true')
    parser.add_argument('--data_scale', type=float, default=1.0)
    return parser.parse_args()


