
import argparse


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--version', type=str, default='qml', choices=['tq', 'qml'])

    parser.add_argument('--data_dir', type=str, default='../../shaominqi/ns219x/data')
    parser.add_argument('--model_dir', type=str, default='../../shaominqi/ns219x/models')
    parser.add_argument('--analysis_dir', type=str, default='../../shaominqi/ns219x/analysis')
    parser.add_argument('--visual_dir', type=str, default='../../shaominqi/ns219x/visual_results')

    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion_mnist', 'emnist'])
    parser.add_argument('--structure', type=str, default='classical', choices=['classical',
                                                                        'qcl', 'ccqc', 'pure_qcnn',
                                                                        'pure_single', 'pure_multi',
                                                                        'hier', 'drqnn'])
    parser.add_argument('--hier_u', type=str, default='U_SU4', choices=['U_SU4', 'U_TTN', 'U_5', 'U_6', 'U_9', 'U_13', 'U_14', 'U_15', 'U_SO4'])
    parser.add_argument('--encoding', type=str, default='amplitude', choices=['amplitude', 'angle_y', 'angle_xyz',
                                                                              'hde', 'hae'])
    parser.add_argument('--reduction', type=str, default='resize', choices=['resize', 'pca', 'encoder'])
    parser.add_argument('--class_idx', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--resize', action='store_true')
    parser.add_argument('--data_scale', type=float, default=1.0)
    # analysis
    parser.add_argument('--criteria', type=str, default='prob', choices=['prob', 'ent'])
    # coverage
    parser.add_argument('--num_test', type=int, default=10)
    parser.add_argument('--num_train', type=int, default=100)
    parser.add_argument('--cov_cri', type=str, default='prob', choices=['prob', 'exp', 'ent'])
    parser.add_argument('--with_adv', action='store_true')
    # gen adv
    parser.add_argument('--attack', type=str, default='DLFuzz', choices=['DLFuzz', 'FGSM', 'BIM', 'CW', 'QuanTest', 'DIFGSM', 'JSMA'])
    return parser.parse_args()


