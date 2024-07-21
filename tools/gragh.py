import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd


def train_line(e_l, train_l, test_l, save_path, type_='loss'):
    """
    训练过程的train loss/acc和test loss/acc变化折线
    :param type_: loss or accuracy
    :param e_l: epoch list
    :param train_l: list
    :param test_l: list
    :param save_path:
    :return:
    """
    plt.figure(figsize=(10, 5))
    plt.plot(e_l, train_l, label='train_' + type_, color='blue')
    plt.plot(e_l, test_l, label='test_' + type_, color='orange')
    plt.xlabel('epochs')
    plt.ylabel(type_)
    plt.legend()
    plt.xticks(e_l)
    if type_ == 'accuracy':
        plt.yticks(range(0, 101, 10))
    plt.tight_layout()
    plt.show()
    plt.savefig(save_path)


def line_graph(x, y, save_path, label_n='ent_out'):
    """
    折线图
    :param x: 横坐标list，train.py: epoches
    :param y: 纵坐标list, train.py: [mean_ent_out] or [mean_ent_c]
    :param save_path:
    :return:
    """
    x = [i+1 for i in x]
    fig = plt.figure(figsize=(10, 5))
    plt.plot(x, y, c='g', label=label_n)
    plt.xlabel('Epoch')
    plt.ylabel('Entanglement')
    plt.legend(loc='upper right')
    # y_sticks = range(0, 1, 0.01)
    # plt.yticks(y_sticks)
    plt.xticks(x)
    plt.show()
    plt.savefig(save_path)


def box_graph(x, y, data_num, save_path):
    """
    箱图
    :param x: train.py: [epoch]
    :param y: train.py: [[ent_in], [ent_out], [ent_c]]
    :param save_path:
    :return:
    """
    data = {
        'Epoch': [], 'Ent': [], 'group': []
    }
    for epoch in x:
        data['Epoch'] += [epoch+1]*data_num*3
        data['Ent'] += y[0][epoch] + y[1][epoch] + y[2][epoch]
        data['group'] += ['ent_in'] * data_num + ['ent_out'] * data_num + ['ent_c'] * data_num
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='Epoch', y='Ent', hue='group', data=df, showfliers=True)
    # plt.xlabel('Epoch')
    # plt.ylabel('Entanglement')
    plt.legend(loc='upper right')
    plt.show()
    plt.savefig(save_path)


def dot_graph(data, num_per_class, save_path):
    """
    data distribution using features from block prob
    :param data: (n, 2**n_qubits) n: num from all classes
    :param num_per_class:
    :param save_path:
    :return:
    """
    tsne = TSNE(n_components=2)
    tsne_x = tsne.fit_transform(data)

    fig = plt.figure(figsize=(10, 10))
    fig_, ax = plt.subplots(figsize=(10, 10))
    class_num = int(data.shape[0]/num_per_class)
    col = ['b', 'r', 'g', 'y']
    for c in range(class_num):
        ax.scatter(tsne_x[num_per_class*c:num_per_class*(c+1), 0], tsne_x[num_per_class*c:num_per_class*(c+1), 1],
                   c=col[c], label='class ' + str(c))
    ax.legend()
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()
    plt.savefig(save_path)
