import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from scipy.stats import entropy


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


def multi_heatmap_graph(data, k, save_path, l=0, u=1):
    labels = ['|00>', '|01>', '|10>', '|11>']
    #labels = ['|0>', '|1>']
    data_list = []
    ent = np.zeros((len(data)))
    bins = np.linspace(l, u, k + 1)
    for idx, d in enumerate(data):
        hist, _ = np.histogram(d, bins=bins)
        ent[idx] = entropy(hist/hist.sum())
        data_list.append(hist)

    fig, axes = plt.subplots(len(data_list), 1, figsize=(k/2, len(data_list)))
    for ax, data, label in zip(axes, data_list, labels):
        heatmap = ax.imshow(np.array([data]), cmap="Blues")

        # Set the x-axis labels at the edges
        xticks = np.arange(-0.5, len(data), k / 10)
        xticklabels = [f'{x:.1f}' for x in bins]
        xticklabels_ = [xticklabels[int(i)] for i in xticks + 0.5]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels_)

        ax.set_yticks([])

        # Add text annotations
        for i in range(len(data)):
            ax.text(i, 0, str(data[i]), ha='center', va='center', color='black')

        # Add y-axis label
        ax.set_ylabel(label, rotation=0, ha='left', va='center', fontsize=12, fontweight='bold', labelpad=40)

        ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()
    plt.savefig(save_path)

    return ent

def single_heatmap_graph(data, k, save_path, l=0, u=1):
    # ent
    bins = np.linspace(l, u, k + 1)
    hist, _ = np.histogram(data, bins=bins)  # 统计次数

    fig, ax = plt.subplots(figsize=(k/2, 1))
    heatmap = ax.imshow(np.array([hist]), cmap="Blues", aspect='auto')

    xticks = np.arange(-0.5, len(hist), k/10)
    xticklabels = [f'{x:.2f}' for x in bins]
    xticklabels_ = [xticklabels[int(i)] for i in xticks+0.5]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels_)
    ax.set_yticks([])
    for i in range(len(hist)):
        ax.text(i, 0, str(hist[i]), ha='center', va='center', color='black')
    plt.tight_layout()
    plt.show()
    plt.savefig(save_path)

    ent = entropy(hist/hist.sum())

    return ent
