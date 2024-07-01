import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def line_graph(data):
    pass


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
