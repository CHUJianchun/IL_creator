import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import sys
from matplotlib import pyplot as plt
from torch.autograd import Variable
from tqdm import tqdm

from cmd_args import cmd_args
from A_InputDataProcess.AA_read_data import *
from C_Evaluation.att_model_proxy import AttMolProxy


def load_encode():
    a_model = AttMolProxy('Anion')
    c_model = AttMolProxy('Cation')
    all_x = np.load('InputData/' + 'TLZSdiffMLP.pkl' + '-x.npy')
    all_x = all_x.reshape(all_x.shape[0], -1)
    all_y = np.load('InputData/' + 'TLZSdiffMLP.pkl' + '-y.npy')
    all_y = all_y.reshape(all_y.shape[0], -1)
    known_x = []
    known_y = []
    gp_dataset = unpickle_gp_dataset()
    for data in gp_dataset:
        known_x.append(np.concatenate(
            (a_model.encode(one_hot=data.il.anion.onehot.reshape(
                1, data.il.anion.onehot.shape[0], data.il.anion.onehot.shape[1])),
                            c_model.encode(one_hot=data.il.cation.onehot.reshape(
                                1, data.il.cation.onehot.shape[0], data.il.cation.onehot.shape[1]))
            ), axis=1).reshape(-1))
        known_y.append(data.s_diff)
    known_x = np.array(known_x)
    known_y = np.array(known_y)
    return all_x, all_y, known_x, known_y


def principle_component_analysis(colorbar='viridis'):
    # (np.var(all_x[:, :48]) / np.var(all_x[:, 48:])) ** 0.5 = 2.153096532979506
    all_x, all_y, known_x, known_y = load_encode()
    all_x[:, :cmd_args.anion_latent_dim] *= 2.153096532979506
    known_x[:, cmd_args.anion_latent_dim:] *= 2.153096532979506
    pca = PCA(n_components=2)
    pca.fit(all_x)
    pca_all = pca.transform(all_x)
    pca_known = pca.transform(known_x)

    cm = plt.cm.get_cmap(colorbar)
    plt.scatter(pca_all[:, 0], pca_all[:, 1], c=all_y,
                vmin=np.min(all_y), vmax=np.max(all_y), s=5, cmap=cm, alpha=0.5)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig('Graph_Analyze/all_color.png')

    plt.cla()

    plt.scatter(pca_all[:, 0], pca_all[:, 1], c='grey', s=5, alpha=0.2)
    plt.scatter(pca_known[:, 0], pca_known[:, 1], c=known_y,
                vmin=np.min(known_y), vmax=np.max(known_y), s=5, cmap=cm, alpha=0.8)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig('Graph_Analyze/known_color.png')


def t_distributed_stochastic_neighbor_embedding(colorbar='viridis'):
    all_x, all_y, known_x, known_y = load_encode()
    all_x[:, :cmd_args.anion_latent_dim] *= 2.153096532979506
    known_x[:, cmd_args.anion_latent_dim:] *= 2.153096532979506
    all_x = np.array(all_x, dtype='float64')
    all_y = np.array(all_y, dtype='float64')
    known_x = np.array(known_x, dtype='float64')
    known_y = np.array(known_y, dtype='float64')
    tsne = TSNE(n_jobs=5)
    tsne_all_x = tsne.fit_transform(all_x)
    tsne_known_x = np.zeros((known_x.shape[0], 2))
    for i in trange(known_x.shape[0]):
        for j in range(all_x.shape[0]):
            if all(known_x[i] == all_x[j]):
                tsne_known_x[i] = tsne_all_x[j]
                break
    np.save('InputData/tsne_all_x.npy', tsne_all_x)
    np.save('InputData/tsne_known_x.npy', tsne_known_x)

    cm = plt.cm.get_cmap(colorbar)
    plt.scatter(tsne_all_x[:, 0], tsne_all_x[:, 1], c=all_y,
                vmin=np.min(all_y), vmax=np.max(all_y), s=5, cmap=cm, alpha=0.5)
    plt.colorbar()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig('Graph_Analyze/tsne_all_color.png')

    plt.cla()

    cm = plt.cm.get_cmap(colorbar)
    plt.scatter(tsne_all_x[:, 0], tsne_all_x[:, 1], c='grey', marker='x', s=5, alpha=0.8)
    plt.scatter(tsne_known_x[:, 0], tsne_known_x[:, 1], c=known_y,
                vmin=np.min(known_y), vmax=np.max(known_y), s=5, cmap=cm, alpha=0.2)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.colorbar()
    plt.savefig('Graph_Analyze/tsne_known_color.png')

    plt.cla()
    all_y_real = np.zeros(all_y.shape)
    for i in range(all_y_real.shape[0]):
        if all_y[i] < 0:
            all_y_real[i] = 0
        else:
            all_y_real[i] = all_y[i]
    cm = plt.cm.get_cmap(colorbar)
    plt.scatter(tsne_all_x[:, 0], tsne_all_x[:, 1], c=all_y_real,
                vmin=np.min(all_y_real), vmax=np.max(all_y_real), s=5, cmap=cm, alpha=0.5)
    plt.colorbar()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig('Graph_Analyze/tsne_all_color_real.png')


if __name__ == '__main__':
    # all_x, all_y, known_x, known_y = load_encode()
    #  principle_component_analysis()
    t_distributed_stochastic_neighbor_embedding()
    pass
