import csv

import numpy as np
from tqdm import tqdm, trange

from rdkit import Chem
from rdkit.Chem import Draw

from cmd_args import cmd_args
from A_InputDataProcess.AA_read_data import *
from C_Evaluation.att_model_proxy import AttMolProxy


def get_best_known():
    a_model = AttMolProxy('Anion')
    c_model = AttMolProxy('Cation')
    known_x = []
    known_y = []
    gp_dataset = unpickle_gp_dataset()
    for data in gp_dataset:
        known_x.append(np.concatenate((a_model.encode(one_hot=data.il.anion.onehot.reshape(
                1, data.il.anion.onehot.shape[0], data.il.anion.onehot.shape[1])),
                            c_model.encode(one_hot=data.il.cation.onehot.reshape(
                                1, data.il.cation.onehot.shape[0], data.il.cation.onehot.shape[1]))), axis=1))
        known_y.append(data.s_diff)
    known_x = np.array(known_x)
    known_y = np.array(known_y)
    return known_x, known_y, known_x[np.argmax(known_y)], np.argmax(known_y)


def get_best_known_neighbor(num=1):
    known_x, known_y, best_x, best_y = get_best_known()
    best_x = best_x.reshape(-1)
    best_a = best_x[:cmd_args.anion_latent_dim]
    best_c = best_x[cmd_args.anion_latent_dim:]
    all_x = np.load('InputData/' + 'TLZSdiffMLP.pkl' + '-x.npy')
    all_x = all_x.reshape(all_x.shape[0], -1)
    all_a = np.unique(all_x[:, :cmd_args.anion_latent_dim], axis=0)
    all_c = np.unique(all_x[:, cmd_args.anion_latent_dim:], axis=0)
    distance_a = np.zeros(all_a.shape[0])
    distance_c = np.zeros(all_a.shape[0])
    for d in trange(distance_a.shape[0]):
        distance_a[d] = np.sum((all_a[d] - best_a) ** 2) ** 0.5
        distance_c[d] = np.sum((all_c[d] - best_c) ** 2) ** 0.5
    neighbor_a_distance = np.zeros(num)
    neighbor_c_distance = np.zeros(num)
    for i in trange(num):
        neighbor_a_distance[i] = np.min(distance_a)
        neighbor_c_distance[i] = np.min(distance_c)
        distance_a[np.argmin(distance_a)] = 100000
        distance_c[np.argmin(distance_c)] = 100000
    return neighbor_a_distance, neighbor_c_distance


def random_search(a_limit=None, c_limit=None, sample=20, rand=False):
    if a_limit is None or c_limit is None:
        nad, ncd = get_best_known_neighbor(num=5)
        a_limit = nad[2]
        c_limit = ncd[2]
    known_x, known_y, best_x, best_y = get_best_known()
    del known_x, known_y
    best_x = best_x.reshape(-1)
    best_a = best_x[:cmd_args.anion_latent_dim]
    best_c = best_x[cmd_args.anion_latent_dim:]
    sample_a = np.random.randn(sample, cmd_args.anion_latent_dim) / 2 * a_limit + best_a.reshape(1, -1)
    sample_c = np.random.randn(sample, cmd_args.cation_latent_dim) / 2 * c_limit + best_c.reshape(1, -1)
    a_model = AttMolProxy('Anion')
    c_model = AttMolProxy('Cation')
    a_list = []
    c_list = []
    for sample_ in sample_a:
        a_list.append(a_model.decode(np.array(sample_.reshape(1, -1), dtype='f'), use_random=rand)[0])
    for sample_ in sample_c:
        c_list.append(c_model.decode(np.array(sample_.reshape(1, -1), dtype='f'), use_random=rand)[0])
    i = 1
    j = 1
    with open('Graph_Mol/Grid_MolGraph/smiles.csv', 'wb') as f_:
        pass

    for a in a_list:
        for c in c_list:
            smiles = a + '.' + c
            if '-' in smiles and '+' in smiles:
                img_path = 'Graph_Mol/Grid_MolGraph/point_anion_' + str(i) + '_cation_' + str(j) + '.png'
                try:
                    Draw.MolToFile(Chem.MolFromSmiles(smiles), img_path)
                    with open('Graph_Mol/Grid_MolGraph/smiles.csv', 'a+') as f_:
                        csv_write = csv.writer(f_)
                        csv_write.writerow([str(i), str(j), smiles])
                except ValueError:
                    with open('Graph_Mol/Grid_MolGraph/smiles.csv', 'a+') as f_:
                        csv_write = csv.writer(f_)
                        csv_write.writerow([str(i), str(j - i * sample), 'Failed'])
            j += 1
        i += 1


if __name__ == '__main__':
    # known_x, known_y, best_x, best_y = get_best_known()
    # neighbor_a_distance, neighbor_c_distance = get_best_known_neighbor(num=20)
    # [0.         2.00556814 2.03061877 2.27638436 2.30045512 2.3225733, 2.36694204 2.4083538  2.53437744 2.57456873
    # 2.78644    2.81855284, 2.9960694  3.04927446 3.07467101 3.09539535 3.29324908 3.30657977, 3.31423013 3.38578162]
    # [0.15603292 0.22431947 0.22591399 0.27616147 0.32392084 0.35307392, 0.3770435  0.37860678 0.3789659  0.38638
    # 0.39639648 0.40116276, 0.4047683  0.40858128 0.42054945 0.42152457 0.44566723 0.45221232, 0.47600266 0.49413093]
    random_search(a_limit=0.203, c_limit=0.016)
    pass
