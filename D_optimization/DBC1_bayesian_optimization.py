import os
from tqdm import tqdm
import numpy as np
import csv

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from sklearn.gaussian_process import GaussianProcessRegressor
from skopt import gp_minimize

from rdkit import Chem
from rdkit.Chem import Draw

from cmd_args import cmd_args
from C_Evaluation.att_model_proxy import AttMolProxy, batch_decode
from D_optimization.DBB_train_transfer_learning import OneHotTransferMLP, ZTransferMLP, OneHotSDiff, ZSDiff
from D_optimization.DBA_transfer_learning_data_preperation import TLDataset, SDiffDataset


class TLDataset(Dataset):
    def __init__(self, onehot_list, z_list, mw_list, nve_list, tpsa_list, hka_list, lasa_list):
        self.onehot = Variable(torch.from_numpy(np.array(onehot_list)))
        self.z = Variable(torch.from_numpy(np.array(z_list)))
        self.mw = Variable(torch.from_numpy(np.array(mw_list)))
        self.nve = Variable(torch.from_numpy(np.array(nve_list)))
        self.tpsa = Variable(torch.from_numpy(np.array(tpsa_list)))
        self.hka = Variable(torch.from_numpy(np.array(hka_list)))
        self.lasa = Variable(torch.from_numpy(np.array(lasa_list)))

    def __getitem__(self, index):
        return self.onehot[index], self.z[index], Variable(
            torch.Tensor([self.mw[index], self.nve[index], self.tpsa[index], self.hka[index], self.lasa[index]]))

    def __len__(self):
        return len(self.onehot)


def bayesian_optimization(model_name, lower_bond=-1, upper_bond=1, times=1):
    a_model = AttMolProxy('Anion')
    c_model = AttMolProxy('Cation')
    x = np.load('InputData/' + model_name + '-x.npy')
    x = x.reshape(x.shape[0], -1)
    y = np.load('InputData/' + model_name + '-y.npy')
    y = y.reshape(y.shape[0], -1)
    gpr = GaussianProcessRegressor(random_state=0).fit(x, y)
    print(gpr.score(x, y))
    file_name = 'Graph_Mol/BO_' + model_name.replace('.pkl') + 'MolGraph'
    if not os.path.exists(file_name + '/'):
        os.makedirs(file_name + '/')

    file_list = os.listdir(file_name)
    for file in file_list:
        os.remove(file_name + '/' + file)

    def predict_s_diff(x_):
        s_diff, std = gpr.predict(np.array(x_).reshape(1, -1))
        return -(s_diff + std)[0, 0]

    bound = []
    for i_ in range(cmd_args.anion_latent_dim + cmd_args.cation_latent_dim):
        bound.append((lower_bond, upper_bond))

    for time in range(times):
        res = gp_minimize(func=predict_s_diff, dimensions=bound, n_calls=10)
        file_list = os.listdir(file_name + '/')
        if 'smiles.csv' in file_list:
            file_list.remove('smiles.csv')
        for i_ in range(len(file_list)):
            file_list[i_] = int(file_list[i_].strip('.png'))
        if len(file_list) > 0:
            name_start = np.max(np.array(file_list))
        else:
            name_start = 0
        for i_ in range(len(res.x_iters)):
            x_iter = res.x_iters[i_]
            a_smiles = a_model.decode(
                np.array(x_iter[:cmd_args.anion_latent_dim], dtype='f').reshape(1, -1), use_random=False)
            c_smiles = c_model.decode(
                np.array(x_iter[cmd_args.anion_latent_dim:], dtype='f').reshape(1, -1), use_random=False)
            il_smiles = a_smiles[0] + '.' + c_smiles[0]
            if '-' in il_smiles and '+' in il_smiles:
                try:
                    Draw.MolToFile(
                        Chem.MolFromSmiles(il_smiles), file_name + '/' + str(name_start + 1) + '.png')
                    with open(file_name + '/smiles.csv', 'a+') as f_:
                        csv_write = csv.writer(f_)
                        csv_write.writerow([str(name_start + 1), il_smiles, a_smiles, c_smiles,
                                            gpr.predict(np.array(x_iter).reshape(1, -1))])
                    name_start += 1

                except ValueError:
                    pass


if __name__ == '__main__':
    # bayesian_optimization(model_name='TLZSdiffMLP.pkl', times=10)
    # bayesian_optimization(model_name='ZSdiffMLP.pkl', times=10)
    pass
    # fail with less than 132 GiB memory
