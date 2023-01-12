import pickle
import numpy as np
import argparse
import collections
import os

os.environ['CONDA_DLL_SEARCH_MODIFICATION_ENABLE'] = '1'
import csv
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from skopt import gp_minimize

from rdkit import Chem
from rdkit.Chem import Draw

from cmd_args import cmd_args
from C_Evaluation.att_model_proxy import AttMolProxy, batch_decode


def bayesian_optimization(lower_bond=-0.8, upper_bond=0.8, times=1):
    a_model = AttMolProxy('Anion')
    c_model = AttMolProxy('Cation')
    X = np.load('InputData/gp-features.npy')
    y = np.load('InputData/gp-solubility-difference.npy')
    gpr = GaussianProcessRegressor(random_state=0).fit(X, y)

    def predict_s_diff(x_):
        s_diff = gpr.predict(np.array(x_).reshape(1, -1))
        return -s_diff[0, 0]

    if not os.path.exists('Graph_Mol/BO_MolGraph/'):
        os.makedirs('Graph_Mol/BO_MolGraph/')

    file_list = os.listdir('Graph_Mol/BO_MolGraph/')
    for file in file_list:
        os.remove('Graph_Mol/BO_MolGraph/' + file)

    bound = []
    for i_ in range(cmd_args.anion_latent_dim + cmd_args.cation_latent_dim):
        bound.append((lower_bond, upper_bond))
    for time in range(times):
        res = gp_minimize(func=predict_s_diff, dimensions=bound, n_initial_points=100)

        for i in range(len(file_list)):
            if '.png' in file_list[i]:
                file_list[i] = int(file_list[i].strip('.png'))
            else:
                file_list[i] = 0
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
                    Draw.MolToFile(Chem.MolFromSmiles(il_smiles), 'Graph_Mol/BO_MolGraph/' + str(
                        name_start + 1) + '_' + str(i_ + 1) + '.png')
                    with open('Graph_Mol/BO_MolGraph/smiles.csv', 'a+') as f_:
                        csv_write = csv.writer(f_)
                        csv_write.writerow([str(name_start + 1), str(i_ + 1), il_smiles, a_smiles, c_smiles,
                                            gpr.predict(np.array(x_iter).reshape(1, -1))[0][0]])
                    name_start += 1

                except ValueError:
                    pass


if __name__ == '__main__':
    bayesian_optimization(times=50)
