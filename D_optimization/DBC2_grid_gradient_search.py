import os
from tqdm import tqdm, trange
import numpy as np
import csv

import torch
from torch.autograd import Variable

from rdkit import Chem
from rdkit.Chem import Draw

from cmd_args import cmd_args
from C_Evaluation.att_model_proxy import AttMolProxy, batch_decode
from DBB_train_transfer_learning import OneHotTransferMLP, ZTransferMLP, OneHotSDiff, ZSDiff


def grid_gradient_search(model_name, limitation=(-1, 1), random_point=10, momentum=0.004, times=200):
    a_model = AttMolProxy('Anion')
    c_model = AttMolProxy('Cation')

    file_name = 'Graph_Mol/Gradient_' + model_name.replace('.pkl', '')+ '_MolGraph'

    if not os.path.exists(file_name + '/'):
        os.makedirs(file_name + '/')

    file_list = os.listdir(file_name)
    for file in file_list:
        os.remove(file_name + '/' + file)

    mlp = ZSDiff().cuda()
    mlp.load_state_dict(torch.load('SavedModel/' + model_name))
    for param in mlp.parameters():
        param.requires_grad = False

    coordinates = Variable(torch.rand(size=(random_point, cmd_args.anion_latent_dim + cmd_args.cation_latent_dim)))
    coordinates = (coordinates * (limitation[1] - limitation[0]) + limitation[0]).to(torch.float32).detach().cuda()

    optimized_coordinate = np.zeros((random_point, times, cmd_args.anion_latent_dim + cmd_args.cation_latent_dim))
    optimized_result = np.zeros((random_point, times))

    for coordinate_num in trange(len(coordinates)):
        for epoch in range(times):
            mlp.eval()
            coordinate = coordinates[coordinate_num].reshape(1, -1)
            coordinate.requires_grad_(True)
            output = -mlp(coordinate)
            optimizer = torch.optim.Adam([coordinate], lr=momentum)
            optimizer.zero_grad()
            output.backward()
            optimizer.step()
            result = mlp(coordinate).detach().cpu().numpy()
            coordinates[coordinate_num] = coordinate.detach().reshape(coordinates[coordinate_num].shape)
            optimized_result[coordinate_num, epoch] = result
            optimized_coordinate[coordinate_num, epoch] = coordinates[coordinate_num].detach().cpu().numpy()
            a_smiles = a_model.decode(
                np.array(
                    optimized_coordinate[coordinate_num, epoch][:cmd_args.anion_latent_dim].reshape(
                        1, -1), dtype='f'), use_random=False)
            c_smiles = c_model.decode(
                np.array(
                    optimized_coordinate[coordinate_num, epoch][cmd_args.anion_latent_dim:].reshape(
                        1, -1), dtype='f'), use_random=False)
            il_smiles = a_smiles[0] + '.' + c_smiles[0]
            if '-' in il_smiles and '+' in il_smiles:
                img_path = file_name + '/point_' + str(coordinate_num + 1) + 'epoch_' + str(epoch + 1) + '.png'
                try:
                    Draw.MolToFile(
                        Chem.MolFromSmiles(il_smiles), img_path)
                    with open(file_name + '/smiles.csv', 'a+') as f_:
                        csv_write = csv.writer(f_)
                        csv_write.writerow([str(coordinate_num + 1), str(epoch + 1), il_smiles, a_smiles, c_smiles,
                                            optimized_result[coordinate_num, epoch]])
                except ValueError:
                    with open(file_name + '/smiles.csv', 'a+') as f_:
                        csv_write = csv.writer(f_)
                        csv_write.writerow([str(coordinate_num + 1), str(epoch + 1), 'Fail'])
                    pass


if __name__ == '__main__':
    # grid_gradient_search('TLZSdiffMLP.pkl')
    grid_gradient_search('ZSdiffMLP.pkl')
