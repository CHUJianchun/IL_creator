import csv

import torch

from A_InputDataProcess.AA_read_data import *
from C_Evaluation.att_model_proxy import AttMolProxy
from D_optimization.DD1_factorization_machine import DeepFM
from cmd_args import cmd_args


def predict(smiles_list_):
    a_model = AttMolProxy('Anion')
    c_model = AttMolProxy('Cation')
    fm_model = DeepFM()
    fm_model.load_state_dict(torch.load('SavedModel/FM.pkl'))
    fm_model.eval()
    s_list = []
    for smiles in tqdm(smiles_list_):
        if smiles == '':
            s_list.append(0)
            continue
        a_smiles, c_smiles = [smiles.split('.')[0]], [smiles.split('.')[1]]
        if '+' in a_smiles and '-' in c_smiles:
            a_smiles, c_smiles = c_smiles, a_smiles
        a_z = torch.tensor(a_model.encode(smiles=a_smiles)).reshape(1, -1).to(torch.float32)
        a_z = torch.cat((a_z, torch.zeros((1, cmd_args.cation_latent_dim - cmd_args.anion_latent_dim))),
                        dim=1).to(torch.float32)
        c_z = torch.tensor(c_model.encode(smiles=c_smiles)).reshape(1, -1).to(torch.float32)
        s_list.append(fm_model(a_z, c_z).detach().cpu().numpy())
    return s_list


def predict_csv():
    with open('Graph_Mol/BO_MolGraph/smiles.csv', 'r') as f:
        table = []
        smiles_list = []
        reader = csv.reader(f)
        for row in reader:
            table.append(row)
    for line in table:
        if line:
            smiles_list.append(line[2])
    solubility_list = predict(smiles_list)
    with open('Graph_Mol/BO_MolGraph/smiles.csv', 'w') as f:
        i = 0
        writer = csv.writer(f)
        for line in table:
            if line:
                writer.writerow([line[0], line[1], line[2], line[3], line[4], line[5], solubility_list[i][0][0]])
                i += 1

    with open('Graph_Mol/GPSO_FM_MolGraph__/smiles.csv', 'r') as f:
        table = []
        smiles_list = []
        reader = csv.reader(f)
        for row in reader:
            table.append(row)
    for line in table:
        if line:
            smiles_list.append(line[2])
    solubility_list = predict(smiles_list)
    with open('Graph_Mol/GPSO_FM_MolGraph__/smiles_.csv', 'w') as f:
        i = 0
        writer = csv.writer(f)
        for line in table:
            if line:
                writer.writerow([line[0], line[1], line[2], solubility_list[i][0][0]])
                i += 1

    with open('Graph_Mol/Gradient_TLZSdiffMLP_MolGraph/smiles.csv', 'r') as f:
        table = []
        smiles_list = []
        reader = csv.reader(f)
        for row in reader:
            table.append(row)
    for line in table:
        if line:
            if line[2] != 'Fail':
                smiles_list.append(line[2])
    solubility_list = predict(smiles_list)
    with open('Graph_Mol/Gradient_TLZSdiffMLP_MolGraph/smiles_.csv', 'w') as f:
        i = 0
        writer = csv.writer(f)
        for line in table:
            if line:
                if line[2] != 'Fail':
                    writer.writerow([line[0], line[1], line[2], line[3], line[4], line[5], solubility_list[i][0][0]])
                    i += 1

    with open('Graph_Mol/Grid_MolGraph/smiles.csv', 'r') as f:
        table = []
        smiles_list = []
        reader = csv.reader(f)
        for row in reader:
            table.append(row)
    for line in table:
        if line:
            smiles_list.append(line[2])
    solubility_list = predict(smiles_list)
    with open('Graph_Mol/Grid_MolGraph/smiles_.csv', 'w') as f:
        i = 0
        writer = csv.writer(f)
        for line in table:
            if line:
                writer.writerow([line[0], line[1], line[2], solubility_list[i][0][0]])
                i += 1

    with open('Graph_Mol/PSO_FM_MolGraph_/smiles.csv', 'r') as f:
        table = []
        smiles_list = []
        reader = csv.reader(f)
        for row in reader:
            table.append(row)
    for line in table:
        if line:
            smiles_list.append(line[2])
    solubility_list = predict(smiles_list)

    with open('Graph_Mol/PSO_FM_MolGraph_/smiles_.csv', 'w') as f:
        i = 0
        writer = csv.writer(f)
        for line in table:
            if line:
                writer.writerow([line[0], line[1], line[2], solubility_list[i][0][0]])
                i += 1


def predict_cart(company='cjc'):
    # cjc
    # ionike
    with open('InputData/' + company + '_name.txt', 'rb') as f:
        name = f.readlines()
    with open('InputData/' + company + '_smiles.txt', 'rb') as f:
        smiles_list = f.readlines()
        for i in range(len(smiles_list)):
            if smiles_list[i] != b'\r\n':
                smiles_list[i] = smiles_list[i].decode('utf-8', 'ignore')[:-2]
            else:
                smiles_list[i] = ''
    a_model = AttMolProxy('Anion')
    c_model = AttMolProxy('Cation')
    fm_model = DeepFM()
    fm_model.load_state_dict(torch.load('SavedModel/FM.pkl'))
    fm_model.eval()
    s_list = []
    for smiles in tqdm(smiles_list):
        if smiles == '' or '.' not in smiles:
            s_list.append(0)
            continue
        a_smiles, c_smiles = [smiles.split('.')[0]], [smiles.split('.')[1]]
        if '+' in a_smiles and '-' in c_smiles:
            a_smiles, c_smiles = c_smiles, a_smiles
        try:
            a_z = torch.tensor(a_model.encode(smiles=a_smiles)).reshape(1, -1).to(torch.float32)
            a_z = torch.cat((a_z, torch.zeros((1, cmd_args.cation_latent_dim - cmd_args.anion_latent_dim))),
                            dim=1).to(torch.float32)
            c_z = torch.tensor(c_model.encode(smiles=c_smiles)).reshape(1, -1).to(torch.float32)
            s_list.append(fm_model(a_z, c_z).detach().cpu().numpy())
        except AssertionError:
            s_list.append(0)
    s_list = np.array(s_list)
    return name, smiles_list, s_list, name[s_list.argmax()], s_list.max()


if __name__ == '__main__':
    # test = predict(['F[B-](F)(F)F.C[N+]1=CN(C=C1)CCCCCCCC'])
    # name, smiles, sdiff, best_one, best_diff = predict_cart()
    a = predict(['Cc1ccc(S(=O)(=O)[O-])cc1.CCN1C=C[N+](=C1)C'])
    # 'Cc1ccc(S(=O)(=O)[O-])cc1.C[N+](CCCCCCCC)(CCCCCCCC)CCCCCCCC'
    pass
