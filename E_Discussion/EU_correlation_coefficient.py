import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from tqdm import trange
from A_InputDataProcess.AA_read_data import unpickle_gp_dataset, unpickle_ion_list
from C_Evaluation.att_model_proxy import AttMolProxy


def sigmoid(x_):
    s_ = 1 / (1 + np.exp(-x_))
    return s_


def gp_dataset_cc():
    gp_dataset = unpickle_gp_dataset()

    a_model = AttMolProxy('Anion')
    c_model = AttMolProxy('Cation')

    s_diff_list = []
    z_list = []
    a_list = []
    c_list = []
    for data in gp_dataset:
        z_list.append(
            np.concatenate((a_model.encode(one_hot=data.il.anion.onehot.reshape(
                1, data.il.anion.onehot.shape[0], data.il.anion.onehot.shape[1])),
                            c_model.encode(one_hot=data.il.cation.onehot.reshape(
                                1, data.il.cation.onehot.shape[0], data.il.cation.onehot.shape[1]))), axis=1)
        )
        a_list.append(Chem.MolFromSmiles(data.il.anion.smiles))
        c_list.append(Chem.MolFromSmiles(data.il.cation.smiles))
        s_diff_list.append(data.s_diff)

    z_list = np.array(z_list)
    s_diff_list = np.array(s_diff_list)

    z_euler_distance = np.zeros((len(z_list), len(z_list)))
    s_diff_distance = np.zeros((len(z_list), len(z_list)))
    mol_distance = np.zeros((len(z_list), len(z_list)))

    z_euler_distance_flat = []
    s_diff_distance_flat = []
    mol_distance_flat = []
    for i in trange(z_euler_distance.shape[0]):
        for j in range(i + 1, z_euler_distance.shape[0]):
            z_euler_distance[i][j] = np.sum(np.square(z_list[i] - z_list[j]))
            s_diff_distance[i][j] = s_diff_list[i] - s_diff_list[j]
            mol_distance[i][j] = 1 - (DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(a_list[i]),
                                                                        Chem.RDKFingerprint(a_list[j]))
                                      + DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(c_list[i]),
                                                                          Chem.RDKFingerprint(c_list[j]))) / 2

            z_euler_distance_flat.append(z_euler_distance[i][j])
            s_diff_distance_flat.append(s_diff_distance[i][j])
            mol_distance_flat.append(mol_distance[i][j])

    z_euler_distance_flat = np.array(z_euler_distance_flat)
    s_diff_distance_flat = np.abs(np.array(s_diff_distance_flat))
    mol_distance_flat = np.array(mol_distance_flat)

    gp_z_s_cc_ = np.corrcoef(z_euler_distance_flat, s_diff_distance_flat)
    gp_z_mol_cc_ = np.corrcoef(z_euler_distance_flat, mol_distance_flat)
    gp_s_mol_cc_ = np.corrcoef(s_diff_distance_flat, mol_distance_flat)
    gp_revised_z_s_cc_ = np.corrcoef(sigmoid(z_euler_distance_flat / np.mean(z_euler_distance_flat) * 1),
                                     s_diff_distance_flat)
    gp_revised_z_mol_cc_ = np.corrcoef(sigmoid(z_euler_distance_flat / np.mean(z_euler_distance_flat) * 1),
                                       sigmoid(mol_distance_flat / np.mean(mol_distance_flat) * 1))
    return gp_z_s_cc_, gp_z_mol_cc_, gp_s_mol_cc_, gp_revised_z_s_cc_, gp_revised_z_mol_cc_


def all_cc():
    anion_list = unpickle_ion_list('Anion')
    cation_list = unpickle_ion_list('Cation')
    a_model = AttMolProxy('Anion')
    c_model = AttMolProxy('Cation')

    anion_z_list = []
    cation_z_list = []
    anion_mol_list = []
    cation_mol_list = []

    for anion in anion_list:
        anion_z_list.append(a_model.encode(one_hot=anion.onehot.reshape(
            1, anion.onehot.shape[0], anion.onehot.shape[1])))
        anion_mol_list.append(Chem.MolFromSmiles(anion.smiles))
    for cation in cation_list:
        cation_z_list.append(c_model.encode(one_hot=cation.onehot.reshape(
            1, cation.onehot.shape[0], cation.onehot.shape[1])))
        cation_mol_list.append(Chem.MolFromSmiles(cation.smiles))

    z_euler_distance_flat = []
    mol_distance_flat = []
    for i in trange(len(anion_z_list)):
        for j in range(i + 1, len(anion_z_list)):
            z_euler_distance_flat.append(np.sum(np.square(anion_z_list[i] - anion_z_list[j])))
            mol_distance_flat.append(1 - DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(anion_mol_list[i]),
                                                                           Chem.RDKFingerprint(anion_mol_list[j])))

    for i in trange(len(cation_z_list)):
        for j in range(i + 1, len(cation_z_list)):
            z_euler_distance_flat.append(np.sum(np.square(cation_z_list[i] - cation_z_list[j])))
            mol_distance_flat.append(1 - DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(cation_mol_list[i]),
                                                                           Chem.RDKFingerprint(cation_mol_list[j])))

    z_euler_distance_flat = np.array(z_euler_distance_flat)
    mol_distance_flat = np.array(mol_distance_flat)
    cc = np.corrcoef(z_euler_distance_flat, mol_distance_flat)
    revised_cc = np.corrcoef(sigmoid(z_euler_distance_flat / np.mean(z_euler_distance_flat) * 1),
                             sigmoid(mol_distance_flat / np.mean(mol_distance_flat) * 1))
    return cc, revised_cc


if __name__ == '__main__':
    gp_z_s_cc, gp_z_mol_cc, gp_s_mol_cc, gp_revised_z_s_cc, gp_revised_z_mol_cc = gp_dataset_cc()
    all_cc, all_revised_cc = all_cc()
