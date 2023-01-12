import numpy as np
from tqdm import tqdm
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles, AddHs
from rdkit.Chem.Descriptors import MolWt, NumValenceElectrons, NumRadicalElectrons
from rdkit.Chem.rdMolDescriptors import CalcLabuteASA
from rdkit.Chem.rdMolDescriptors import CalcTPSA, CalcHallKierAlpha
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from A_InputDataProcess.AA_read_data import unpickle_ion_list, unpickle_gp_dataset
from C_Evaluation.att_model_proxy import AttMolProxy
from cmd_args import cmd_args


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


class SDiffDataset(Dataset):
    def __init__(self, onehot_list, z_list, sdiff_list):
        self.onehot = Variable(torch.from_numpy(np.array(onehot_list)))
        self.z = Variable(torch.from_numpy(np.array(z_list)))
        self.sdiff = Variable(torch.from_numpy(np.array(sdiff_list)))

    def __getitem__(self, index):
        return self.onehot[index], self.z[index], self.sdiff[index]

    def __len__(self):
        return len(self.z)


class PropertyDataset(Dataset):
    def __init__(self, onehot_list, z_list, sdiff_list, mw_list, nve_list, tpsa_list, hka_list, lasa_list):
        self.onehot = Variable(torch.from_numpy(np.array(onehot_list)))
        self.z = Variable(torch.from_numpy(np.array(z_list)))
        self.sdiff = Variable(torch.from_numpy(np.array(sdiff_list)))
        self.mw = Variable(torch.from_numpy(np.array(mw_list)))
        self.nve = Variable(torch.from_numpy(np.array(nve_list)))
        self.tpsa = Variable(torch.from_numpy(np.array(tpsa_list)))
        self.hka = Variable(torch.from_numpy(np.array(hka_list)))
        self.lasa = Variable(torch.from_numpy(np.array(lasa_list)))

    def __getitem__(self, index):
        return self.onehot[index], self.z[index], self.sdiff[index], Variable(
            torch.Tensor([self.mw[index], self.nve[index], self.tpsa[index], self.hka[index], self.lasa[index]]))

    def __len__(self):
        return len(self.z)


def save_TLDataset(dataset_, path):
    torch.save([dataset_.onehot, dataset_.z, dataset_.mw, dataset_.nve, dataset_.tpsa, dataset_.hka, dataset_.lasa],
               path)


def load_TLDataset(path):
    params = torch.load(path)
    return TLDataset(params[0], params[1], params[2], params[3], params[4], params[5], params[6])


def save_SDiffDataset(dataset_, path):
    torch.save([dataset_.onehot, dataset_.z, dataset_.sdiff], path)


def load_SDiffDataset(path):
    params = torch.load(path)
    return SDiffDataset(params[0], params[1], params[2])


def save_PropertyDataset(dataset_, path):
    torch.save([
        dataset_.onehot, dataset_.z, dataset_.sdiff,
        dataset_.mw, dataset_.nve, dataset_.tpsa, dataset_.hka, dataset_.lasa], path)


def load_PropertyDataset(path):
    params = torch.load(path)
    return PropertyDataset(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7])


def create_tl_dataset():
    a_model = AttMolProxy('Anion')
    c_model = AttMolProxy('Cation')
    a_list = unpickle_ion_list('Anion')
    c_list = unpickle_ion_list('Cation')

    onehot_list, z_list, mw_list, nve_list, tpsa_list, hka_list, lasa_list = [], [], [], [], [], [], []

    for anion in tqdm(a_list):
        for cation in c_list:
            smiles = anion.smiles + '.' + cation.smiles
            onehot = np.concatenate((anion.onehot, cation.onehot))
            mol = MolFromSmiles(smiles)
            mol = AddHs(mol)
            Chem.GetSSSR(mol)
            AllChem.EmbedMultipleConfs(mol, 1)
            onehot_list.append(onehot)
            z_list.append(
                np.concatenate(
                    (a_model.encode(one_hot=anion.onehot.reshape(1, anion.onehot.shape[0], anion.onehot.shape[1])).T,
                     c_model.encode(one_hot=cation.onehot.reshape(1, cation.onehot.shape[0], cation.onehot.shape[1])).T
                     )))
            mw_list.append(MolWt(mol))
            nve_list.append(NumValenceElectrons(mol))
            tpsa_list.append(CalcTPSA(mol))
            hka_list.append(CalcHallKierAlpha(mol))
            lasa_list.append(CalcLabuteASA(mol))

    dataset_ = TLDataset(onehot_list, z_list, mw_list, nve_list, tpsa_list, hka_list, lasa_list)
    save_TLDataset(dataset_, 'InputData/TL_dataset.pkl')
    return dataset_


def create_sdiff_dataset():
    a_model = AttMolProxy('Anion')
    c_model = AttMolProxy('Cation')
    gp_dataset = unpickle_gp_dataset()
    onehot_list = []
    z_list = []
    sdiff_list = []
    for data in gp_dataset:
        onehot_list.append(np.concatenate((data.il.anion.onehot, data.il.cation.onehot)))
        z_list.append(
            np.concatenate(
                (a_model.encode(one_hot=data.il.anion.onehot.reshape(
                    1, data.il.anion.onehot.shape[0], data.il.anion.onehot.shape[1])).T,
                 c_model.encode(one_hot=data.il.cation.onehot.reshape(
                     1, data.il.cation.onehot.shape[0], data.il.cation.onehot.shape[1])).T
                 )))
        sdiff_list.append(data.s_diff)

    dataset_ = SDiffDataset(onehot_list=onehot_list, z_list=z_list, sdiff_list=sdiff_list)
    save_SDiffDataset(dataset_, 'InputData/SDIFF_dataset.pkl')
    return dataset_


def create_property_dataset():
    a_model = AttMolProxy('Anion')
    c_model = AttMolProxy('Cation')
    gp_dataset = unpickle_gp_dataset()
    onehot_list = []
    z_list = []
    sdiff_list = []
    mw_list, nve_list, tpsa_list, hka_list, lasa_list = [], [], [], [], []
    for data in gp_dataset:
        onehot_list.append(np.concatenate((data.il.anion.onehot, data.il.cation.onehot)))
        sdiff_list.append(data.s_diff)
        z_list.append(
            np.concatenate(
                (a_model.encode(one_hot=data.il.anion.onehot.reshape(
                    1, data.il.anion.onehot.shape[0], data.il.anion.onehot.shape[1])).T,
                 c_model.encode(one_hot=data.il.cation.onehot.reshape(
                     1, data.il.cation.onehot.shape[0], data.il.cation.onehot.shape[1])).T
                 )))
        smiles = data.il.anion.smiles + '.' + data.il.cation.smiles
        onehot = np.concatenate((data.il.anion.onehot, data.il.cation.onehot))

        mol = MolFromSmiles(smiles)
        mol = AddHs(mol)
        Chem.GetSSSR(mol)
        AllChem.EmbedMultipleConfs(mol, 1)
        onehot_list.append(onehot)
        z_list.append(
            np.concatenate(
                (a_model.encode(one_hot=data.il.anion.onehot.reshape(1, data.il.anion.onehot.shape[0],
                                                                     data.il.anion.onehot.shape[1])).T,
                 c_model.encode(one_hot=data.il.cation.onehot.reshape(1, data.il.cation.onehot.shape[0],
                                                                      data.il.cation.onehot.shape[1])).T
                 )))
        mw_list.append(MolWt(mol))
        nve_list.append(NumValenceElectrons(mol))
        tpsa_list.append(CalcTPSA(mol))
        hka_list.append(CalcHallKierAlpha(mol))
        lasa_list.append(CalcLabuteASA(mol))
    dataset_ = PropertyDataset(onehot_list, z_list, sdiff_list, mw_list, nve_list, tpsa_list, hka_list, lasa_list)
    save_PropertyDataset(dataset_, 'InputData/Property_dataset.pkl')
    return dataset_


if __name__ == '__main__':
    # tl = create_tl_dataset()
    # sd = create_sdiff_dataset()
    property_dataset = create_property_dataset()
    data = pd.DataFrame({'sdiff': property_dataset.sdiff.numpy().squeeze(),
                         'hka': property_dataset.hka,
                         'lasa': property_dataset.lasa,
                         'mw': property_dataset.mw,
                         'nve': property_dataset.nve,
                         'tpsa': property_dataset.tpsa})
