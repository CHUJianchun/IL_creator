import numpy as np
from tqdm import tqdm
from Util_mol import SAscorer
from A_InputDataProcess.AA_read_data import *


def ilsascore():
    il_list = unpickle_ionic_liquid_list()
    sascore_list = []
    for il in tqdm(il_list):
        smiles = il.anion.smiles + '.' + il.cation.smiles
        mol = Chem.MolFromSmiles(smiles)
        sascore_list.append(SAscorer.calculateScore(mol))
    return np.array(sascore_list)


if __name__ == '__main__':
    score = ilsascore()
