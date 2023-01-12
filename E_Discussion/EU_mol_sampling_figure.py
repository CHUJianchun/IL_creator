import os
import shutil
from tqdm import tqdm, trange
import numpy as np
import cairosvg

import torch
from torch.autograd import Variable

from rdkit import rdBase, Chem, DataStructs
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.AtomPairs import Pairs, Torsions
from rdkit.Chem.Draw import SimilarityMaps

from cmd_args import cmd_args
from C_Evaluation.att_model_proxy import AttMolProxy, batch_decode
from D_optimization.DBA_transfer_learning_data_preperation import *


def sampling_figure(ion, distance=None, samples=None, given_start=True, start_index=10):
    if distance is None:
        distance = [1.0, 1.6, 2.2, 2.8]
    if samples is None:
        samples = [8, 16, 24, 32]
    sdiff_dataset = load_SDiffDataset('InputData/SDIFF_dataset.pkl')
    start = sdiff_dataset.z[start_index].numpy()

    if ion == 'Anion':
        model = AttMolProxy('Anion')
        latent_dim = cmd_args.anion_latent_dim
        start = start[:cmd_args.anion_latent_dim]
    else:
        model = AttMolProxy('Cation')
        latent_dim = cmd_args.cation_latent_dim
        start = start[cmd_args.anion_latent_dim:]
    if not os.path.exists('Graph_Mol/GridGraph/' + ion):
        os.mkdir('Graph_Mol/GridGraph/' + ion)
    else:
        shutil.rmtree('Graph_Mol/GridGraph/' + ion)
        os.mkdir('Graph_Mol/GridGraph/' + ion)
    start_mol = Chem.MolFromSmiles(model.decode(np.array(start.reshape(1, -1), dtype='f'), use_random=False)[0])
    z_sample = []
    smiles_sample = []
    for i_ in range(len(samples)):
        sample_matrix = np.random.randn(samples[i_], latent_dim)
        for j_ in range(sample_matrix.shape[0]):
            if given_start:
                sample_matrix[j_] = ((sample_matrix[j_] / np.sum(sample_matrix[j_] ** 2) * distance[i_]).reshape(
                    -1, 1) + start.reshape(-1, 1)).reshape(-1)
            else:
                sample_matrix[j_] = sample_matrix[j_] / np.sum(sample_matrix[j_] ** 2) * distance[i_]
        z_sample.append(sample_matrix)

    def MolToImgFile(z, filename_, s_filename_):
        try:
            smiles = model.decode(np.array(z.reshape(1, -1), dtype='f'))[0]
            mol = Chem.MolFromSmiles(smiles)
            Draw.MolToFile(mol, filename_)
            d = Draw.MolDraw2DSVG(400, 400)
            d.ClearDrawing()
            t, m = SimilarityMaps.GetSimilarityMapForFingerprint(start_mol, mol,
                                                                 lambda m, i: SimilarityMaps.GetMorganFingerprint(
                                                                     m, i, radius=2, fpType='bv'), draw2d=d)
            d.FinishDrawing()
            with open(s_filename_, 'w+') as f:
                f.write(d.GetDrawingText())
            if ion == 'Anion':
                assert '-' in smiles
            else:
                assert '+' in smiles
            return smiles
        except (AssertionError, ValueError):
            MolToImgFile(z, filename_, s_filename_)

    for i_ in trange(len(z_sample)):
        smiles_sample_item = []
        for j_ in range(z_sample[i_].shape[0]):
            z = z_sample[i_][j_]
            filename = 'Graph_Mol/GridGraph/' + ion + '/' + str(i_) + '_' + str(j_) + '.png'
            s_filename = 'Graph_Mol/GridGraph/' + ion + '/s' + str(i_) + '_' + str(j_) + '.svg'
            s_filename_png = 'Graph_Mol/GridGraph/' + ion + '/s' + str(i_) + '_' + str(j_) + '.png'
            smiles = MolToImgFile(z, filename, s_filename)
            cairosvg.svg2png(url=s_filename, write_to=s_filename_png)
            smiles_sample_item.append(smiles)
        smiles_sample.append(smiles_sample_item)
    return smiles_sample


if __name__ == '__main__':
    # smiles_sample = sampling_figure('Anion')
    smiles_sample = sampling_figure('Cation', distance=[5.0, 10.0, 15.0, 20.0])
