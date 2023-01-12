import os
import sys
import numpy as np
import math
import random
import argparse
from tqdm import tqdm
import pickle

from cmd_args import cmd_args
from A_InputDataProcess.AA_read_data import unpickle_ion_list, unpickle_gp_dataset
from C_Evaluation.att_model_proxy import AttMolProxy

cmd_opt = argparse.ArgumentParser(description='Argparser for encoding')
cmd_opt.add_argument('-mode', default='gpu', help='cpu/gpu')
cmd_opt.add_argument('-round', type=int, default=0, help='encoding round')
cmd_opt.add_argument('-noisy', type=int, default=0, help='use noisy encoding')
args, _ = cmd_opt.parse_known_args()


def dump_encoding(ion_type):
    ion_list = unpickle_ion_list(ion_type)
    ion_onehot_list = []
    for ion in ion_list:
        ion_onehot_list.append(ion.onehot)
    ion_latent_points_batches = []
    ion_model = AttMolProxy(ion_type)
    BATCH = cmd_args.batch_size

    for i_ in tqdm(range(0, len(ion_onehot_list), BATCH)):
        if i_ + BATCH < len(ion_onehot_list):
            size_ = BATCH
        else:
            size_ = len(ion_onehot_list) - i_
        ion_onehot_batch = []
        for j_ in range(i_, i_ + size_):
            ion_onehot_batch.append(ion_onehot_list[j_])
        ion_latent_points_batches.append(ion_model.encode(one_hot=ion_onehot_batch, use_random=args.noisy))

    ion_latent_points = np.vstack(ion_latent_points_batches)
    np.save('InputData/' + ion_type + '-features.npy', ion_latent_points)
    return ion_latent_points


def dump_gp_dataset():
    gp_dataset_list = unpickle_gp_dataset()

    il_onehot_list = []
    gp_s_diff_list = []

    for gp_dataset in gp_dataset_list:
        il_onehot_list.append([gp_dataset.il.anion.onehot, gp_dataset.il.cation.onehot])
        gp_s_diff_list.append(gp_dataset.s_diff)

    gp_il_latent_points_batches = []
    gp_s_diff_batches = []

    a_model = AttMolProxy('Anion')
    c_model = AttMolProxy('Cation')

    BATCH = cmd_args.batch_size
    gp_il_latent_points_batches_list = []
    for i_ in tqdm(range(0, len(gp_dataset_list), BATCH)):
        if i_ + BATCH < len(gp_dataset_list):
            size_ = BATCH
        else:
            size_ = len(gp_dataset_list) - i_
        gp_il_onehot_batch = []
        gp_s_diff_batch = []
        for j_ in range(i_, i_ + size_):
            gp_il_onehot_batch.append(il_onehot_list[j_])
            gp_s_diff_batch.append(gp_s_diff_list[j_])
        a_slice = []
        c_slice = []
        for slice_ in gp_il_onehot_batch:
            a_slice.append(slice_[0])
            c_slice.append(slice_[1])
        gp_il_latent_points_batches.append(np.concatenate(
            (a_model.encode(one_hot=a_slice, use_random=args.noisy),
             c_model.encode(one_hot=c_slice, use_random=args.noisy)), axis=1))
        gp_s_diff_batches.append(gp_s_diff_batch)

    for k_ in range(len(gp_il_latent_points_batches)):
        for l_ in range(len(gp_il_latent_points_batches[k_])):
            gp_il_latent_points_batches_list.append(gp_il_latent_points_batches[k_][l_].reshape(-1, 1))
    gp_x = np.squeeze(np.array(gp_il_latent_points_batches_list))
    gp_y = np.vstack(gp_s_diff_batches)
    np.save('InputData/gp-features.npy', gp_x)
    np.save('InputData/gp-solubility-difference.npy', gp_y)
    return gp_x, gp_y


if __name__ == '__main__':
    a = dump_encoding('Anion')
    c = dump_encoding('Cation')
    x, y = dump_gp_dataset()
