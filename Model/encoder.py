from Util_mol.params import TOTAL_NUM_RULES, DECISION_DIM, rule_ranges, terminal_indexes
from Util_mol.molecule_tree import Node, tree_to_smiles, annotated_tree_to_mol_tree, annotated_tree_to_one_hot
from Util_mol.pytorch_initializer import weights_init
import Util_mol.molecule_tree
import Util_cfg.cfg_parser as parser

from cmd_args import cmd_args

import os
import sys
import csv
import importlib
import numpy as np
import math
import random
from collections import defaultdict
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CNNEncoder(nn.Module):
    def __init__(self, max_len, latent_dim):
        super(CNNEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.max_len = max_len

        self.conv1 = nn.Conv1d(DECISION_DIM, 9, 9)
        self.conv2 = nn.Conv1d(9, 9, 9)
        self.conv3 = nn.Conv1d(9, 10, 11)

        self.last_conv_size = max_len - 9 + 1 - 9 + 1 - 11 + 1
        self.w1 = nn.Linear(self.last_conv_size * 10, 435)
        self.mean_w = nn.Linear(435, latent_dim)
        self.log_var_w = nn.Linear(435, latent_dim)
        weights_init(self)

    def forward(self, x_cpu):
        if cmd_args.mode == 'cpu':
            batch_input = Variable(torch.from_numpy(x_cpu).float())
        else:
            batch_input = Variable(torch.from_numpy(x_cpu).float().cuda())

        h1 = self.conv1(batch_input)
        h1 = F.relu(h1)
        h2 = self.conv2(h1)
        h2 = F.relu(h2)
        h3 = self.conv3(h2)
        h3 = F.relu(h3)

        # h3 = torch.transpose(h3, 1, 2).contiguous()
        flatten = h3.view(x_cpu.shape[0], -1)
        h = self.w1(flatten)
        h = F.relu(h)

        z_mean = self.mean_w(h)
        z_log_var = self.log_var_w(h)

        return z_mean, z_log_var


if __name__ == '__main__':
    importlib.reload(Util_mol.molecule_tree)
    smiles_list = ['[N-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F', 'F[P-](F)(F)(F)(F)F', 'FC(C(=O)[O-])(F)F']
    cfg_trees = []
    cfg_onehots = []
    grammar = parser.Grammar()
    for smiles in smiles_list:
        ts = parser.parse(smiles, grammar)
        n = annotated_tree_to_mol_tree(ts[0])
        cfg_trees.append(n)
        cfg_onehots.append(annotated_tree_to_one_hot(ts[0], max_len=200))
    cfg_onehots = np.stack(cfg_onehots, axis=0)
    encoder = CNNEncoder(max_len=200, latent_dim=64)
    if cmd_args.mode == 'gpu':
        encoder.cuda()
    z = encoder(cfg_onehots)
    print(z[0].size())
