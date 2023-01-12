import numpy as np
import sys
import os
import Util_cfg.cfg_parser as parser
from cmd_args import cmd_args
from collections import defaultdict
from Util_mol.params import prod, rule_ranges, TOTAL_NUM_RULES, MAX_NESTED_BONDS, DECISION_DIM
from Util_mol.molecule_tree import *
from Model.decoder import batch_make_att_masks


if __name__ == '__main__':
    smiles = 'C(#N)C=1[N-]C=CC1'
    grammar = parser.Grammar()
    n = annotated_tree_to_mol_tree(parser.parse(smiles, grammar)[0])
    onehot = batch_make_att_masks([n])
