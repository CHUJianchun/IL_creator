import os
import sys
import numpy as np
import math
import random

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from Util_mol.params import rule_ranges, terminal_indexes, DECISION_DIM
from Util_mol.molecule_tree import Node, tree_to_smiles, annotated_tree_to_mol_tree, annotated_tree_to_one_hot
from cmd_args import cmd_args
from Model.decoder import batch_make_att_masks, PerpCalculator, StateDecoder
from Model.encoder import CNNEncoder
import Util_cfg.cfg_parser as parser


def get_encoder(ion):
    if ion == 'Anion':
        latent_dim = cmd_args.anion_latent_dim
    else:
        latent_dim = cmd_args.cation_latent_dim
    if cmd_args.encoder_type == 'cnn':
        return CNNEncoder(max_len=cmd_args.max_decode_steps, latent_dim=latent_dim)
    else:
        raise ValueError('unknown encoder type %s' % cmd_args.encoder_type)


class MolAutoEncoder(nn.Module):
    def __init__(self, ion):
        super(MolAutoEncoder, self).__init__()
        print('using auto encoder')
        if ion == 'Anion':
            latent_dim = cmd_args.anion_latent_dim
        else:
            latent_dim = cmd_args.cation_latent_dim
        self.latent_dim = latent_dim
        self.encoder = get_encoder(ion)
        self.state_decoder = StateDecoder(max_len=cmd_args.max_decode_steps, latent_dim=latent_dim)
        self.perp_calc = PerpCalculator()

    def forward(self, x_inputs, true_binary, rule_masks):
        z, _ = self.encoder(x_inputs)

        raw_logits = self.state_decoder(z)
        perplexity = self.perp_calc(true_binary, rule_masks, raw_logits)

        return perplexity


class MolVAE(nn.Module):
    def __init__(self, ion):
        super(MolVAE, self).__init__()
        # print('using vae')
        if ion == 'Anion':
            latent_dim = cmd_args.anion_latent_dim
        else:
            latent_dim = cmd_args.cation_latent_dim

        self.latent_dim = latent_dim
        self.encoder = get_encoder(ion)
        self.state_decoder = StateDecoder(max_len=cmd_args.max_decode_steps, latent_dim=latent_dim)
        self.perp_calc = PerpCalculator()

    def reparameterize(self, mu, logvar):
        if self.training:
            eps = mu.data.new(mu.size()).normal_(0, cmd_args.eps_std)
            if cmd_args.mode == 'gpu':
                eps = eps.cuda()
            eps = Variable(eps)

            return mu + eps * torch.exp(logvar * 0.5)
        else:
            return mu

    def forward(self, x_inputs, true_binary, rule_masks):
        z_mean, z_log_var = self.encoder(x_inputs)

        z = self.reparameterize(z_mean, z_log_var)

        raw_logits = self.state_decoder(z)
        perplexity = self.perp_calc(true_binary, rule_masks, raw_logits)

        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean ** 2 - torch.exp(z_log_var), -1)

        return perplexity, cmd_args.kl_coeff * torch.mean(kl_loss)
