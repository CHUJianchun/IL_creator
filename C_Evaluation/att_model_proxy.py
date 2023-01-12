from tqdm import tqdm
import os
import sys
import numpy as np
import math
import random
import string
import torch
from torch.autograd import Variable

from cmd_args import cmd_args
from Util_mol.molecule_tree import annotated_tree_to_mol_tree, tree_to_smiles, Node
from Model.vae import MolVAE, MolAutoEncoder
from Model.attribute_tree_decoder import create_tree_decoder
from Model.decoder import batch_make_att_masks
from Model.tree_walker import OnehotBuilder, ConditionalDecoder
import Util_cfg.cfg_parser as parser


def parse_single(smiles, grammar):
    ts = parser.parse(smiles, grammar)
    assert isinstance(ts, list) and len(ts) == 1
    n = annotated_tree_to_mol_tree(ts[0])
    return n


def parse(chunk, grammar):
    return [parse_single(smiles, grammar) for smiles in chunk]


def batch_decode(raw_logits, use_random, decode_times):
    tree_decoder = create_tree_decoder()
    chunk_result = [[] for _ in range(raw_logits.shape[1])]

    for i in tqdm(range(raw_logits.shape[1])):
        pred_logits = raw_logits[:, i, :]
        walker = ConditionalDecoder(np.squeeze(pred_logits), use_random)

        for _decode in range(decode_times):
            new_t = Node('smiles')
            try:
                tree_decoder.decode(new_t, walker)
                sampled = tree_to_smiles(new_t)
            except Exception as ex:
                if not type(ex).__name__ == 'DecodingLimitExceeded':
                    pass
                    # print('Warning, decoder failed with', ex)
                # failed. output a random junk.

                sampled = 'JUNK' + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(256))

            chunk_result[i].append(sampled)

    return chunk_result


class AttMolProxy(object):
    def __init__(self, ion, *args, **kwargs):
        if cmd_args.ae_type == 'vae':
            self.ae = MolVAE(ion)
        elif cmd_args.ae_type == 'autoenc':
            self.ae = MolAutoEncoder(ion)
        else:
            raise Exception('unknown ae type %s' % cmd_args.ae_type)
        if cmd_args.mode == 'gpu':
            self.ae = self.ae.cuda()

        assert cmd_args.anion_saved_model is not None
        assert cmd_args.cation_saved_model is not None
        if ion == 'Anion':
            if cmd_args.mode == 'cpu':
                self.ae.load_state_dict(torch.load(cmd_args.anion_saved_model, map_location=lambda storage, loc: storage))
            else:
                self.ae.load_state_dict(torch.load(cmd_args.anion_saved_model))
        elif ion == 'Cation':
            if cmd_args.mode == 'cpu':
                self.ae.load_state_dict(torch.load(cmd_args.cation_saved_model, map_location=lambda storage, loc: storage))
            else:
                self.ae.load_state_dict(torch.load(cmd_args.cation_saved_model))

        self.onehot_walker = OnehotBuilder()
        self.tree_decoder = create_tree_decoder()
        self.grammar = parser.Grammar()

    def encode(self, smiles=None, one_hot=None, use_random=False):
        """
        Args:
            smiles: a list of `n` strings, each being a SMILES.
            onehot: pass
            use_random: pass
        Returns:
            A numpy array of dtype np.float32, of shape (n, latent_dim)
            Note: Each row should be the *mean* of the latent space distrubtion
                  rather than a sampled point from that distribution.
            (It can be anythin as long as it fits what self.decode expects)
        """
        if smiles is not None and type(smiles[0]) is str:
            cfg_tree_list = parse(smiles, self.grammar)
            onehot, _ = batch_make_att_masks(cfg_tree_list, self.tree_decoder, self.onehot_walker, dtype=np.float32)
        elif smiles is not None:
            cfg_tree_list = smiles
            onehot, _ = batch_make_att_masks(cfg_tree_list, self.tree_decoder, self.onehot_walker, dtype=np.float32)
        else:
            onehot = one_hot

        x_inputs = np.transpose(onehot, [0, 2, 1])
        if use_random:
            self.ae.train()
        else:
            self.ae.eval()
        z_mean, _ = self.ae.encoder(x_inputs)

        return z_mean.data.cpu().numpy()

    def pred_raw_logits(self, chunk, n_steps=None):
        '''
        Args:
            chunk: A numpy array of dtype np.float32, of shape (n, latent_dim)
        Return:
            numpy array of MAXLEN x batch_size x DECISION_DIM
        '''
        if cmd_args.mode == 'cpu':
            z = Variable(torch.from_numpy(chunk))
        else:
            z = Variable(torch.from_numpy(chunk).cuda())

        raw_logits = self.ae.state_decoder(z, n_steps)

        raw_logits = raw_logits.data.cpu().numpy()

        return raw_logits

    def decode(self, chunk, use_random=True):
        """
        :Args:
            chunk: A numpy array of dtype np.float32, of shape (n, latent_dim)
        :Return:
            a list of `n` strings, each being a SMILES.
        """
        raw_logits = self.pred_raw_logits(chunk)

        result_list = []
        for i in range(raw_logits.shape[1]):
            pred_logits = raw_logits[:, i, :]

            walker = ConditionalDecoder(np.squeeze(pred_logits), use_random)

            new_t = Node('smiles')
            try:
                self.tree_decoder.decode(new_t, walker)
                sampled = tree_to_smiles(new_t)
            except Exception as ex:
                if not type(ex).__name__ == 'DecodingLimitExceeded':
                    pass
                    # print('Warning, decoder failed with', ex)
                # failed. output a random junk.
                sampled = 'JUNK' + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(256))

            result_list.append(sampled)

        return result_list


if __name__ == '__main__':

    proxy = AttMolProxy('Anion')
    test_list = ['C(CCCCCCCCCCCCCC)(C)C', 'F[B-](F)(F)F', 'S(=O)(=O)(OCC)[O-]', 'N[C@@H](C(C)C)C(=O)[O-]']
    z_mean = proxy.encode(smiles=test_list)
    print(z_mean.shape)
    decoded_list = proxy.decode(z_mean, False)
    print('origin: ', test_list)
    print('decode: ', decoded_list)

    proxy = AttMolProxy('Cation')
    test_list = ['C[N+]1=CN(C=C1)CCCCCCCC']
    z_mean = proxy.encode(smiles=test_list)
    print(z_mean.shape)
    decoded_list = proxy.decode(z_mean, False)
    print('origin: ', test_list)
    print('decode: ', decoded_list)