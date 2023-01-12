import os
import sys
import numpy as np


from Util_mol.params import atom_valence, bond_types, bond_valence, prod, TOTAL_NUM_RULES, rule_ranges, DECISION_DIM
from Util_mol.molecule_tree import Node, tree_to_smiles, annotated_tree_to_mol_tree
from cmd_args import cmd_args
from Util_cfg import cfg_parser
from Model import attribute_tree_decoder


class DecodingLimitExceeded(Exception):
    def __init__(self):
        pass

    def __str__(self):
        return 'DecodingLimitExceeded'


class TreeWalker(object):

    def __init__(self):
        pass

    def reset(self):
        raise NotImplementedError

    def sample_index_with_mask(self, node, idxes):
        raise NotImplementedError


class OnehotBuilder(TreeWalker):

    def __init__(self):
        super(OnehotBuilder, self).__init__()
        self.reset()

    def reset(self):
        self.num_steps = 0
        self.global_rule_used = []
        self.mask_list = []

    def sample_index_with_mask(self, node, idxes):
        assert node.rule_used is not None
        g_range = rule_ranges[node.symbol]
        global_idx = g_range[0] + node.rule_used
        self.global_rule_used.append(global_idx)
        self.mask_list.append(np.array(idxes))

        self.num_steps += 1

        result = None
        for i in range(len(idxes)):
            if idxes[i] == global_idx:
                result = i
        if result is None:
            print('Rule range, node, possible rule, global used rule, local used rule')
            print(rule_ranges[node.symbol], node.symbol, idxes, global_idx, node.rule_used)
        assert result is not None
        return result

    def sample_att(self, node, candidates):
        assert hasattr(node, 'bond_idx')
        assert node.bond_idx in candidates

        global_idx = TOTAL_NUM_RULES + node.bond_idx
        self.global_rule_used.append(global_idx)
        self.mask_list.append(np.array(candidates) + TOTAL_NUM_RULES)

        self.num_steps += 1

        return node.bond_idx


class ConditionalDecoder(TreeWalker):

    def __init__(self, raw_logits, use_random):
        super(ConditionalDecoder, self).__init__()
        self.raw_logits = raw_logits
        self.use_random = use_random
        assert len(raw_logits.shape) == 2 and raw_logits.shape[1] == DECISION_DIM

        self.reset()

    def reset(self):
        self.num_steps = 0

    def _get_idx(self, cur_logits):
        if self.use_random:
            cur_prob = np.exp(cur_logits)
            cur_prob /= np.sum(cur_prob)

            result = np.random.choice(len(cur_prob), 1, p=cur_prob)[0]
            result = int(result)  # enusre it's converted to int
        else:
            result = np.argmax(cur_logits)

        self.num_steps += 1
        return result

    def sample_index_with_mask(self, node, idxes):
        if self.num_steps >= self.raw_logits.shape[0]:
            raise DecodingLimitExceeded()
        cur_logits = self.raw_logits[self.num_steps][idxes]

        return self._get_idx(cur_logits)

    def sample_att(self, node, candidates):
        if self.num_steps >= self.raw_logits.shape[0]:
            raise DecodingLimitExceeded()
        cur_logits = self.raw_logits[self.num_steps][np.array(candidates) + TOTAL_NUM_RULES]

        idx = self._get_idx(cur_logits)

        return candidates[idx]