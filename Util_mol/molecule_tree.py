import numpy as np
import sys
import os
import Util_cfg.cfg_parser as parser
from cmd_args import cmd_args
from collections import defaultdict
from Util_mol.params import prod, rule_ranges, TOTAL_NUM_RULES, MAX_NESTED_BONDS, DECISION_DIM


class Node(object):

    def __init__(self, s, father=None):
        self.symbol = s
        self.children = []
        self.rule_used = None
        self.left_remain = None
        self.right_remain = None
        self.single_atom = None
        self.task = False
        self.pre_node = None
        self.atom_pos = None
        self.is_aromatic = None
        self.father = father
        if self.father is None:
            assert self.symbol == 'smiles'

    def init_attribute(self):
        self.left_remain = None
        self.right_remain = None
        self.single_atom = None
        self.task = False
        self.pre_node = None
        self.atom_pos = None
        self.is_aromatic = None

    def is_created(self):
        if self.rule_used is None:
            return False
        return len(prod[self.symbol][self.rule_used]) == len(self.children)

    def add_child(self, child, postorder=None):
        if self.is_created():
            return
        if postorder is None:
            self.children.append(child)
        else:
            self.children.insert(postorder, child)

    def get_pre(self):  # 一般返回的全是 -1
        if self.pre_node is None:
            if self.father is None:
                self.pre_node = -1
            else:
                self.pre_node = self.father.get_pre()
        assert self.pre_node is not None
        return self.pre_node


def depth_first_search(node, result):
    if len(node.children):
        for c in node.children:
            depth_first_search(c, result)
    else:
        assert node.symbol[0] == node.symbol[-1] == '\''  # 是字符 ‘ ，判断是不是字符串
        result.append(node.symbol[1:-1])


def tree_to_smiles(root):  # root 是个Node，输出是smiles
    result = []
    depth_first_search(root, result)
    st = ''.join(result)
    return st


def _annotated_tree_to_mol_tree(annotated_root, bond_set, father):
    n = Node(str(annotated_root.symbol), father=father)
    n.rule_used = annotated_root.rule_selection_id
    for c in annotated_root.children:
        new_c = _annotated_tree_to_mol_tree(c, bond_set, n)
        n.children.append(new_c)
    if n.symbol == 'ringbond':
        assert len(n.children)
        d = n.children[-1]
        assert d.symbol == 'DIGIT'
        st = d.children[0].symbol
        assert len(st) == 3
        idx = int(st[1: -1]) - 1
        if idx in bond_set:
            n.bond_idx = idx
            bond_set.remove(idx)
        else:
            bond_set.add(idx)
            n.bond_idx = MAX_NESTED_BONDS
    if isinstance(annotated_root.symbol, parser.Nonterminal):  # it is a non-terminal
        assert len(n.children)
        assert n.is_created()
    else:
        assert isinstance(annotated_root.symbol, str)
        assert len(n.symbol) < 3 or (n.symbol[0] != '\'' and n.symbol[-1] != '\'')
        n.symbol = '\'' + n.symbol + '\''
    return n


def annotated_tree_to_mol_tree(annotated_root):
    bond_set = set()
    ans = _annotated_tree_to_mol_tree(annotated_root, bond_set, father=None)
    assert len(bond_set) == 0
    return ans


def depth_first_search_indices(node, result):
    if len(node.children):
        assert node.rule_selection_id >= 0
        g_range = rule_ranges[str(node.symbol)]
        idx = g_range[0] + node.rule_selection_id
        assert 0 <= idx < TOTAL_NUM_RULES
        result.append(idx)
        for c in node.children:
            depth_first_search_indices(c, result)


def annotated_tree_to_rule_indices(annotated_root):
    result = []
    depth_first_search_indices(annotated_root, result)
    return np.array(result)


def annotated_tree_to_one_hot(annotated_root, max_len):
    cur_indices = annotated_tree_to_rule_indices(annotated_root)
    assert len(cur_indices) <= max_len

    x_cpu = np.zeros((DECISION_DIM, max_len), dtype=np.float32)
    x_cpu[cur_indices, np.arange(len(cur_indices))] = 1.0
    x_cpu[-1, np.arange(len(cur_indices), max_len)] = 1.0  # padding
    return x_cpu


if __name__ == '__main__':
    smiles = 'C1CC1C(=O)[O-]'
    grammar = parser.Grammar()

    ts = parser.parse(smiles, grammar)
    # assert isinstance(ts, list) and len(ts) == 1

    print(annotated_tree_to_rule_indices(ts[0]))
