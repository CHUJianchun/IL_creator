# CFG: 上下文无关文法

import random
import nltk
from nltk.grammar import Nonterminal, Production
import itertools
from nltk.tree import Tree
from nltk.parse.chart import LeafEdge


# 文法类，用于读取文法
class Grammar(object):
    """
     self.cfg_parser: nltk.parser
     first_head : 'smiles'
     head_to_rules : (dict: 21) 'smiles' : list[tuple(chain,)]
     rule_ranges : (dict: 21) 'smiles' : tuple(0, 1)
     total_num_rules : 70
     valid_tokens : (set:31) 终结符
     generate: return str <随机>生成的SMILES

    """
    def __init__(self):
        self.head_to_rules = {}
        self.valid_tokens = set()
        self.rule_ranges = {}
        self.total_num_rules = 0
        self.first_head = None

        cfg_string = ''.join(list(open('Grammar/molecule.grammar').readlines()))
        cfg_grammar = nltk.CFG.fromstring(cfg_string)
        self.cfg_parser = nltk.ChartParser(cfg_grammar)
        for line in cfg_string.split('\n'):
            if len(line.strip()) > 0:
                head, rules = line.split('->')
                head = Nonterminal(head.strip())  # remove space
                rules = [_.strip() for _ in rules.split('|')]  # split and remove space
                rules = [tuple([Nonterminal(_) if not _.startswith("'") else _[1:-1] for _ in rule.split()]) for rule in
                         rules]
                self.head_to_rules[head] = rules

                for rule in rules:
                    for t_ in rule:
                        if isinstance(t_, str):
                            self.valid_tokens.add(t_)

                if self.first_head is None:
                    self.first_head = head

                self.rule_ranges[head] = (self.total_num_rules, self.total_num_rules + len(rules))
                self.total_num_rules += len(rules)

    def generate(self):
        """
        :return: str <随机>生成的SMILES
        """
        frontier = [self.first_head]
        while True:
            is_ended = not any(isinstance(item, Nonterminal) for item in frontier)
            if is_ended:
                break
            for i in range(len(frontier)):
                item = frontier[i]
                if isinstance(item, Nonterminal):
                    replacement_id = random.randint(0, len(self.head_to_rules[item]) - 1)
                    replacement = self.head_to_rules[item][replacement_id]
                    frontier = frontier[:i] + list(replacement) + frontier[i + 1:]
                    break
        return ''.join(frontier)

    def tokenize(self, sent):
        """
        :param sent: str smiles
        :return: list [token]
        """

        result = []
        n = len(sent)
        i = 0
        while i < n:
            j = i
            while j + 1 <= n and sent[i:j + 1] in self.valid_tokens:
                j += 1
            if i == j:
                return None
            result.append(sent[i: j])
            i = j
        return result


class AnnotatedTree(object):
    """
    Annotated Tree.

    It uses Nonterminal / Production class from nltk,
    see http://www.nltk.org/_modules/nltk/grammar.html for code.

    Attributes:
        :param symbol: a str object (for terminal) or a Nonterminal object (for non-terminal).
        :param children: a (maybe-empty) list of children.
        :param rule: a Production object.
        :param rule_selection_id: the 0-based index of which part of rule being selected. -1 for terminal.

    Method:
        is_leaf(): :return: True iff len(children) == 0
    """

    def __init__(self, symbol=None, children=None, rule=None, rule_selection_id=-1):
        symbol = symbol or ''
        children = children or []
        rule = rule or None
        rule_selection_id = rule_selection_id or 0

        assert (len(children) > 0 and rule is not None) or (len(children) == 0 and rule is None)
        self.symbol = symbol
        self.children = children
        self.rule = rule
        self.rule_selection_id = rule_selection_id

    def is_leaf(self):
        return len(self.children) == 0

    def __str__(self):
        return '[Symbol = %s / Rule = %s / Rule Selection ID = %d / Children = %s]' % (
            self.symbol,
            self.rule,
            self.rule_selection_id,
            self.children
        )

    def __repr__(self):
        return self.__str__()


def parse(sent, grammar_):
    """
    Returns a list of trees
    (for it's possible to have multiple parse tree)

    Returns None if the parsing fails.
    """
    # `sent` should be string
    assert isinstance(sent, str)

    sent = grammar_.tokenize(sent)
    if sent is None:
        return None

    try:
        trees = list(grammar_.cfg_parser.parse(sent))
    except ValueError:
        return None

    # print(trees)

    def _child_names(tree):
        names = []
        for child in tree:
            if isinstance(child, nltk.tree.Tree):
                names.append(Nonterminal(child._label))
            else:
                names.append(child)
        return names

    def _find_rule_selection_id(production):
        lhs, rhs = production.lhs(), production.rhs()
        assert lhs in grammar_.head_to_rules
        rules = grammar_.head_to_rules[lhs]
        for index, rule in enumerate(rules):
            if rhs == rule:
                return index
        assert False
        return 0

    def convert(tree):
        # convert from ntlk.tree.Tree to our AnnotatedTree

        if isinstance(tree, nltk.tree.Tree):
            symbol = Nonterminal(tree.label())
            children = list(convert(_) for _ in tree)
            rule = Production(Nonterminal(tree.label()), _child_names(tree))
            rule_selection_id = _find_rule_selection_id(rule)
            return AnnotatedTree(
                symbol=symbol,
                children=children,
                rule=rule,
                rule_selection_id=rule_selection_id
            )
        else:
            return AnnotatedTree(symbol=tree)

    # trees = [convert(tree) for tree in trees]
    trees = [convert(trees[0])]
    return trees


if __name__ == '__main__':
    grammar = Grammar()
    ts = parse('CCCCC', grammar)
    t = ts[0]

    print('(ugly) tree:')
    print(t)
    print()

    print('for root:')
    print('symbol is %s, is it non-terminal = %s, it\'s value is %s (of type %s)' % (
        t.symbol,
        isinstance(t, Nonterminal),
        t.symbol.symbol(),
        type(t.symbol.symbol())
    ))
    print('rule is %s, its left side is %s (of type %s), its right side is %s, a tuple '
          'which each element can be either str (for terminal) or Nonterminal (for nonterminal)' % (
              t.rule,
              t.rule.lhs(),
              type(t.rule.lhs()),
              t.rule.rhs(),
          ))