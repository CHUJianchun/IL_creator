from tqdm import tqdm

from cmd_args import cmd_args
from A_InputDataProcess.AA_read_data import *
from C_Evaluation.att_model_proxy import AttMolProxy


def reconstruct_accuracy():
    a_model = AttMolProxy('Anion')
    c_model = AttMolProxy('Cation')
    a_list = unpickle_ion_list('Anion')
    c_list = unpickle_ion_list('Cation')
    a_sum = len(a_list)
    c_sum = len(c_list)
    a_right = 0
    c_right = 0
    for anion in tqdm(a_list):
        if a_model.decode(a_model.encode(one_hot=anion.onehot.reshape(1, anion.onehot.shape[0], anion.onehot.shape[1]
                                                                      )), use_random=False)[0] == anion.smiles:
            a_right += 1
    for cation in tqdm(c_list):
        if c_model.decode(c_model.encode(one_hot=cation.onehot.reshape(1, cation.onehot.shape[0], cation.onehot.shape[1]
                                                                       )), use_random=False)[0] == cation.smiles:
            c_right += 1
    return a_right / a_sum, c_right / c_sum


if __name__ == '__main__':
    a_accuracy, c_accuracy = reconstruct_accuracy()
