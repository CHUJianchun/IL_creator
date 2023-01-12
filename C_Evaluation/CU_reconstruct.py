import numpy as np
from tqdm import tqdm
import sys
from A_InputDataProcess.AA_read_data import *
from C_Evaluation.att_model_proxy import AttMolProxy
from cmd_args import cmd_args

nb_smiles = 200
chunk_size = 10
encode_times = 10
decode_times = 5


def reconstruct(model, smiles=None, onehot=None):
    decode_result = []
    if smiles is not None:
        chunk = smiles
        chunk_result = [[] for _ in range(len(chunk))]
        for _encode in range(encode_times):
            z1 = model.encode(smiles=chunk, use_random=True)
            encode_id, encode_total = _encode + 1, encode_times
            for _decode in tqdm(list(range(decode_times)),
                                'encode %d/%d decode' % (encode_id, encode_total)
                                ):
                _result = model.decode(z1, use_random=True)
                for index, s in enumerate(_result):
                    chunk_result[index].append(s)
    elif onehot is not None:
        chunk = onehot
        chunk_result = [[] for _ in range(len(chunk))]
        for _encode in range(encode_times):
            z1 = model.encode(one_hot=chunk, use_random=True)
            encode_id, encode_total = _encode + 1, encode_times
            for _decode in tqdm(list(range(decode_times)),
                                'encode %d/%d decode' % (encode_id, encode_total)
                                ):
                _result = model.decode(z1, use_random=True)
                for index, s in enumerate(_result):
                    chunk_result[index].append(s)
    decode_result.extend(chunk_result)
    # assert len(decode_result) == len(smiles)
    return decode_result


def save_decode_result(ion_type, decode_result, filename):
    smiles = unpickle_smiles_list(ion_type)
    with open(filename, 'w') as f_:
        for s, cand in zip(smiles, decode_result):
            print(','.join([s] + cand), file=f_)


def cal_accuracy(ion_type, decode_result):
    smiles = unpickle_smiles_list(ion_type)
    accuracy = [sum([1 for c in cand if c == s]) * 1.0 / len(cand) for s, cand in zip(smiles, decode_result)]
    junk = [sum([1 for c in cand if c.startswith('JUNK')]) * 1.0 / len(cand) for s, cand in zip(smiles, decode_result)]
    return (sum(accuracy) * 1.0 / len(accuracy)), (sum(junk) * 1.0 / len(accuracy))


def save_decode_result_and_accuracy(ion_type):
    onehot = unpickle_onehot_list(ion_type)
    decode_result_save_file = 'OutputData/' + ion_type + 'epoch-best-reconstruct_decode_result.csv'
    accuracy_save_file = 'OutputData/' + ion_type + 'epoch-best-reconstruct_accuracy.txt'
    model = AttMolProxy(ion_type)
    decode_result = reconstruct(model, onehot=onehot)
    accuracy, junk = cal_accuracy(ion_type, decode_result)
    print('accuracy:', accuracy, 'junk:', junk)

    save_result = True
    if save_result:
        with open(accuracy_save_file, 'w') as f_:
            print('accuracy:', accuracy, 'junk:', junk, file=f_)
        save_decode_result(ion_type, decode_result, decode_result_save_file)


if __name__ == '__main__':
    save_decode_result_and_accuracy('Anion')
    save_decode_result_and_accuracy('Cation')
