
import numpy as np
from rdkit import Chem
from collections import Counter

from C_Evaluation.att_model_proxy import batch_decode
from C_Evaluation.att_model_proxy import AttMolProxy
from cmd_args import cmd_args


nb_latent_point = 200
chunk_size = 100
sample_times = 100


def cal_valid_prior(ion_type):
    if ion_type == 'Anion':
        latent_dim = cmd_args.anion_latent_dim
    else:
        latent_dim = cmd_args.cation_latent_dim
    model = AttMolProxy(ion_type)
    seed = cmd_args.seed
    np.random.seed(seed)
    latent_point = np.random.normal(size=(nb_latent_point, latent_dim))
    latent_point = latent_point.astype(np.float32)

    raw_logits = model.pred_raw_logits(latent_point)
    decoded_array = batch_decode(raw_logits, True, decode_times=sample_times)

    decode_list = []
    for i in range(nb_latent_point):
        c = Counter()
        for j in range(sample_times):
            c[decoded_array[i][j]] += 1
        decoded = c.most_common(1)[0][0]
        if decoded.startswith('JUNK'):
            continue
        m = Chem.MolFromSmiles(decoded)
        if m is None:
            continue
        decode_list.append(decoded)
        if len(decode_list) == 100:
            break

    valid_prior_save_file = 'OutputData/' + ion_type + '-sampled_prior.txt'
    with open(valid_prior_save_file, 'w') as f_:
        for row in decode_list:
            f_.write('%s\n' % row)


if __name__ == '__main__':
    cal_valid_prior('Anion')
    cal_valid_prior('Cation')
