import numpy as np
from tqdm import tqdm
from rdkit import Chem

from C_Evaluation.att_model_proxy import batch_decode
from C_Evaluation.att_model_proxy import AttMolProxy
from C_Evaluation.att_model_proxy import cmd_args
nb_latent_point = 1000
chunk_size = 100
sample_times = 10


def cal_valid_prior(ion_type):
    seed = 1215
    np.random.seed(seed)
    if ion_type == 'Anion':
        latent_dim = cmd_args.anion_latent_dim
    else:
        latent_dim = cmd_args.cation_latent_dim
    model = AttMolProxy(ion_type)
    whole_valid, whole_total = 0, 0
    pbar = tqdm(list(range(0, nb_latent_point, chunk_size)), desc='decoding')
    for start in pbar:
        end = min(start + chunk_size, nb_latent_point)
        latent_point = np.random.normal(size=(end - start, latent_dim))
        latent_point = latent_point.astype(np.float32)

        raw_logits = model.pred_raw_logits(latent_point, 1500)
        decoded_array = batch_decode(raw_logits, True, decode_times=sample_times)

        for i in range(end - start):
            for j in range(sample_times):
                s = decoded_array[i][j]
                if not s.startswith('JUNK') and Chem.MolFromSmiles(s) is not None:
                    whole_valid += 1
                whole_total += 1
        pbar.set_description('valid : total = %d : %d = %.5f' % (
            whole_valid, whole_total, whole_valid * 1.0 / whole_total))
    valid_prior = 1.0 * whole_valid / whole_total
    valid_prior_save_file = 'OutputData/' + ion_type + '-valid_prior.txt'
    print('valid prior:', valid_prior)
    with open(valid_prior_save_file, 'w') as f_:
        print('valid prior:', valid_prior, file=f_)


if __name__ == '__main__':
    cal_valid_prior('Anion')
    cal_valid_prior('Cation')
