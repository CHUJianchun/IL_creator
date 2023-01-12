from A_InputDataProcess.AA_read_data import *

from B_Train.BA_train_vae import *

from C_Evaluation.CA_dump_encoding import *
from C_Evaluation.CU_reconstruct import *
from C_Evaluation.CU_similarity import *
from C_Evaluation import CU_valid_prior
from C_Evaluation import CU_sample_prior
from D_optimization.DA_bayesian_optimization import bayesian_optimization
from D_optimization.DBA_transfer_learning_data_preperation import create_tl_dataset, create_sdiff_dataset
from D_optimization.DBB_train_transfer_learning import *
from D_optimization.DBC1_bayesian_optimization import *

if __name__ == '__main__':
    generate processes
    ionic_liquid_list = load_ionic_liquid_list()
    data_point_list, data_frame_list = load_data_classified()
    gp_dataset_list = regress_il_property()

    train('Anion', 0)
    train('Cation', 0)

    anion_encode = dump_encoding('Anion')
    cation_encode = dump_encoding('Cation')
    x, y = dump_gp_dataset()

    bayesian_optimization(times=5)

    create_tl_dataset()
    create_sdiff_dataset()

    train_onehot_mlp()
    train_z_mlp()

    train_onehot_sdiff(mode='transfer learning')
    train_z_sdiff(mode='transfer learning')

    train_onehot_sdiff(mode='normal')
    train_z_sdiff(mode='normal')

    dump_prediction('OneHotSdiffMLP.pkl')
    dump_prediction('TLOneHotSdiffMLP.pkl')
    dump_prediction('ZSdiffMLP.pkl')
    dump_prediction('TLZSdiffMLP.pkl')

    analysis processes
    save_decode_result_and_accuracy('Anion')
    save_decode_result_and_accuracy('Cation')

    CU_valid_prior.cal_valid_prior('Anion')
    CU_valid_prior.cal_valid_prior('Cation')

    CU_sample_prior.cal_valid_prior('Anion')
    CU_sample_prior.cal_valid_prior('Cation')
