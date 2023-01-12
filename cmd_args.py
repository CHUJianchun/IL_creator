import argparse

cmd_opt = argparse.ArgumentParser(description='Argparser for molecule vae')
cmd_opt.add_argument('-mode', default='gpu', help='cpu/gpu')
cmd_opt.add_argument('-save_dir', default='SavedModel', help='result output root')
cmd_opt.add_argument('-anion_saved_model', default='SavedModel/Anion-epoch-best.model',
                     help='start from existing model')
cmd_opt.add_argument('-cation_saved_model', default='SavedModel/Cation-epoch-best.model',
                     help='start from existing model')
cmd_opt.add_argument('-encoder_type', default='cnn', help='choose encoder from [tree_lstm | s2v]')
cmd_opt.add_argument('-ae_type', default='vae', help='choose ae arch from [autoenc | vae]')
cmd_opt.add_argument('-rnn_type', default='gru', help='choose rnn cell from [gru | sru]')
cmd_opt.add_argument('-loss_type', default='perplexity', help='choose loss from [perplexity | binary]')
cmd_opt.add_argument('-max_decode_steps', type=int, default=278, help='maximum steps for making decoding decisions')
cmd_opt.add_argument('-batch_size', type=int, default=6, help='minibatch size')
cmd_opt.add_argument('-seed', type=int, default=114, help='random seed')
cmd_opt.add_argument('-skip_deter', type=int, default=0, help='skip deterministic position')
cmd_opt.add_argument('-bondcompact', type=int, default=0, help='compact ringbond representation or not')
cmd_opt.add_argument('-anion_latent_dim', type=int, default=16, help='anion latent dimension')
cmd_opt.add_argument('-cation_latent_dim', type=int, default=48, help='cation latent dimension')
cmd_opt.add_argument('-data_gen_threads', type=int, help='number of threads for data generation')
cmd_opt.add_argument('-num_epochs', type=int, default=500, help='number of epochs')
cmd_opt.add_argument('-learning_rate', type=float, default=0.001, help='init learning_rate')
cmd_opt.add_argument('-prob_fix', type=float, default=0, help='numerical problem')
cmd_opt.add_argument('-kl_coeff', type=float, default=0.02, help='coefficient for kl divergence used in vae')
cmd_opt.add_argument('-eps_std', type=float, default=0.01,
                     help='the standard deviation used in reparameterization tric')

cmd_args, _ = cmd_opt.parse_known_args()
