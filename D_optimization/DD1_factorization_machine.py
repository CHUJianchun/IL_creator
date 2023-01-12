import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

from A_InputDataProcess.AA_read_data import *
from cmd_args import cmd_args


class FMDataset(Dataset):
    def __init__(self):
        super(FMDataset, self).__init__()
        sdiff_dataset = torch.load('InputData/SDIFF_dataset.pkl')
        xa = sdiff_dataset[1][:, :cmd_args.anion_latent_dim].reshape(sdiff_dataset[1].shape[0], -1)
        self.xa = torch.cat(
            (xa, torch.zeros((sdiff_dataset[1].shape[0], cmd_args.cation_latent_dim - cmd_args.anion_latent_dim))),
            dim=1)
        self.xc = sdiff_dataset[1][:, cmd_args.anion_latent_dim:].reshape(sdiff_dataset[1].shape[0], -1)
        self.y = sdiff_dataset[2]

    def __getitem__(self, idx):
        return self.xa[idx], self.xc[idx], self.y[idx]

    def __len__(self):
        return len(self.xa)


class DeepFM(nn.Module):
    def __init__(self, feature_sizes=None,
                 embedding_size=cmd_args.cation_latent_dim,
                 hidden_dims=None, num_classes=1, dropout=None, use_cuda=True):
        """
            Initialize a new network
            Inputs:
            - feature_size: A list of integer giving the size of features for each field.
            - embedding_size: An integer giving size of feature embedding.
            - hidden_dims: A list of integer giving the size of each hidden layer.
            - num_classes: An integer giving the number of classes to predict. For example,
                        someone may rate 1,2,3,4 or 5 stars to a film.
            - batch_size: An integer giving size of instances used in each iteration.
            - use_cuda: Bool, Using cuda or not
        """
        super().__init__()
        if feature_sizes is None:
            feature_sizes = cmd_args.cation_latent_dim
        if hidden_dims is None:
            hidden_dims = [8, 4]
        if dropout is None:
            dropout = [0.5, 0.5]

        self.field_size = cmd_args.cation_latent_dim
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dtype = torch.long

        """
            check if use cuda
        """
        if use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        """
            init fm part
        """
        self.fm_first_order_embeddings_1 = nn.ModuleList(
            [nn.Linear(1, 1) for i_ in range(self.feature_sizes)])
        self.fm_first_order_embeddings_2 = nn.ModuleList(
            [nn.Linear(1, 1) for i_ in range(self.feature_sizes)])
        self.fm_second_order_embeddings_1 = nn.ModuleList(
            [nn.Linear(1, self.embedding_size
                       ) for i_ in range(self.feature_sizes)])
        self.fm_second_order_embeddings_2 = nn.ModuleList(
            [nn.Linear(1, self.embedding_size
                       ) for i_ in range(self.feature_sizes)])

        """
            init deep part
        """
        self.all_dims = [self.field_size * self.embedding_size] + self.hidden_dims + [self.num_classes]
        for i in range(1, len(hidden_dims) + 1):
            setattr(self, 'linear_' + str(i), nn.Linear(self.all_dims[i - 1], self.all_dims[i]))
            init.normal_(getattr(self, 'linear_' + str(i)).weight, mean=0., std=0.01)
            init.constant_(getattr(self, 'linear_' + str(i)).bias, val=0.21)
            if i != len(hidden_dims):
                nn.init.constant_(getattr(self, 'linear_' + str(i)).bias, val=0.)
                setattr(self, 'batchNorm_' + str(i), nn.BatchNorm1d(self.all_dims[i]))
                setattr(self, 'Tanh_' + str(i), nn.Tanh())
                setattr(self, 'dropout_' + str(i), nn.Dropout(dropout[i - 1]))

    def forward(self, Xa, Xc):
        """
        Forward process of network.

        Inputs:
        - Xi: A tensor of input's index, shape of (N, field_size, 1)
        - Xv: A tensor of input's value, shape of (N, field_size, 1)
        """
        """
            fm part
        """
        fm_first_order_emb_arr_1 = [(emb(Xa[:, i].reshape(-1, 1)).t() * Xc[:, i]).t() for i, emb in
                                    enumerate(self.fm_first_order_embeddings_1)]
        fm_first_order_emb_arr_2 = [(Xa[:, i].t() * emb(Xc[:, i].reshape(-1, 1))).t() for i, emb in
                                    enumerate(self.fm_first_order_embeddings_2)]
        fm_first_order_emb_arr = []
        for i_ in range(len(fm_first_order_emb_arr_1)):
            fm_first_order_emb_arr.append(fm_first_order_emb_arr_1[i_] + fm_first_order_emb_arr_2[i_])
            fm_first_order_emb_arr[i_] = torch.unsqueeze(fm_first_order_emb_arr[i_], -1)
        fm_first_order = torch.cat(fm_first_order_emb_arr, 1)

        fm_second_order_emb_arr_1 = [(emb(Xa[:, i].reshape(-1, 1)).t() * Xc[:, i]).t() for i, emb in
                                     enumerate(self.fm_second_order_embeddings_1)]
        fm_second_order_emb_arr_2 = [(Xa[:, i] * emb(Xc[:, i].reshape(-1, 1)).t()).t() for i, emb in
                                     enumerate(self.fm_second_order_embeddings_2)]
        fm_second_order_emb_arr = []
        for i_ in range(len(fm_second_order_emb_arr_1)):
            fm_second_order_emb_arr.append(fm_second_order_emb_arr_1[i_] + fm_second_order_emb_arr_2[i_])
            fm_second_order_emb_arr[i_] = torch.unsqueeze(fm_second_order_emb_arr[i_], -1)

        fm_sum_second_order_emb = sum(fm_second_order_emb_arr)
        fm_sum_second_order_emb_square = fm_sum_second_order_emb * fm_sum_second_order_emb  # (x+y)^2
        fm_second_order_emb_square = [item * item for item in fm_second_order_emb_arr]
        fm_second_order_emb_square_sum = sum(fm_second_order_emb_square)  # x^2+y^2
        fm_second_order = (fm_sum_second_order_emb_square - fm_second_order_emb_square_sum) * 0.5

        """
            deep part
        """
        for i_ in range(len(fm_second_order_emb_arr)):
            fm_second_order_emb_arr[i_] = torch.unsqueeze(fm_second_order_emb_arr[i_], -1)
        deep_emb = torch.cat(fm_second_order_emb_arr, 1).squeeze()
        deep_out = deep_emb
        for i in range(1, len(self.hidden_dims) + 1):
            deep_out = getattr(self, 'linear_' + str(i))(deep_out).reshape(-1, self.all_dims[i])
            if i != len(self.hidden_dims):
                deep_out = getattr(self, 'batchNorm_' + str(i))(deep_out)
                deep_out = getattr(self, 'Tanh_' + str(i))(deep_out)
                deep_out = getattr(self, 'dropout_' + str(i))(deep_out)
        """
            sum
        """
        total_sum = (torch.mean(fm_first_order, dim=1) + torch.mean(fm_second_order, dim=1) + torch.mean(
            deep_out, dim=1).reshape(-1, 1))
        return total_sum

    def fit(self, loader_train, optimizer, epochs=cmd_args.num_epochs, mode='train'):
        """
        Training a model and valid accuracy.

        Inputs:
        - loader_train: I
        - loader_val: .
        - optimizer: Abstraction of optimizer used in training process, e.g., "torch.optim.Adam()""torch.optim.SGD()".
        - epochs: Integer, number of epochs.
        - verbose: Bool, if print.
        - print_every: Integer, print after every number of iterations.
        """
        """
            load input data
        """
        model = self.train().to(device=self.device)
        criterion = nn.L1Loss()
        best_loss = 10000000.
        lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=10, verbose=True, min_lr=0.00001)
        if mode == 'train':
            model_name = 'SavedModel/FM.pkl'
        else:
            model_name = 'SavedModel/FM_test.pkl'
        for _ in range(epochs):
            total_loss = 0.
            for t, (xa, xc, y) in enumerate(loader_train):
                xa = xa.to(device=self.device, dtype=torch.float)
                xc = xc.to(device=self.device, dtype=torch.float)
                y = y.to(device=self.device, dtype=torch.float)

                total = model(xa, xc)
                loss = criterion(total, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.data.item()
            total_loss /= 135
            lr_scheduler.step(total_loss)
            print('Total loss in %d is %.4f' % (_, total_loss))

            if total_loss < best_loss:
                best_model = model.state_dict()

                print('Save model')
                best_loss = total_loss
        torch.save(best_model, model_name)


def train_FM():
    dataset = FMDataset()
    data_loader = DataLoader(dataset, batch_size=cmd_args.batch_size, drop_last=True)
    model = DeepFM()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.001)
    model.cheat_fit(loader_train=data_loader, test_dataset=dataset, optimizer=optimizer)


if __name__ == '__main__':
    train_FM()
