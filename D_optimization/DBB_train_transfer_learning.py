import os
from tqdm import tqdm
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

from cmd_args import cmd_args
import D_optimization.DBA_transfer_learning_data_preperation as dba


"""class TLDataset(Dataset):
    def __init__(self, onehot_list, z_list, mw_list, nve_list, tpsa_list, hka_list, lasa_list):
        self.onehot = Variable(torch.from_numpy(np.array(onehot_list)))
        self.z = Variable(torch.from_numpy(np.array(z_list)))
        self.mw = Variable(torch.from_numpy(np.array(mw_list)))
        self.nve = Variable(torch.from_numpy(np.array(nve_list)))
        self.tpsa = Variable(torch.from_numpy(np.array(tpsa_list)))
        self.hka = Variable(torch.from_numpy(np.array(hka_list)))
        self.lasa = Variable(torch.from_numpy(np.array(lasa_list)))

    def __getitem__(self, index):
        return self.onehot[index], self.z[index], Variable(
            torch.Tensor([self.mw[index], self.nve[index], self.tpsa[index], self.hka[index], self.lasa[index]]))

    def __len__(self):
        return len(self.onehot)


class SDiffDataset(Dataset):
    def __int__(self, onehot_list, z_list, sdiff_list):
        self.onehot = Variable(torch.from_numpy(np.array(onehot_list)))
        self.z = Variable(torch.from_numpy(np.array(z_list)))
        self.sdiff = Variable(torch.from_numpy(np.array(sdiff_list)))

    def __getitem__(self, index):
        return self.onehot[index], self.z[index], self.sdiff[index]

    def __len__(self):
        return len(self.z)"""


class OneHotTransferMLP(nn.Module):
    def __init__(self):
        super(OneHotTransferMLP, self).__init__()
        self.conv = nn.Sequential(  # 1, 556, 82
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=8, stride=2, padding=1),  # 2, 276, 39
            nn.Dropout(),
            nn.ReLU6(),
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=6, stride=3, padding=3),  # 2, 93, 14
            nn.Dropout(),
            nn.ReLU6(),
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=14, stride=1, padding=0),  # 4, 80, 1
            nn.Dropout(),
            nn.ReLU6()
        )
        self.full = nn.Sequential(
            nn.Linear(4 * 80, 80),
            nn.Dropout(),
            nn.ReLU6(),
            nn.Linear(80, 5)
        )
        for p in self.full.parameters():
            nn.init.normal_(p, mean=0, std=0.1)
        nn.init.constant_(self.full[0].bias, val=0.)
        nn.init.constant_(self.full[3].bias, val=0.)

    def forward(self, input_):
        return self.full(self.conv(input_).reshape(cmd_args.batch_size, 4 * 80))


class ZTransferMLP(nn.Module):
    def __init__(self):
        super(ZTransferMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cmd_args.anion_latent_dim + cmd_args.cation_latent_dim, cmd_args.anion_latent_dim),
            nn.Tanh(),
            nn.Linear(cmd_args.anion_latent_dim, 30),
            nn.Tanh(),
            nn.Linear(30, 5)
        )
        for p in self.mlp.parameters():
            nn.init.normal_(p, mean=0, std=0.1)
        nn.init.constant_(self.mlp[0].bias, val=0.)
        nn.init.constant_(self.mlp[2].bias, val=0.)
        nn.init.constant_(self.mlp[4].bias, val=0.)

    def forward(self, input_):
        return self.mlp(input_)


class OneHotSDiff(nn.Module):
    def __init__(self):
        super(OneHotSDiff, self).__init__()
        self.conv = nn.Sequential(  # 1, 556, 82
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=8, stride=2, padding=1),  # 2, 276, 39
            nn.Dropout(),
            nn.ReLU6(),
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=6, stride=3, padding=3),  # 2, 93, 14
            nn.Dropout(),
            nn.ReLU6(),
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=14, stride=1, padding=0),  # 4, 80, 1
            nn.Dropout(),
            nn.ReLU6()
        )
        self.full = nn.Sequential(
            nn.Linear(4 * 80, 80),
            nn.Dropout(),
            nn.ReLU6(),
            nn.Linear(80, 1)
        )
        for p in self.full.parameters():
            nn.init.normal_(p, mean=0, std=0.1)
        nn.init.constant_(self.full[0].bias, val=0.)
        nn.init.constant_(self.full[3].bias, val=0.)

    def forward(self, input_):
        return self.full(self.conv(input_).reshape(-1, 4 * 80))


class ZSDiff(nn.Module):
    def __init__(self):
        super(ZSDiff, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cmd_args.anion_latent_dim + cmd_args.cation_latent_dim, cmd_args.anion_latent_dim),
            nn.Tanh(),
            nn.Linear(cmd_args.anion_latent_dim, 30),
            nn.Tanh(),
            nn.Linear(30, 1)
        )
        for p in self.mlp.parameters():
            nn.init.normal_(p, mean=0, std=0.1)
        nn.init.constant_(self.mlp[0].bias, val=0.)
        nn.init.constant_(self.mlp[2].bias, val=0.)
        nn.init.constant_(self.mlp[4].bias, val=0.)

    def forward(self, input_):
        return self.mlp(input_)


def train_onehot_mlp(device=0):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    mlp = OneHotTransferMLP().cuda()
    dataset = dba.load_TLDataset('InputData/TL_dataset.pkl')
    data_loader = DataLoader(dataset, batch_size=cmd_args.batch_size, drop_last=True, shuffle=True)
    data_size = len(dataset)
    del dataset
    optimizer = optim.Adam(mlp.parameters(), lr=cmd_args.learning_rate, weight_decay=0.001)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True, min_lr=0.0001)
    best_loss = 10000000.
    mse = nn.MSELoss()
    for epoch in range(cmd_args.num_epochs):
        mlp.train()
        total_loss = 0
        for _, data in enumerate(tqdm(data_loader)):
            onehot = data[0]
            label = data[2]
            optimizer.zero_grad()
            onehot = onehot.reshape(onehot.shape[0], 1, onehot.shape[1], onehot.shape[2]).to(
                torch.float32).clone().detach().cuda()
            label = label.to(torch.float32).clone().detach().cuda()
            prediction = mlp(onehot)
            loss = mse(prediction, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.data.item()
        lr_scheduler.step(total_loss)
        total_loss /= data_size
        print('Total loss in %d is %.4f' % (epoch, total_loss))

        if total_loss < best_loss:
            torch.save(mlp.state_dict(), 'SavedModel/TLOneHotMLP.pkl')
            print('Save model')
            best_loss = total_loss


def train_onehot_sdiff(mode, device=0):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    mlp = OneHotSDiff().cuda()
    if mode == 'transfer learning':
        model_dict = mlp.state_dict()
        saved_model = torch.load('SavedModel/TLOneHotMLP.pkl')
        state_dict = {k: v for k, v in saved_model.items() if k in model_dict.keys() and model_dict[k].shape == v.shape}
        model_dict.update(state_dict)
        mlp.load_state_dict(model_dict)
        model_name = 'SavedModel/TLOneHotSdiffMLP.pkl'
    else:
        model_name = 'SavedModel/OneHotSdiffMLP.pkl'
    assert mode == 'transfer learning' or mode == 'normal'
    dataset = dba.load_SDiffDataset('InputData/SDIFF_dataset.pkl')
    data_size = len(dataset)
    data_loader = DataLoader(dataset, batch_size=cmd_args.batch_size, drop_last=True, shuffle=True)
    del dataset
    optimizer = optim.Adam(mlp.parameters(), lr=cmd_args.learning_rate, weight_decay=0.001)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True, min_lr=0.00001)
    best_loss = 10000000.
    mse = nn.MSELoss()
    for epoch in range(cmd_args.num_epochs):
        mlp.train()
        total_loss = 0
        for _, data in enumerate(tqdm(data_loader)):
            onehot = data[0]
            label = data[2]
            optimizer.zero_grad()
            onehot = onehot.reshape(onehot.shape[0], 1, onehot.shape[1], onehot.shape[2]).to(
                torch.float32).clone().detach().cuda()
            label = label.to(torch.float32).clone().detach().cuda()
            prediction = mlp(onehot)
            loss = mse(prediction, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.data.item()
        lr_scheduler.step(total_loss)
        total_loss /= data_size
        print('Total loss in %d is %.4f' % (epoch, total_loss))
        if total_loss < best_loss:
            torch.save(mlp.state_dict(), model_name)
            print('Save model')
            best_loss = total_loss
    print(best_loss)


def train_z_mlp(device=0):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    mlp = ZTransferMLP().cuda()
    dataset = dba.load_TLDataset('InputData/TL_dataset.pkl')
    data_size = len(dataset)
    data_loader = DataLoader(dataset, batch_size=cmd_args.batch_size, drop_last=True, shuffle=True)
    del dataset
    optimizer = optim.Adam(mlp.parameters(), lr=cmd_args.learning_rate, weight_decay=0.001)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True, min_lr=0.0001)
    best_loss = 10000000.
    mse = nn.MSELoss()
    for epoch in range(cmd_args.num_epochs):
        mlp.train()
        total_loss = 0
        for _, data in enumerate(tqdm(data_loader)):
            z = data[1]
            label = data[2]
            optimizer.zero_grad()
            z = z.reshape(cmd_args.batch_size, cmd_args.anion_latent_dim + cmd_args.cation_latent_dim
                          ).to(torch.float32).clone().detach().cuda()
            label = label.to(torch.float32).clone().detach().cuda()
            prediction = mlp(z)
            loss = mse(prediction, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.data.item()

        lr_scheduler.step(total_loss)
        total_loss /= data_size
        print('Total loss in %d is %.4f' % (epoch, total_loss))

        if total_loss < best_loss:
            best_model = mlp.state_dict()

            print('Save model')
            best_loss = total_loss
    torch.save(best_model, 'SavedModel/TLZMLP.pkl')


def train_z_sdiff(mode, device=0):
    if device >= 0:
        device = 'cuda:' + str(device)
    else:
        device = 'cpu'

    mlp = ZSDiff().to(device)
    if mode == 'transfer learning':
        model_dict = mlp.state_dict()
        saved_model = torch.load('SavedModel/TLZMLP.pkl')
        state_dict = {k: v for k, v in saved_model.items() if k in model_dict.keys() and model_dict[k].shape == v.shape}
        model_dict.update(state_dict)
        mlp.load_state_dict(model_dict)
        model_name = 'SavedModel/TLZSdiffMLP.pkl'
    else:
        model_name = 'SavedModel/ZSdiffMLP.pkl'
    assert mode == 'transfer learning' or mode == 'normal'
    dataset = dba.load_SDiffDataset('InputData/SDIFF_dataset.pkl')
    data_size = len(dataset)
    data_loader = DataLoader(dataset, batch_size=cmd_args.batch_size, drop_last=True, shuffle=True)
    del dataset
    optimizer = optim.Adam(mlp.parameters(), lr=cmd_args.learning_rate, weight_decay=0.001)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True, min_lr=0.00001)
    best_loss = 10000000.
    mse = nn.MSELoss()
    for epoch in range(cmd_args.num_epochs):
        mlp.train()
        total_loss = 0
        for _, data in enumerate(tqdm(data_loader)):
            z = data[1]
            label = data[2]
            optimizer.zero_grad()
            z = z.reshape(cmd_args.batch_size, cmd_args.anion_latent_dim + cmd_args.cation_latent_dim
                          ).to(torch.float32).clone().detach().to(device)
            label = label.to(torch.float32).clone().detach().to(device)
            prediction = mlp(z)
            loss = mse(prediction, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.data.item()

        lr_scheduler.step(total_loss)
        total_loss /= data_size
        print('Total loss in %d is %.4f' % (epoch, total_loss))

        if total_loss < best_loss:
            best_model = mlp.state_dict()
            print('Save model')
            best_loss = total_loss
    torch.save(best_model, model_name)
    print(best_loss)


def dump_prediction(model_name):
    mlp = None
    if 'ZSdiff' in model_name:
        mlp = ZSDiff().cuda()
        mlp.load_state_dict(torch.load('SavedModel/' + model_name))
    elif 'OneHotSdiff' in model_name:
        mlp = OneHotSDiff().cuda()
        mlp.load_state_dict(torch.load('SavedModel/' + model_name))

    assert mlp is not None
    mlp.eval()
    x, y = [], []
    dataset = dba.load_TLDataset('InputData/TL_dataset.pkl')
    if 'ZSdiff' in model_name:
        for onehot, z, _ in tqdm(dataset):
            prediction = mlp(z.reshape(1, cmd_args.anion_latent_dim + cmd_args.cation_latent_dim
                                       ).to(torch.float32).clone().detach().cuda())
            x.append(z.detach().cpu().numpy())
            y.append(prediction.detach().cpu().numpy())
    elif 'OneHotSdiff' in model_name:
        for onehot, z, _ in tqdm(dataset):
            prediction = mlp(onehot.reshape(1, 1, onehot.shape[0], onehot.shape[1]).to(torch.float32).clone().detach().cuda())
            x.append(onehot.detach().cpu().numpy())
            y.append(prediction.detach().cpu().numpy())

    np.save('InputData/' + model_name + '-x.npy', np.array(x))
    np.save('InputData/' + model_name + '-y.npy', np.array(y))
    return np.array(x), np.array(y)


if __name__ == '__main__':
    # train_onehot_mlp()
    # train_z_mlp()
    # 改一下epoch再继续
    # train_onehot_sdiff(mode='transfer learning')
    # train_z_sdiff(mode='transfer learning')

    # train_onehot_sdiff(mode='normal')
    # train_z_sdiff(mode='normal')
    # a, b = dump_prediction('OneHotSdiffMLP.pkl')
    # c, d = dump_prediction('TLOneHotSdiffMLP.pkl')
    e, f = dump_prediction('ZSdiffMLP.pkl')
    # g, h = dump_prediction('TLZSdiffMLP.pkl')
    pass
