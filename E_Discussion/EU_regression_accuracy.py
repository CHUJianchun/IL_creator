import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

import os
from tqdm import tqdm
import numpy as np
import math
import matplotlib.pyplot as plt

from cmd_args import cmd_args
from A_InputDataProcess.AA_read_data import *
import D_optimization.DBA_transfer_learning_data_preperation as dba
from D_optimization.DBB_train_transfer_learning import *
from D_optimization.DD1_factorization_machine import *


def t_mlp(mode, device=0):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    mlp = ZSDiff().cuda()
    if mode == 'transfer learning':
        model_dict = mlp.state_dict()
        saved_model = torch.load('SavedModel/TLZMLP.pkl')
        state_dict = {k: v for k, v in saved_model.items() if k in model_dict.keys() and model_dict[k].shape == v.shape}
        model_dict.update(state_dict)
        mlp.load_state_dict(model_dict)
        model_name = 'SavedModel/TLZSdiffMLP_test.pkl'
    else:
        model_name = 'SavedModel/ZSdiffMLP_test.pkl'
    assert mode == 'transfer learning' or mode == 'normal'
    dataset = dba.load_SDiffDataset('InputData/SDIFF_dataset.pkl')
    train_size = int(len(dataset) * 0.7)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(8))
    data_size = len(dataset)
    data_loader = DataLoader(train_dataset, batch_size=cmd_args.batch_size, drop_last=True, shuffle=True)
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
    torch.save(best_model, model_name)
    input = test_dataset.dataset.z.reshape(test_dataset.dataset.z.shape[0], -1).to(torch.float32).clone().detach().cuda()
    prediction = mlp(input).detach().cpu().numpy()
    origin = test_dataset.dataset.sdiff.detach().cpu().numpy()
    mse = np.mean((prediction - origin) ** 2)
    rmse = mse ** 0.5
    print(mode + '----mse:' + str(mse))
    print(mode + '----rmse:' + str(rmse))
    return prediction, origin, mse, rmse


def t_fm(device=0):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    dataset = FMDataset()
    dataset.xa = torch.cat((dataset.xa[0: -2], dataset.xa[-1:]), dim=0)
    dataset.xc = torch.cat((dataset.xc[0: -2], dataset.xc[-1:]), dim=0)
    dataset.y = torch.cat((dataset.y[0: -2], dataset.y[-1:]), dim=0)
    train_size = int(len(dataset) * 0.7)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(8))
    data_size = len(dataset)
    data_loader = DataLoader(train_dataset, batch_size=cmd_args.batch_size, drop_last=True, shuffle=True)
    model = DeepFM()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.001)
    # model.fit(loader_train=data_loader, optimizer=optimizer, mode='test')
    model.fit(loader_train=data_loader, test_dataset=test_dataset, optimizer=optimizer, mode='test')
    model = DeepFM()
    model.load_state_dict(torch.load('SavedModel/FM_test.pkl'))
    model.cuda()
    model.eval()
    prediction = model(test_dataset.dataset.xa.to(torch.float32).clone().detach().cuda(),
                       test_dataset.dataset.xc.to(torch.float32).clone().detach().cuda()).detach().cpu().numpy()
    origin = test_dataset.dataset.y.detach().cpu().numpy()
    mse = np.mean((prediction - origin) ** 2)
    rmse = mse ** 0.5
    print('FM----mse:' + str(mse))
    print('FM----rmse:' + str(rmse))
    return prediction, origin, mse, rmse


def t_gp():
    dataset = dba.load_SDiffDataset('InputData/SDIFF_dataset.pkl')
    train_size = int(len(dataset) * 0.7)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(8))
    data_size = len(dataset)
    input = train_dataset.dataset.z.reshape(train_dataset.dataset.z.shape[0], -1).numpy()
    output = train_dataset.dataset.sdiff.reshape(train_dataset.dataset.sdiff.shape[0], -1).numpy()
    test_input = test_dataset.dataset.z.reshape(test_dataset.dataset.z.shape[0], -1).numpy()
    test_output = test_dataset.dataset.sdiff.reshape(test_dataset.dataset.sdiff.shape[0], -1).numpy()
    kernel = C(0.1, (0.001, 0.1)) * RBF(0.5, (1e-4, 10))
    reg = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)
    reg.fit(input, output)
    prediction, err = reg.predict(test_input, return_std=True)
    mse = np.mean((prediction - test_output) ** 2)
    rmse = mse ** 0.5
    return mse, rmse


def baseline():
    dataset = dba.load_SDiffDataset('InputData/SDIFF_dataset.pkl')
    train_size = int(len(dataset) * 0.7)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(8))
    test_input = test_dataset.dataset.z.reshape(test_dataset.dataset.z.shape[0], -1).numpy()
    test_output = test_dataset.dataset.sdiff.reshape(test_dataset.dataset.sdiff.shape[0], -1).numpy()

    prediction = np.random.rand(test_output.shape[0], test_output.shape[1]) * 0.4769
    mse = np.mean((prediction - test_output) ** 2)
    rmse = mse ** 0.5
    return mse, rmse


if __name__ == '__main__':
    # prediction, origin, tlmlp_mse, tlmlp_rmse = t_mlp(mode='transfer learning')
    # prediction, origin, mlp_mse, mlp_rmse = t_mlp(mode='normal')
    # prediction, origin, fm_mse, fm_rmse = t_fm()
    # mse, rmse = t_gp()
    mse, rmse = baseline()
    # print('np.var(prediction), np.var(origin), np.mean(prediction), np.mean(origin)')
    # print(np.var(prediction), np.var(origin), np.mean(prediction), np.mean(origin))
    # plt.scatter(origin, prediction)
    pass
