import numpy as np
from numpy import random
import csv
import os
import torch
from torch import autograd

from rdkit import Chem
from rdkit.Chem import Draw
from tqdm import trange

from C_Evaluation.att_model_proxy import AttMolProxy
from D_optimization.DD1_factorization_machine import DeepFM
from Util_mol.SAscorer import calculateScore
from cmd_args import cmd_args


class GradientPSO:
    def __init__(self, edge=(-2., 2.), dim=cmd_args.anion_latent_dim + cmd_args.cation_latent_dim,
                 max_iter_=100, c1=0.5, c2=0.499, c3=0.001, particle_number=100, model_name='SavedModel/FM.pkl',
                 use_grad=True):
        self.edge = edge
        self.w = 0.8  # 惯性
        self.c1 = c1  # 自身最优影响比例
        self.c2 = c2  # 全局最优影响比例
        self.c3 = c3  # 梯度影响比例
        self.particle_number = particle_number
        self.dim = dim
        self.max_iter = max_iter_
        self.x = np.zeros((self.particle_number, self.dim))  # 粒子坐标
        self.v = np.zeros((self.particle_number, self.dim))  # 粒子速度
        self.grad = np.zeros((self.particle_number, self.dim))  # 粒子梯度
        self.p_best = np.zeros((self.particle_number, self.dim))  # 粒子最优坐标
        self.g_best = np.zeros(self.dim)  # 全局最优坐标
        self.p_fit = np.zeros(self.particle_number)  # 粒子最优值
        self.g_fit = 10.  # 全局最优值
        self.model = DeepFM()
        self.model.load_state_dict(torch.load(model_name))
        self.model.cuda()
        self.model.eval()
        self.a_model = AttMolProxy('Anion')
        self.c_model = AttMolProxy('Cation')
        self.use_grad = use_grad

    def function(self, x):
        a_tensor = torch.tensor(x[:cmd_args.anion_latent_dim]).reshape(1, -1).to(torch.float32)
        a_tensor = torch.cat((a_tensor, torch.zeros((1, cmd_args.cation_latent_dim - cmd_args.anion_latent_dim))),
                             dim=1).to(torch.float32).clone().detach().cuda()
        c_tensor = torch.tensor(x[cmd_args.anion_latent_dim:]).to(torch.float32).reshape(1, -1).to(
            torch.float32).clone().detach().cuda()
        sdiff = self.model(a_tensor, c_tensor).detach().cpu().numpy()
        a_smiles = self.a_model.decode(np.array(x[:cmd_args.anion_latent_dim].reshape(1, -1), dtype='f'
                                                ), use_random=False)[0]
        c_smiles = self.c_model.decode(np.array(x[cmd_args.anion_latent_dim:].reshape(1, -1), dtype='f'
                                                ), use_random=False)[0]
        a_mol = Chem.MolFromSmiles(a_smiles)
        c_mol = Chem.MolFromSmiles(c_smiles)
        il_like = 0.
        if '-' in a_smiles and '+' in c_smiles and a_mol is not None and c_mol is not None and 0. < sdiff < 5:
            if a_smiles.count('-') == 1 and c_smiles.count('+') == 1:
                il_mol = Chem.MolFromSmiles(a_smiles + '.' + c_smiles)
                sa_score = calculateScore(il_mol)
                il_info = il_mol.GetRingInfo()
                atoms = il_mol.GetAtoms()
                ring_atom = 0
                ssr = Chem.GetSymmSSSR(il_mol)
                ring_len = [0]
                for ring in ssr:
                    ring_len.append(len(list(ring)))
                ring_len = np.array(ring_len)
                for atom in atoms:
                    ring_atom += atom.IsInRing()
                if sa_score <= 6.5 and il_info.NumRings() < 4 and ring_atom <= 18 and np.max(ring_len) < 7:
                    il_like = 10.

        if sdiff > 5.:
            sdiff = 5.
        return sdiff + il_like

    def init_population(self):
        self.x = np.random.rand(self.particle_number, self.dim) * (self.edge[1] - self.edge[0]) + self.edge[0]
        self.v = np.random.rand(self.particle_number, self.dim)
        for i in range(self.particle_number):
            self.p_best[i] = self.x[i]
            tmp = self.function(self.x[i])
            self.p_fit[i] = tmp
            if tmp > self.g_fit:
                self.g_fit = tmp
                self.g_best = self.x[i]

    def iterator(self):
        iter_bar = trange(self.max_iter)
        for t in iter_bar:
            for i in range(self.particle_number):
                temp = self.function(self.x[i])
                if temp > self.p_fit[i]:
                    self.p_fit[i] = temp
                    self.p_best[i] = self.x[i]
                    if self.p_fit[i] > self.g_fit:
                        self.g_best = self.x[i]
                        self.g_fit = self.p_fit[i]

            for i in range(self.particle_number):
                a_tensor = torch.tensor(self.x[i][:cmd_args.anion_latent_dim]
                                        ).reshape(1, -1)
                a_tensor = torch.cat(
                    (a_tensor, torch.zeros((1, cmd_args.cation_latent_dim - cmd_args.anion_latent_dim))),
                    dim=1).to(torch.float32).clone().detach().cuda().requires_grad_(True)
                c_tensor = torch.tensor(self.x[i][cmd_args.anion_latent_dim:]
                                        ).reshape(1, -1).to(torch.float32).clone().detach().cuda().requires_grad_(True)

                if self.use_grad:
                    sdiff = self.model(a_tensor, c_tensor)
                    grads = autograd.grad(sdiff, [a_tensor, c_tensor])
                    self.grad[i][:cmd_args.anion_latent_dim] = grads[0][0, :cmd_args.anion_latent_dim].detach(
                    ).cpu().numpy()
                    self.grad[i][cmd_args.anion_latent_dim:] = grads[1].detach().cpu().numpy()

                    self.v[i] = self.w * self.v[i] + random.rand() * self.c1 * (
                            self.p_best[i] - self.x[i]) + random.rand() * self.c2 * (
                                        self.g_best - self.x[i]) + random.rand() * self.c3 * self.grad[i]
                    self.x[i] = self.x[i] + self.v[i]
                else:
                    self.v[i] = self.w * self.v[i] + random.rand() * self.c1 * (
                            self.p_best[i] - self.x[i]) + random.rand() * self.c2 * (
                                        self.g_best - self.x[i])
                    self.x[i] = self.x[i] + self.v[i]
            a_smiles = self.a_model.decode(np.array(self.g_best[:cmd_args.anion_latent_dim].reshape(1, -1), dtype='f'
                                                    ), use_random=False)[0]
            c_smiles = self.c_model.decode(np.array(self.g_best[cmd_args.anion_latent_dim:].reshape(1, -1), dtype='f'
                                                    ), use_random=False)[0]
            print(a_smiles + '.' + c_smiles)
            v_mean = np.mean(np.abs(self.v), axis=1)
            if np.mean(v_mean[v_mean.argsort()[::-1][40:]]) < 0.01:
                break
            iter_bar.set_description('The best score is %.2f , the particle speed is %.6f' % (
                self.g_fit - 10., np.mean(v_mean[v_mean.argsort()[::-1][40:]])))

        a_smiles = self.a_model.decode(np.array(self.g_best[:cmd_args.anion_latent_dim].reshape(1, -1), dtype='f'
                                                ), use_random=False)[0]
        c_smiles = self.c_model.decode(np.array(self.g_best[cmd_args.anion_latent_dim:].reshape(1, -1), dtype='f'
                                                ), use_random=False)[0]
        a_mol = Chem.MolFromSmiles(a_smiles)
        c_mol = Chem.MolFromSmiles(c_smiles)
        if '-' in a_smiles and '+' in c_smiles and a_mol is not None and c_mol is not None:
            print(a_smiles + '.' + c_smiles)
            return [a_smiles, c_smiles, a_smiles + '.' + c_smiles, self.g_fit - 10.]
        else:
            print('failed')
            return [-1, -1, -1]


def gradient_pso(sample=50):
    with open('Graph_Mol/GPSO_FM_MolGraph__/smiles.csv', 'wb') as f_:
        pass

    for times in range(sample):
        pso_ = GradientPSO()
        pso_.init_population()
        result = pso_.iterator()
        if result[0] != -1:
            img_path = 'Graph_Mol/GPSO_FM_MolGraph__/' + str(times) + '.png'
            try:
                Draw.MolToFile(Chem.MolFromSmiles(result[2]), img_path)
                with open('Graph_Mol/GPSO_FM_MolGraph__/smiles.csv', 'a+') as f_:
                    csv_write = csv.writer(f_)
                    csv_write.writerow(result)
            except ValueError:
                with open('Graph_Mol/GPSO_FM_MolGraph__/smiles.csv', 'a+') as f_:
                    csv_write = csv.writer(f_)
                    csv_write.writerow('Failed')


def pso(sample=20):
    with open('Graph_Mol/PSO_FM_MolGraph__/smiles.csv', 'wb') as f_:
        pass

    for times in range(sample):
        pso_ = GradientPSO(particle_number=30, use_grad=False)
        pso_.init_population()
        result = pso_.iterator()
        if result[0] != -1:
            img_path = 'Graph_Mol/PSO_FM_MolGraph__/' + str(times) + '.png'
            try:
                Draw.MolToFile(Chem.MolFromSmiles(result[2]), img_path)
                with open('Graph_Mol/PSO_FM_MolGraph__/smiles.csv', 'a+') as f_:
                    csv_write = csv.writer(f_)
                    csv_write.writerow(result)
            except ValueError:
                with open('Graph_Mol/PSO_FM_MolGraph__/smiles.csv', 'a+') as f_:
                    csv_write = csv.writer(f_)
                    csv_write.writerow('Failed')


def observe():
    pso_ = GradientPSO()
    pso_.init_population()
    result = pso_.iterator()
    return result


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
    # gradient_pso()
    # pso()
