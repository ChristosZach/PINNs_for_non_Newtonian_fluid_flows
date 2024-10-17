import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import math
import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
from pyDOE import lhs
from scipy.integrate import quad
import pandas as pd
from sklearn.metrics import mean_squared_error
from functools import partial

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#PINN domain parameters

u0 = 0.1
D = 0.3
Npoints = 2500
Nb = 200
Nu = 500
y_u = 0.3/D
y_l = 0
x_in = 0
x_out = 0.6/D

m = 30
n = 0.4
rho = 1000

Re = rho*(D**n)*(u0**(2-n))/m

#BC points

ylB=[]
for i in range(Nb):
  ylB.append(y_l)
ylB = np.array(ylB)

ylBt = torch.from_numpy(ylB).float().reshape((ylB.size,1)).to(device)
ylBt.requires_grad=True

yuB=[]
for i in range(Nb):
  yuB.append(y_u)
yuB = np.array(yuB)

yuBt = torch.from_numpy(yuB).float().reshape((yuB.size,1)).to(device)
yuBt.requires_grad=True

xinB=[]
for i in range(Nb):
    xinB.append(x_in)
xinB = np.array(xinB)

xinBt = torch.from_numpy(xinB).float().reshape((xinB.size,1)).to(device)
xinBt.requires_grad=True

xoutB=[]
for i in range(Nb):
    xoutB.append(x_out)
xoutB = np.array(xoutB)

xoutBt = torch.from_numpy(xoutB).float().reshape((xoutB.size,1)).to(device)
xoutBt.requires_grad=True

yt = torch.from_numpy((y_u - y_l) * lhs(1, Nb) + y_l/D).float().to(device)
yt.requires_grad = True
xt = torch.from_numpy((x_out - x_in) * lhs(1, Nb) + x_in / D).float().to(device)
xt.requires_grad = True

#Observational Data Points

def read_exp_data(file_path):

    df = pd.read_excel(file_path)
    data = df.iloc[0:, ]
    data = data.iloc[:, 0:3]
    data_x = data.iloc[:, 0:1]
    data_y = data.iloc[:, 1:2]
    data_ux = data.iloc[:, 2:3]
    data_list = data.values.tolist()
    data_x_list = data_x.values.tolist()
    data_y_list = data_y.values.tolist()
    data_ux_list = data_ux.values.tolist()
    data_exp= np.array(data_list)
    x_exp= np.array(data_x_list)
    y_exp= np.array(data_y_list)
    ux_exp= np.array(data_ux_list)


    return x_exp, y_exp, ux_exp

file_path1 = 'ux_at_centerline_.xlsx'
x_exp1, y_exp1, ux_exp1 = read_exp_data(file_path1)
ind = np.linspace(0, x_exp1.shape[0]-1,10, dtype=int)
x_exp1 = x_exp1[ind]/D
y_exp1 = y_exp1[ind]/D
ux_exp1 = ux_exp1[ind]/u0
# #2% Gaussian Noise addition
# ux_exp1 = ux_exp1 + 0.02*ux_exp1*np.random.randn(10).reshape((10, 1))


x_exp_ux = torch.from_numpy(x_exp1).to(device).float()
y_exp_ux = torch.from_numpy(y_exp1).to(device).float()
ux_exp = torch.from_numpy(ux_exp1).to(device).float()

file_path1 = 'P_at_centerline_.xlsx'
x_exp1, y_exp1, P_exp1 = read_exp_data(file_path1)
ind = np.linspace(0, x_exp1.shape[0]-1,10, dtype=int)
x_exp1 = x_exp1[ind]/D
y_exp1 = y_exp1[ind]/D
P_exp1 = P_exp1[ind]/(rho*(u0**2))
# #2% Gaussian Noise additions
# P_exp1 = P_exp1 + 0.02*P_exp1*np.random.randn(10).reshape((10, 1))


x_exp_P = torch.from_numpy(x_exp1).to(device).float()
y_exp_P = torch.from_numpy(y_exp1).to(device).float()
P_exp = torch.from_numpy(P_exp1).to(device).float()

#PINN Model Formulation

# def model_parameters(module):
#     for param in module.parameters():
#         yield param
#
# def loss_weight_gen(weight):
#     yield weight



class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        # self.train_with_mseU = train_with_mseU
        self.input_dim = 2
        self.hidden_dim = 50
        self.output_dim = 6
        self.sigma = 2
        self.num_features = 30
        # self.tasks = nn.ModuleList(tasks)
        # self.num_layers = 10
        self.B = torch.normal(0, self.sigma, size=(self.input_dim, self.num_features))
        self.B = self.B.to(device)

        self.weight_f1 = nn.Parameter(torch.tensor([1.0]))
        self.weight_f2 = nn.Parameter(torch.tensor([1.0]))
        self.weight_f3 = nn.Parameter(torch.tensor([1.0]))
        self.weight_f4 = nn.Parameter(torch.tensor([1.0]))
        self.weight_f5 = nn.Parameter(torch.tensor([1.0]))
        self.weight_f6 = nn.Parameter(torch.tensor([1.0]))
        self.weight_b1 = nn.Parameter(torch.tensor([1.0]))
        self.weight_b2 = nn.Parameter(torch.tensor([1.0]))
        self.weight_b3 = nn.Parameter(torch.tensor([1.0]))
        self.weight_b3 = nn.Parameter(torch.tensor([1.0]))
        self.weight_u1 = nn.Parameter(torch.tensor([1.0]))
        self.weight_u2 = nn.Parameter(torch.tensor([1.0]))


        self.activation_fn = nn.Tanh()
        self.input_layer_fourier = nn.Linear(2*self.num_features, self.hidden_dim)
        nn.init.xavier_normal_(self.input_layer_fourier.weight)
        self.hidden_layer1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        nn.init.xavier_normal_(self.hidden_layer1.weight)
        self.hidden_layer2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        nn.init.xavier_normal_(self.hidden_layer2.weight)
        self.hidden_layer3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        nn.init.xavier_normal_(self.hidden_layer3.weight)
        self.hidden_layer4 = nn.Linear(self.hidden_dim, self.hidden_dim)
        nn.init.xavier_normal_(self.hidden_layer4.weight)
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)
        nn.init.xavier_normal_(self.output_layer.weight)

        self.mse = nn.MSELoss()

        self.m = nn.Parameter(torch.tensor(10.0, device=device))
        self.n = nn.Parameter(torch.tensor(1.0, device=device))




    def tensor_grad(self,tensor,variable_points):
        gradient = torch.autograd.grad(tensor.sum(), variable_points, create_graph=True)[0]

        return gradient


    def solution(self, x, y):
        X = torch.cat((x,y),axis=1)
        mapped_X = torch.cat((torch.sin(torch.matmul(X, self.B)),torch.cos(torch.matmul(X, self.B))), axis=1)
        u = self.forward(mapped_X)
        ux = u[:, 0:1]
        uy = u[:, 1:2]
        P = u[:, 2:3]
        s11 = u[:, 3:4]
        s22 = u[:, 4:5]
        s12 = u[:, 5:6]
        return ux, uy, P, s11, s22, s12



    def residuals(self, x, y, m, n, rho, u0):
        Re = rho*(D**self.n)*(u0**(2-self.n))/self.m
        
        ux, uy, P, s11, s22, s12 = self.solution(x, y)

        ux_x = self.tensor_grad(ux, x)
        ux_y = self.tensor_grad(ux, y)
        uy_x = self.tensor_grad(uy, x)
        uy_y = self.tensor_grad(uy, y)
        s11_1 = self.tensor_grad(s11, x)
        s22_2 = self.tensor_grad(s22, y)
        s12_1 = self.tensor_grad(s12, x)
        s12_2 = self.tensor_grad(s12, y)

        sr = (0.5*(2*ux_x**2+2*(ux_y+uy_x)**2+2*uy_y**2))**0.5
        eta = sr**(self.n-1)

        momentum_x = ux*ux_x + uy*ux_y - (1/Re)*s11_1 - (1/Re)*s12_2
        momentum_y = ux*uy_x + uy*uy_y - (1/Re)*s12_1 - (1/Re)*s22_2
        continuity = ux_x + uy_y

        f_s11 = -Re*P + 2*eta*ux_x - s11
        f_s22 = -Re*P + 2*eta*uy_y - s22
        f_s12 = eta*(ux_y + uy_x) - s12

        return momentum_x, momentum_y, continuity, f_s11, f_s22, f_s12

    def boundary_solution(self, xb, yb, xin, xout, yu, yl):

        uxin, uyin, Pin, s11in, s22in, s12in = self.solution(xin, yb)
        uxout, uyout, Pout, s11out, s22out, s12out = self.solution(xout, yb)
        uxu, uyu, Pu, s11u, s22u, s12u = self.solution(xb, yu)
        uxl, uyl, Pl, s11l, s22l, s12l = self.solution(xb, yl)

        Pin_x = self.tensor_grad(Pin, xin)
        Pu_y = self.tensor_grad(Pu, yu)
        Pl_y = self.tensor_grad(Pl, yl)
        uxout_x = self.tensor_grad(uxout, xout)
        uyout_x = self.tensor_grad(uyout, xout)

        return uxin, uyin, Pin, uxout, uyout, Pout, uxu, uyu, Pu, uxl, uyl, Pl, Pin_x, Pu_y, Pl_y, uxout_x, uyout_x


    def loss_terms(self, x, y, m, n, D, rho, u0, xb, yb, xin, xout, yu, yl, x_exp_ux, y_exp_ux, x_exp_P, y_exp_P, ux_exp, P_exp):
        momentum_x, momentum_y, continuity, f_s11, f_s22, f_s12 = self.residuals(x, y, m, n, rho, u0)
        mseF1 = self.mse(momentum_x,torch.zeros_like(momentum_x))
        mseF2 = self.mse(momentum_y, torch.zeros_like(momentum_y))
        mseF3 = self.mse(continuity, torch.zeros_like(continuity))
        mseF4 = self.mse(f_s11, torch.zeros_like(f_s11))
        mseF5 = self.mse(f_s22, torch.zeros_like(f_s22))
        mseF6 = self.mse(f_s12, torch.zeros_like(f_s12))

        uxin, uyin, Pin, uxout, uyout, Pout, uxu, uyu, Pu, uxl, uyl, Pl, Pin_x, Pu_y, Pl_y, uxout_x, uyout_x = self.boundary_solution(xb, yb, xin, xout, yu, yl)
        mseB1_1 = self.mse(uxin, 4*(D-D*yb)*D*yb/(D**2))
        mseB1_2 = self.mse(uyin, torch.zeros_like(uyin))
        mseB1 = mseB1_1 + mseB1_2
        mseB2_1 = self.mse(uxout_x, torch.zeros_like(uxout_x))
        mseB2_2 = self.mse(uyout_x , torch.zeros_like(uyout_x))
        mseB2_3 = self.mse(Pout , torch.zeros_like(Pout))
        mseB2 = mseB2_1 + mseB2_2 + mseB2_3
        mseB3_1 = self.mse(uxu, torch.zeros_like(uxu))
        mseB3_2 = self.mse(uyu, torch.zeros_like(uyu))

        mseB4_1 = self.mse(uxl, torch.zeros_like(uxl))
        mseB4_2 = self.mse(uyl, torch.zeros_like(uyl))
        mseB3 = mseB3_1 + mseB3_2 + mseB4_1 + mseB4_2

        ux_exp1, uy_exp1, P_exp1, s11_exp1, s22_exp1, s12_exp1 = self.solution(x_exp_ux, y_exp_ux)
        ux_exp2, uy_exp2, P_exp2, s11_exp2, s22_exp2, s12_exp2 = self.solution(x_exp_P, y_exp_P)

        mseU1 = self.mse(ux_exp, ux_exp1)
        mseU2 = self.mse(P_exp, P_exp2)

        return mseF1, mseF2, mseF3, mseF4, mseF5, mseF6, mseB1, mseB2, mseB3, mseU1, mseU2

    def loss(self, x, y, m, n, D, rho, u0, xb, yb, xin, xout, yu, yl, x_exp_ux, y_exp_ux, x_exp_P, y_exp_P, ux_exp, P_exp):

        mseF1, mseF2, mseF3, mseF4, mseF5, mseF6, mseB1, mseB2, mseB3, mseU1, mseU2 = self.loss_terms(x, y, m, n, D, rho, u0, xb, yb, xin, xout, yu, yl, x_exp_ux, y_exp_ux, x_exp_P, y_exp_P, ux_exp, P_exp)
        loss = self.weight_f1*mseF1 + self.weight_f2*mseF2 + self.weight_f3*mseF3 + self.weight_f4*mseF4 + self.weight_f5*mseF5 + self.weight_f6*mseF6 + self.weight_b1*mseB1 + self.weight_b2*mseB2 + self.weight_b3*mseB3
        non_weighted_loss = mseF1 + mseF2 + mseF3 + mseF4 + mseF5 + mseF6 + mseB1 + mseB2 + mseB3

        return loss, non_weighted_loss


    def loss_mseu(self, x, y, m, n, D, rho, u0, xb, yb, xin, xout, yu, yl, x_exp_ux, y_exp_ux, x_exp_P, y_exp_P, ux_exp, P_exp):

        mseF1, mseF2, mseF3, mseF4, mseF5, mseF6, mseB1, mseB2, mseB3, mseU1, mseU2 = self.loss_terms(x, y, m, n, D, rho, u0, xb, yb, xin, xout, yu, yl, x_exp_ux, y_exp_ux, x_exp_P, y_exp_P, ux_exp, P_exp)
        loss = self.weight_f1*mseF1 + self.weight_f2*mseF2 + self.weight_f3*mseF3 + self.weight_f4*mseF4 + self.weight_f5*mseF5 + self.weight_f6*mseF6 + self.weight_b1*mseB1 + self.weight_b2*mseB2 + self.weight_b3*mseB3 + self.weight_u1 * mseU1 + self.weight_u2 * mseU2


        return loss

    #standard MLP (Random Fourier Embeddings)
    def forward(self, x):
        x = self.input_layer_fourier(x)

        x = self.activation_fn(x)

        x = self.hidden_layer1(x)
        x = self.activation_fn(x)
        x = self.hidden_layer2(x)
        x = self.activation_fn(x)
        x = self.hidden_layer3(x)
        x = self.activation_fn(x)
        x = self.hidden_layer4(x)
        x = self.activation_fn(x)
        x = self.output_layer(x)

        return x

    def neural_network_parameters_mseu(self):
        # Return only the neural network parameters
        return [param for name, param in self.named_parameters() if name not in ['weight_f1', 'weight_f2', 'weight_f3', 'weight_f4', 'weight_f5', 'weight_f6', 'weight_b1', 'weight_b2', 'weight_b3', 'weight_u1', 'weight_u2']]

    def neural_network_parameters(self):
        # Return only the neural network parameters
        return [param for name, param in self.named_parameters() if name not in ['m', 'n', 'weight_f1', 'weight_f2', 'weight_f3', 'weight_f4', 'weight_f5', 'weight_f6', 'weight_b1', 'weight_b2', 'weight_b3', 'weight_u1', 'weight_u2']]

    def update_weights_mseu(self, gradients):
        with torch.no_grad():
            learning_rates = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            for weight, grad, lr in zip(
                    [self.weight_f1, self.weight_f2, self.weight_f3, self.weight_f4, self.weight_f5, self.weight_f6,
                     self.weight_b1, self.weight_b2, self.weight_b3, self.weight_u],
                    gradients, learning_rates):
                weight += lr * grad

    def update_weights(self, gradients):
        with torch.no_grad():
            learning_rates = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            for weight, grad, lr in zip(
                    [self.weight_f1, self.weight_f2, self.weight_f3, self.weight_f4, self.weight_f5,
                     self.weight_f6,
                     self.weight_b1, self.weight_b2, self.weight_b3],
                    gradients, learning_rates):
                weight += lr * grad


def read_data(file_path):

    df = pd.read_excel(file_path)
    data = df.iloc[5:, ]
    data_x = data.iloc[:, 0:1]
    data_y = data.iloc[:, 1:2]
    data_P = data.iloc[:, 3:4]
    data_ux = data.iloc[:, 4:5]
    data_uy = data.iloc[:, 5:6]
    data_list = data.values.tolist()
    data_x_list = data_x.values.tolist()
    data_y_list = data_y.values.tolist()
    data_P_list = data_P.values.tolist()
    data_ux_list = data_ux.values.tolist()
    data_uy_list = data_uy.values.tolist()

    data_array = np.array(data_list)
    x_sim = np.array(data_x_list)/D
    y_sim = np.array(data_y_list)/D
    P_sim = np.array(data_P_list)/(rho*(u0**2))
    ux_sim = np.array(data_ux_list)/u0
    uy_sim = np.array(data_uy_list)/u0

    x_t = torch.from_numpy(x_sim).float()
    # x_t.requires_grad = True
    y_t = torch.from_numpy(y_sim).float()
    # y_t.requires_grad = True

    ux_t = torch.from_numpy(ux_sim).float()
    uy_t = torch.from_numpy(uy_sim).float()
    P_t = torch.from_numpy(P_sim).float()



    return x_sim, y_sim, x_t,y_t, ux_sim, uy_sim, P_sim, ux_t, uy_t, P_t

def validate(ux_sim, uy_sim, P_sim,ux_pred, uy_pred, P_pred):


    U_pred = torch.cat((ux_pred,uy_pred, P_pred),axis=1)
    U_pred_array = U_pred.detach().cpu().numpy()
    U_sim_array = np.concatenate((ux_sim, uy_sim, P_sim),axis=1)

    mse_error_total = mean_squared_error(U_sim_array, U_pred_array)
    L2_rel_error_total = np.linalg.norm(U_sim_array - U_pred_array) / np.linalg.norm(U_sim_array)

    ux_pred_array = ux_pred.detach().cpu().numpy()
    mse_error_ux = mean_squared_error(ux_sim, ux_pred_array)
    L2_rel_error_ux= np.linalg.norm(ux_sim - ux_pred_array) / np.linalg.norm(ux_sim)

    uy_pred_array = uy_pred.detach().cpu().numpy()
    mse_error_uy = mean_squared_error(uy_sim, uy_pred_array)
    L2_rel_error_uy= np.linalg.norm(uy_sim - uy_pred_array) / np.linalg.norm(uy_sim)

    P_pred_array = P_pred.detach().cpu().numpy()
    mse_error_P = mean_squared_error(P_sim, P_pred_array)
    L2_rel_error_P= np.linalg.norm(P_sim - P_pred_array) / np.linalg.norm(P_sim)

    return mse_error_total, L2_rel_error_total, mse_error_ux, L2_rel_error_ux, mse_error_uy, L2_rel_error_uy, mse_error_P, L2_rel_error_P


loss_history = list(np.random.rand(100))

loss_array = []
validation_loss_array =[]
weight_f_array = []
weight_b_array = []

#Validation Data (from Ansys Fluent)

x_sim_array, y_sim_array, x_sim,y_sim, ux_sim, uy_sim, P_sim, ux_sim_t, uy_sim_t, P_sim_t = read_data('export_n=0.8_m=16.3.xlsx')
x_sim = x_sim.to(device)
y_sim = y_sim.to(device)
ux_sim_t = ux_sim_t.to(device)
uy_sim_t = uy_sim_t.to(device)
P_sim_t = P_sim_t.to(device)

from torch.utils.data import DataLoader, Dataset, TensorDataset

# Assuming lhs is a function that generates the dataset
def generate_data(Npoints, D):
    y_points = (y_u - y_l) * lhs(1, Npoints) + y_l / D
    x_points = (x_out - x_in) * lhs(1, Npoints) + x_in / D
    return x_points, y_points


class CustomDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]
        return x, y



from torch.utils.data import DataLoader, Dataset, TensorDataset


class CustomDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]
        return x, y


x_points, y_points = generate_data(Npoints, D)

fig = plt.figure(figsize=(8,4.8))
plt.scatter(x_points,y_points, s=5)
plt.show()

# Convert to tensors
x_points = torch.from_numpy(x_points).float()
y_points = torch.from_numpy(y_points).float()
x_points= x_points.to(device)
y_points = y_points.to(device)

# Create dataset and dataloader
dataset = CustomDataset(x_points, y_points)
batch_size = 250
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

def loss_grad(loss):
    
    gradients = torch.autograd.grad(loss, model.parameters(), create_graph=True, allow_unused = True)
    total_norm = 0.0
    for grad in gradients:
        if grad is not None:
            param_norm = grad.norm(2)
            total_norm += param_norm.item() ** 2

    total_norm = total_norm ** 0.5
    
    return total_norm

loss_history = list(np.random.rand(100))
weight_f1_array = []
weight_f2_array = []
weight_f3_array = []
weight_f4_array = []
weight_f5_array = []
weight_f6_array = []
weight_b1_array = []
weight_b2_array = []
weight_b3_array = []
weight_u1_array = []
weight_u2_array = []
m_array = []
n_array = []

model = PINN().to(device)
# torch.save(model.state_dict(),'model_pretrain_state.pth')

#Optimizer parameters
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
factor = 0.0
lr_lambda = lambda iter: 1/(1+factor*iter)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

optimizer_nn = torch.optim.Adam(model.neural_network_parameters(),lr=learning_rate)
scheduler_nn = torch.optim.lr_scheduler.LambdaLR(optimizer_nn, lr_lambda)

optimizer_nn_mseu = torch.optim.Adam(model.neural_network_parameters_mseu(),lr=learning_rate)
scheduler_nn_mseu = torch.optim.lr_scheduler.LambdaLR(optimizer_nn_mseu, lr_lambda)

opt_m = torch.optim.Adam([model.m],lr=1e-3)
scheduler_m = torch.optim.lr_scheduler.LambdaLR(opt_m, lr_lambda)
opt_n = torch.optim.Adam([model.n],lr=1e-3)
scheduler_n = torch.optim.lr_scheduler.LambdaLR(opt_n, lr_lambda)

optimizer_loss_weights = torch.optim.SGD(
    [model.weight_f1, model.weight_f2, model.weight_f3,
        model.weight_f4, model.weight_f5, model.weight_f6,
        model.weight_b1, model.weight_b2, model.weight_b3],lr=1, maximize=True)
opt_weight_f1 = torch.optim.Adam([model.weight_f1],lr=1, maximize=True)
opt_weight_f2 = torch.optim.Adam([model.weight_f2],lr=1, maximize=True)
opt_weight_f3 = torch.optim.Adam([model.weight_f3],lr=10, maximize=True)
opt_weight_f4 = torch.optim.Adam([model.weight_f4],lr=10, maximize=True)
opt_weight_f5 = torch.optim.Adam([model.weight_f5],lr=1, maximize=True)
opt_weight_f6 = torch.optim.Adam([model.weight_f6],lr=1, maximize=True)
opt_weight_b1 = torch.optim.Adam([model.weight_b1],lr=100, maximize=True)
opt_weight_b2 = torch.optim.Adam([model.weight_b2],lr=100, maximize=True)
opt_weight_b3 = torch.optim.Adam([model.weight_b3],lr=100, maximize=True)
opt_weight_u1 = torch.optim.SGD([model.weight_u1],lr=100, maximize=True)
opt_weight_u2 = torch.optim.SGD([model.weight_u2],lr=100, maximize=True)


training_loss_history = []
loss_f1_history = []
loss_f2_history = []
loss_f3_history = []
loss_f4_history = []
loss_f5_history = []
loss_f6_history = []
loss_b1_history = []
loss_b2_history = []
loss_b3_history = []
loss_u1_history = []
loss_u2_history = []
validation_loss_history = []
l2_rel_ux_history = []
l2_rel_uy_history = []
l2_rel_P_history = []
max_iter_Adam = 8000 #Maximum PINN iterations
iter_mseu = 0 #Inverse Problem formulation from 1st iteration

print("Adam Training")
for iter in range(max_iter_Adam):
    iter = iter + 1

    if iter%50==0:
        x_points, y_points = generate_data(Npoints, D)
        # Convert to tensors
        x_points = torch.from_numpy(x_points).float()
        y_points = torch.from_numpy(y_points).float()
        x_points = x_points.to(device)
        y_points = y_points.to(device)

        # Create dataset and dataloader
        dataset = CustomDataset(x_points, y_points)
        batch_size = 250  # Example batch size
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for batch in dataloader:

        x_batch, y_batch = batch
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        x_batch.requires_grad = True
        y_batch.requires_grad = True

        if iter>iter_mseu:
            loss = model.loss_mseu(x_batch, y_batch, m, n, D, rho, u0, xt, yt, xinBt, xoutBt, yuBt, ylBt, x_exp_ux, y_exp_ux, x_exp_P, y_exp_P, ux_exp, P_exp)
        else:
            loss, non_weighted_loss = model.loss(x_batch, y_batch, m, n, D, rho, u0, xt, yt, xinBt, xoutBt, yuBt, ylBt, x_exp_ux, y_exp_ux, x_exp_P, y_exp_P, ux_exp, P_exp)

        optimizer.zero_grad()

        loss.backward()

        optimizer_nn.step()
        scheduler_nn.step()

        if iter > iter_mseu:
            opt_m.step()
            opt_n.step()



    loss_history.append(loss.data.item())

    if loss_history[-1] < loss_history[-2]: #and loss_history[-2] < loss_history[-3] and loss_history[-1] < loss_history[-10]:
        opt_weight_f1.step()
        opt_weight_f2.step()
        opt_weight_f3.step()
        opt_weight_f4.step()
        opt_weight_f5.step()
        opt_weight_f6.step()
        opt_weight_b1.step()
        opt_weight_b2.step()
        opt_weight_b3.step()

        if iter > iter_mseu:
            opt_weight_u1.step()
            opt_weight_u2.step()

    weight_f1_array.append(np.log10(model.weight_f1[0].item()))
    weight_f2_array.append(np.log10(model.weight_f2[0].item()))
    weight_f3_array.append(np.log10(model.weight_f3[0].item()))
    weight_f4_array.append(np.log10(model.weight_f4[0].item()))
    weight_f5_array.append(np.log10(model.weight_f5[0].item()))
    weight_f6_array.append(np.log10(model.weight_f6[0].item()))
    weight_b1_array.append(np.log10(model.weight_b1[0].item()))
    weight_b2_array.append(np.log10(model.weight_b2[0].item()))
    weight_b3_array.append(np.log10(model.weight_b3[0].item()))
    weight_u1_array.append(np.log10(model.weight_u1[0].item()))
    weight_u2_array.append(np.log10(model.weight_u2[0].item()))
    m_array.append(model.m.data.item())
    n_array.append(model.n.data.item())

    if iter%1 == 0:
        print(iter,"Training Loss:","%.3E" % loss.data.item())
        training_loss_history.append(loss.data.item())
        print(iter,"m parameter", model.m.data.item())
        print(iter,"n parameter", model.n.data.item())

    if iter%50== 0:
        mseF1, mseF2, mseF3, mseF4, mseF5, mseF6, mseB1, mseB2, mseB3, mseU1, mseU2 = model.loss_terms(x_batch, y_batch, m, n, D, rho, u0, xt, yt, xinBt, xoutBt, yuBt, ylBt, x_exp_ux, y_exp_ux, x_exp_P, y_exp_P, ux_exp, P_exp)
        print(iter, "F1 Loss:", "%.2E" % mseF1)
        print(iter, "F2 Loss:", "%.2E" % mseF2)
        print(iter, "F3 Loss:", "%.2E" % mseF3)
        print(iter, "F4 Loss:", "%.2E" % mseF4)
        print(iter, "F5 Loss:", "%.2E" % mseF5)
        print(iter, "F6 Loss:", "%.2E" % mseF6)
        print(iter, "B1 Loss:", "%.2E" % mseB1)
        print(iter, "B2 Loss:", "%.2E" % mseB2)
        print(iter, "B3 Loss:", "%.2E" % mseB3)
        print(iter, "U1 Loss:", "%.2E" % mseU1)
        print(iter, "U2 Loss:", "%.2E" % mseU2)
        loss_f1_history.append(mseF1.data.item())
        loss_f2_history.append(mseF2.data.item())
        loss_f3_history.append(mseF3.data.item())
        loss_f4_history.append(mseF4.data.item())
        loss_f5_history.append(mseF5.data.item())
        loss_f6_history.append(mseF6.data.item())
        loss_b1_history.append(mseB1.data.item())
        loss_b2_history.append(mseB2.data.item())
        loss_b3_history.append(mseB3.data.item())
        loss_u1_history.append(mseU1.data.item())
        loss_u2_history.append(mseU2.data.item())

    if iter%50== 0:
        print(iter, "F1 Weight:", "%.2E" % model.weight_f1.data.item())
        print(iter, "F2 Weight:", "%.2E" % model.weight_f2.data.item())
        print(iter, "F3 Weight:", "%.2E" % model.weight_f3.data.item())
        print(iter, "F4 Weight:", "%.2E" % model.weight_f4.data.item())
        print(iter, "F5 Weight:", "%.2E" % model.weight_f5.data.item())
        print(iter, "F6 Weight:", "%.2E" % model.weight_f6.data.item())
        print(iter, "B1 Weight:", "%.2E" % model.weight_b1.data.item())
        print(iter, "B2 Weight:", "%.2E" % model.weight_b2.data.item())
        print(iter, "B3 Weight:", "%.2E" % model.weight_b3.data.item())
        print(iter, "U1 Weight:", "%.2E" % model.weight_u1.data.item())
        print(iter, "U2 Weight:", "%.2E" % model.weight_u2.data.item())


    if iter%50 == 0:
        print(iter, "Learning Rate:", optimizer_nn.param_groups[0]["lr"])
        ux_pred, uy_pred, P_pred, s11_pred, s22_pred, s12_pred = model.solution(x_sim, y_sim)
        mse_error_total, L2_rel_error_total, mse_error_ux, L2_rel_error_ux, mse_error_uy, L2_rel_error_uy, mse_error_P, L2_rel_error_P = validate(ux_sim, uy_sim, P_sim, ux_pred, uy_pred, P_pred)
        validation_loss_history.append(mse_error_total)
        print(iter,"Total MSE Validation Loss:","%.3E" % mse_error_total,"Total Rel L2 Error:","%.3E" % L2_rel_error_total)
        # ux_error = mean_squared_error(ux_pred.cpu().detach().numpy(), ux_sim.detach().numpy())
        print(iter,"x-velocity MSE Loss:","%.3E" % mse_error_ux,"x-velocity L2 Relative Error:","%.3E" % L2_rel_error_ux)
        # uy_error = mean_squared_error(uy_pred.cpu().detach().numpy(), uy_sim.detach().numpy())
        print(iter,"y-velocity MSE Loss:","%.3E" % mse_error_uy,"y-velocity L2 Relative Error:","%.3E" % L2_rel_error_uy)
        # P_error = mean_squared_error(P_pred.cpu().detach().numpy(), P_sim.detach().numpy())
        print(iter,"Pressure MSE Loss:","%.3E" % mse_error_P,"Pressure L2 Relative Error:","%.3E" % L2_rel_error_P)
        l2_rel_ux_history.append(L2_rel_error_ux)
        l2_rel_uy_history.append(L2_rel_error_uy)
        l2_rel_P_history.append(L2_rel_error_P)


    if iter % 50 == 0:
        abs_error_ux = abs(ux_sim - ux_pred.cpu().detach().numpy())
        abs_error_uy = abs(uy_sim - uy_pred.cpu().detach().numpy())
        abs_error_P = abs(P_sim - P_pred.cpu().detach().numpy())
        fig, axs = plt.subplots(2, 2)
        fig.set_figheight(4)
        fig.set_figwidth(4)
        axs[0, 0].tricontourf(x_sim_array.reshape(1, 3720)[0], y_sim_array.reshape(1, 3720)[0],
                              ux_pred.cpu().detach().numpy().reshape(1,3720)[0])
        axs[0, 0].set_title('u_x predicted', fontsize=9)
        axs[0, 1].tricontourf(x_sim_array.reshape(1, 3720)[0], y_sim_array.reshape(1, 3720)[0],
                              uy_pred.cpu().detach().numpy().reshape(1,3720)[0])
        axs[0, 1].set_title('u_y predicted', fontsize=9)
        axs[1, 0].tricontourf(x_sim_array.reshape(1, 3720)[0], y_sim_array.reshape(1, 3720)[0],
                              P_pred.cpu().detach().numpy().reshape(1,3720)[0])
        axs[1, 0].set_title('P predicted', fontsize=9)

        fig.delaxes(axs[1, 1])
        # axs[1, 1].plot(x, -y, 'tab:red')
        # axs[1, 1].set_title('Axis [1, 1]')
        for ax in axs.flat:
            if ax:  # Check if the axis still exists
                ax.set_xlabel('x', fontsize=9)  # Change axis label font size
                ax.set_ylabel('y', fontsize=9)
                ax.tick_params(axis='both', which='major', labelsize=9)  # Change tick label font size
        for ax in axs.flat:
            ax.label_outer()

        plt.show()

#Figures

fig = plt.figure(figsize=(8,4.8))
plt.plot(weight_f1_array,  label='Residuals F1 Loss Weight')
plt.plot(weight_f2_array, label='Residuals F2 Loss Weight')
plt.plot(weight_f3_array, label='Residuals F3 Loss Weight')
plt.plot(weight_f4_array, label='Residuals F4 Loss Weight')
plt.plot(weight_f5_array, label='Residuals F5 Loss Weight')
plt.plot(weight_f6_array, label='Residuals F6 Loss Weight')
plt.plot(weight_b1_array, label='Boundaries B1 Loss Weight')
plt.plot(weight_b2_array, label='Boundaries B2 Loss Weight')
plt.plot(weight_b3_array, label='Boundaries B3 Loss Weight')
# plt.plot(weight_u_array, label='Boundaries U Loss Weight')
plt.legend(fontsize="8", loc="lower right")
plt.ylabel('log10(loss function weight)',loc='center')
plt.xlabel('Number of Iterations')
# plt.savefig('exp_pinn_loss_weights.png', dpi=200)
plt.show()

fig = plt.figure(figsize=(8,4.8))
plt.plot(np.log10(loss_f1_history),  label='Residuals F1 Loss')
plt.plot(np.log10(loss_f2_history),  label='Residuals F2 Loss')
plt.plot(np.log10(loss_f3_history),  label='Residuals F3 Loss')
plt.plot(np.log10(loss_f4_history),  label='Residuals F4 Loss')
plt.plot(np.log10(loss_f5_history),  label='Residuals F5 Loss')
plt.plot(np.log10(loss_f6_history),  label='Residuals F6 Loss')
plt.plot(np.log10(loss_b1_history),  label='Boundaries B1 Loss')
plt.plot(np.log10(loss_b2_history),  label='Boundaries B2 Loss')
plt.plot(np.log10(loss_b3_history),  label='Boundaries B3 Loss')
# plt.plot(weight_u_array, label='Boundaries U Loss Weight')
plt.legend(fontsize="8", loc="upper right")
plt.ylabel('log10(loss function)',loc='center')
plt.xlabel('Number of Iterations')
# plt.savefig('exp_pinn_loss_weights.png', dpi=200)
plt.show()

fig = plt.figure(figsize=(8,4.8))
plt.plot(np.log10(l2_rel_ux_history),  label='ux rel L2 error')
plt.plot(np.log10(l2_rel_uy_history),  label='uy rel L2 error')
plt.plot(np.log10(l2_rel_P_history),  label='P rel L2 error')
# plt.plot(weight_u_array, label='Boundaries U Loss Weight')
plt.legend(fontsize="8", loc="upper right")
plt.ylabel('log10(relative l2 error)',loc='center')
plt.xlabel('Number of Iterations')
# plt.savefig('exp_pinn_loss_weights.png', dpi=200)
plt.show()

fig = plt.figure(figsize=(8,4.8))
# plt.plot(m_array,  label='m parameter')
plt.plot(n_array, label='n parameter')
plt.ylabel('Power-Law n parameter value',loc='center')
plt.xlabel('Number of Iterations')
# plt.savefig('exp_pinn_loss_weights.png', dpi=200)
plt.show()

fig = plt.figure(figsize=(8,4.8))
# plt.plot(m_array,  label='m parameter')
plt.plot(m_array, label='m parameter')
plt.ylabel('Power-Law m parameter value',loc='center')
plt.xlabel('Number of Iterations')
# plt.savefig('exp_pinn_loss_weights.png', dpi=200)
plt.show()

fig = plt.figure(figsize=(8,4.8))
subfig1 = fig.add_subplot(2,1,1)#plt.subplot(2, 1, 1)
plt.plot(m_array, label='m parameter')
plt.ylabel('m parameter',loc='center')
# plt.xlabel('Number of Iterations')

subfig2 = fig.add_subplot(2,1,2)
plt.plot(n_array, 'r', label='n parameter')
plt.ylabel('n parameter',loc='center')
plt.xlabel('Number of Iterations')
plt.savefig('m_n_parameters_vs_iterations_structured.png', dpi=200)
plt.show()
