import torch
import numpy as np
import torch.nn as nn
from torchdiffeq import odeint
from tqdm.auto import tqdm

max_epochs = 300
width = 12
depth = 4
learning_rate = 3e-3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def epoch(X, y, model, loss_fun, opt, device=device):
    total_loss, total_err = 0., 0.
    n_iter = X.shape[0]
    for i in range(n_iter):
        Xd, yd = X[i].to(device), y[i].to(device)
        opt.zero_grad()
        yp = model(Xd)
        loss = loss_fun(yp.squeeze(), yd.squeeze())
        loss.backward()
        opt.step()
        total_loss += loss.item() * X.shape[0]
    return total_loss / len(x)

class MLP(nn.Module):
    def __init__(self, width, depth, activation=nn.Tanh, in_dim=1, out_dim=1, bias=True, linear=nn.Linear):
        super().__init__()
        self.layers = []
        self.layers.append(linear(in_dim, width, bias=bias))
        for i in range(depth):
            self.layers.append(linear(width, width, bias=bias))
        self.layers.append(linear(width, out_dim, bias=bias))
        self.layers = nn.ModuleList(self.layers)
        self.activation = activation()
        self.out = nn.Identity()

    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        x = self.out(self.layers[-1](x))
        return x


class ReservoirEquation(nn.Module):

    def __init__(self, K):
        super().__init__()
        self.K = K

    def forward(self, t, x):
        return self.K(x) * x
    
class TorchDiffEQNeuralReservoir(nn.Module):

    def __init__(self, width, depth, solver_method="implicit_adams"):
        super().__init__()
        self.K = MLP(width, depth)
        self.res = ReservoirEquation(self.K)
        self.time = torch.tensor([0, 1], dtype=torch.float64)
        self.method = solver_method

    def forward(self, x):
        S_all = odeint(self.res, x, self.time, method=self.method)
        return S_all[-1]

model = TorchDiffEQNeuralReservoir(width, depth).to(device)
loss_fun = torch.nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss_history = []
for i in tqdm(range(max_epochs)):
    train_loss = epoch(train_x, train_Y, model, loss_fun, opt)
    loss_history.append(train_loss)

v = torch.tensor(np.arange(0.0, 1.0, step=0.0001), dtype=dtype)
