import torch
import numpy as np
import torch.nn as nn
from torchdiffeq import odeint
from tqdm.auto import tqdm
import xarray as xr

max_epochs = 300
width = 12
depth = 4
learning_rate = 3e-3

dtype = torch.float32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def epoch(X, y, model, loss_fun, opt, device=device):
    total_loss, total_err = 0.0, 0.0
    n_iter = X.shape[0]
    for i in range(n_iter):
        Xd, yd = X[i].to(device), y[i].to(device)
        opt.zero_grad()
        yp = model(Xd)
        loss = loss_fun(yp.squeeze(), yd.squeeze())
        loss.backward()
        opt.step()
        total_loss += loss.item() * X.shape[0]
    return total_loss / len(X)


class MLP(nn.Module):
    def __init__(
        self,
        width,
        depth,
        activation=nn.Tanh,
        in_dim=1,
        out_dim=1,
        bias=True,
        linear=nn.Linear,
    ):
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
    train_loss = epoch(train_X, train_Y, model, loss_fun, opt)
    loss_history.append(train_loss)

v = torch.tensor(np.arange(0.0, 1.0, step=0.0001), dtype=dtype)


class HydroParam(nn.Module):
    def __init__(self, low, high, net):
        super().__init__()
        self.register_buffer("low", torch.tensor([low], dtype=dtype))
        self.register_buffer("high", torch.tensor([high], dtype=dtype))
        self.register_buffer("range", torch.tensor([high - low], dtype=dtype))

    def forward(self, x):
        x = self.net(x)
        return self.range * torch.sigmoid(x) + self.low


class ETTerm(nn.Module):
    def __init__(self, S0max, p):
        super.__init__()
        self.S0max = S0max
        self.p = p
        self.register_buffer("z", torch.tensor([0.0], dtype=dtype))

    def forward(self, x, S0, pet):
        S0max_val = self.S0max(x)
        p_val = self.p(x)
        et = p_val * pet * (S0 / S0max_val)
        et = torch.clamp(et, self.z, S0)
        return et


class DrainageTerm(nn.Module):
    def __init__(self, S0max, ku, c):
        super().__init__()
        self.S0max = S0max
        self.ku = ku
        self.c = c
        self.register_buffer("z", torch.tensor([0.0], dtype=dtype))

    def forward(self, x, S0):
        S0max_val = self.S0max(x)
        ku_val = self.ku(x)
        c_val = self.c(x)
        drainage = ku_val * (S0 / S0max_val) ** c_val
        drainage = torch.clamp(drainage, self.z, S0)
        return drainage


class SaturatedAreaTerm(nn.Module):
    def __init__(self, S0max, b):
        super().__init__()
        self.S0max = S0max
        self.b = b
        self.register_buffer("z", torch.tensor([0.0], dtype=dtype))

    def forward(self, x, S0):
        S0max_val = self.S0max(x)
        b_val = self.b(x)
        ratio = torch.clamp(S0 / S0max_val, self.z, self.z + 1)
        return 1 - (1 - ratio) ** b_val


class SurfaceFlowTerm(nn.Module):
    def __init__(self, a_sat):
        super().__init__()
        self.a_sat = a_sat
        self.register_buffer("z", torch.tensor([0.0], dtype=dtype))

    def forward(self, x, S0, prcp):
        return torch.clamp(self.a_sat(x, S0) * prcp, self.z)


class SubsurfaceFlowTerm(nn.Module):
    def __init__(self, S1max, ks, n):
        super().__init__()
        self.S1max = S1max
        self.ks = ks
        self.n = n
        self.z = torch.tensor(0.0, dtype=dtype)

    def forward(self, x, S1):
        ks_val = self.ks(x)
        S1max_val = self.S1max(x)
        n_val = self.n(x)
        subsurf_flow = ks_val * (S1 / S1max_val) ** n_val
        return torch.clamp(subsurf_flow, self.z, S1max_val)


class HydroEquation(nn.Module):
    def __init__(self, S0max, S1max, p, ku, ks, c, b, n):
        super().__init__()
        self.S0max = S0max
        self.S1max = S1max
        self.p = p
        self.ku = ku
        self.ks = ks
        self.c = c
        self.b = b
        self.n = n

        self.et_term = ETTerm(self.S0max, self.p)
        self.drainage_term = DrainageTerm(self.S0max, self.ku, self.c)
        self.saturated_area_term = SaturatedAreaTerm(self.S0max, self.b)
        self.surface_flow_term = SurfaceFlowTerm(self.saturated_area_term)
        self.subsurf_flow_term = SubsurfaceFlowTerm(self.S1max, self.ks, self.n)

    def forward(self, t, storage):
        S0, S1 = storage
        pet, prcp, *attrs = self.forcing
        x = torch.stack(attrs)
        self.et = self.et_term(x, S1, pet)
        self.drainage = self.drainage_term(x, S0)
        self.surface_flow = self.surface_flow_term(x, S0, prcp)
        self.subsurf_flow = self.subsurf_flow_term(x, S1)
        self.qtotal = self.surface_flow + self.subserf_flow

        dS0_dt = torch.clamp(
            prcp - self.et - self.drainage - self.surface_flow,
            min=-S0,
            max=self.S0max(x) - S0,
        )
        dS1_dt = torch.clamp(
            self.drainage - self.subsurf_flow, min=-S1, max=self.S1max(x) - S1
        )
        dS_dt = torch.hstack([dS0_dt, dS1_dt])
        return dS_dt


class HydroSimulator(nn.Module):
    def __init__(self, S0max, S1max, p, ku, ks, c, b, n, method="euler"):
        super().__init__()
        self.method = method
        self.register_buffer("t", torch.tensor([0, 1], dtype=dtype))
        self.ode = HydroEquation(S0max, S1max, p, ku, ks, c, b, n)

    def forward(self, forcing, storage):
        qtotal = []
        storage_ts = []
        surf_flow = []
        subsurf_flow = []
        drainage = []
        et = []
        for f in forcing:
            self.ode.forcing = f
            storage = odeint(self.ode, storage, self.t, method=self.method)
            storage = storage[-1].clone().detach()
            storage_ts.append(storage)
            qtotal.append(self.ode.qtotal)
            surf_flow.append(self.ode.surface_flow)
            subsurf_flow.append(self.ode.subsurf_flow)
            drainage.append(self.ode.drainage)
            et.append(self.ode.et)
        self.end_storage = storage
        self.storage_ts = torch.stack(storage_ts)
        self.surface_flow = torch.stack(surf_flow)
        self.subssurface_flow = torch.stack(subsurf_flow)
        self.drainge = torch.stack(drainage)
        self.et = torch.stack(et)
        return torch.stack(qtotal)


ds = xr.open_dataset(
    "camels_attrs_v2_streamflow_v1p2.nc/camels_attrs_v2_streamflow_v1p2.nc"
)
selected_basin = "11143000"
train_time = slice("10-01-1989", "09-30-1996")
test_time = slice("10-01-1997", "09-30-1999")
train_ds = ds.sel(station_id=selected_basin, time=train_time)
test_ds = ds.sel(station_id=selected_basin, time=test_time)
seq_len = 365
attrs = ["elevation", "area", "frac_forest", "aridity"]
in_vars = ["pet", "prcp"] + attrs
out_vars = ["QObs"]
