import torch
import numpy as np
import xarray as xr
import torch.nn as nn
from tqdm.auto import tqdm
from torchdiffeq import odeint
from torch.utils.data import Dataset


max_epochs = 300
width = 12
depth = 4
learning_rate = 3e-3

dtype = torch.float32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
        super().__init__()
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


class MultipleTrajectoryDataset(Dataset):
    def __init__(self, ds, in_vars, out_vars, trajectory_len):
        super().__init__()
        self.ds = ds.load().drop("station_id")
        self.in_vars = in_vars
        self.out_vars = out_vars
        self.trajectory_len = trajectory_len
        self.n_trajectories = int(len(self.ds["time"]) / self.trajectory_len)
        self.time_starts = [i * trajectory_len for i in range(self.n_trajectories)]
        self.time_ends = [(i + 1) * trajectory_len for i in range(self.n_trajectories)]

    def __getitem__(self, idx):
        time_slice = slice(self.time_starts[idx], self.time_ends[idx])
        sample_ds = self.ds.isel(time=time_slice)
        x = torch.from_numpy(sample_ds[self.in_vars].to_dataframe().values)
        y = torch.from_numpy(sample_ds[self.out_vars].to_dataframe().values)
        return x, y

    def __len__(self):
        return len(self.time_starts)


def update_model_step(
    model, opt, train_data, S_init, loss_fun=torch.nn.MSELoss(), device=device
):
    Xd, yd = train_data
    Xd = Xd.to(device)
    yd = yd.to(device)
    opt.zero_grad()
    yp = model(Xd, storage=S_init)
    loss = loss_fun(yp.squeeze(), yd.squeeze())
    loss.backward()
    opt.step()
    return loss


def update_ic_step(model, train_data, S_init):
    forcing, q_true = train_data
    q_pred = model(forcing, storage=S_init)
    final_storage = model.end_storage.close().detach()
    return final_storage


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

train_data = MultipleTrajectoryDataset(train_ds, in_vars, out_vars, seq_len)

test_data = MultipleTrajectoryDataset(test_ds, in_vars, out_vars, len(test_ds["time"]))

initial_storage = torch.tensor([30.0, 50.0], dtype=torch.float32)

width = 6
depth = 1
in_dim = len(attrs)

S0max = HydroParam(50.000, 200.0, MLP(width, depth, in_dim=in_dim))
S1max = HydroParam(100.000, 500.0, MLP(width, depth, in_dim=in_dim))
p = HydroParam(0.001, 1.5, MLP(width, depth, in_dim=in_dim))
ku = HydroParam(0.010, 100.0, MLP(width, depth, in_dim=in_dim))
ks = HydroParam(0.010, 100.0, MLP(width, depth, in_dim=in_dim))
b = HydroParam(0.001, 3.0, MLP(width, depth, in_dim=in_dim))
c = HydroParam(0.010, 10.0, MLP(width, depth, in_dim=in_dim))
n = HydroParam(0.010, 10.0, MLP(width, depth, in_dim=in_dim))

model = HydroSimulator(S0max, S1max, p, ku, ks, b, c, n).to(device)
# model.to(device)

learning_rate = 3e-3
opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fun = torch.nn.MSELoss()

max_epochs = 6
max_sub_epochs = 2
train_loss_history = {i: [] for i in range(len(train_data))}
for epoch in tqdm(range(max_epochs)):
    storage = initial_storage.clone().to(device)
    for idx_traj in np.arange(len(train_data)):
        for sub_epoch in range(max_sub_epochs):
            data = train_data[idx_traj]
            l = update_model_step(model, opt, data, storage.close())
            train_loss_history[idx_traj].append(l.detach().cpu().numpy())
        storage = update_ic_step(model, data, storage.clone())
