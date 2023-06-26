import torch
import torch.nn as nn
from torchdiffeq import odeint

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
        self.time = torch.tensor(0)