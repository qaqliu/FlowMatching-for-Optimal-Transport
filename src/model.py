import torch
import torch.nn as nn
from torch import Tensor
import torchdiffeq as tdeq
import numpy as np

class ODEFunc(nn.Module):
    def __init__(
        self, 
        device=torch.device('cpu'),
        data_dim=2,               # input data dimension 
        layer_config=[128, 128, 128], # list of number of nodes in each hidden layer
        activation='relu',         # activation function
        use_time=False,               # whether to use time as input
    ):
        super(ODEFunc, self).__init__()
        self.use_time = use_time
        self.device = device
        layers = []
        layer_config = (data_dim,) + tuple(layer_config) + (data_dim,) if not use_time else (data_dim + 1,) + tuple(layer_config) + (data_dim,)
        for i, (in_dim, out_dim) in enumerate(zip(layer_config[:-1], layer_config[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            if i < len(layer_config) - 2:
                if activation == 'relu':#softplus
                    layers.append(nn.ReLU())
                elif activation == 'softplus':
                    layers.append(nn.Softplus())
                else:
                    raise ValueError(f'Do not support activation layer {activation}')
        self.layers = nn.Sequential(*layers).to(device)
                
    def forward(self, x, t):
        if self.use_time:
            t_in = t * torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
            x = torch.cat([x, t_in], dim=1)
        x = self.layers(x)
        return x
    
class Flow_Matching(nn.Module):
    def __init__(
        self,
        odefunc: ODEFunc,
        device,
        h_k=1,
        h_steps=3,
        sigma=0.02,
        use_fd=True,
        ode_solver='rk4',
        n_eps=1,
    ):
        super(Flow_Matching, self).__init__()
        self.model = odefunc
        self.device = device
        self.h_k = h_k
        self.h_steps = h_steps
        self.sigma = sigma
        self.use_fd = use_fd
        self.ode_solver = ode_solver
        self.n_eps = n_eps
        
    def forward(self, x, t=0.0):
        return self.model(x, t)
    
    def _fd_div_at(self, x, t, f_m=None):
        """
        Finite difference hutchinson divergence at(x,t). Returns (B,1)
        f_m is if already evaluated f(x, t)
        """
        with torch.no_grad():
            rms = x.pow(2).mean().sqrt().clamp_min(1e-3)
        sigma_base = self.sigma if self.sigma > 0 else 1e-2
        sigma = torch.maximum(
            torch.tensor(1e-4, device=x.device, dtype=x.dtype), rms * sigma_base
        )

        acc = 0.0
        self._eps_cache = [torch.randn_like(x) for _ in range(self.n_eps)]
        for eps in self._eps_cache:
            f_p = self.model(x + sigma * eps, t)  # (B, D)
            if f_m is None:
                f_m = self.model(x, t)

            jvp = (f_p - f_m) / (sigma)  # J_f @ e
            acc += (jvp * eps).sum(dim=-1)
        return acc / len(self._eps_cache)
    
    def _rhs(self, t, state):
        """torchdiffeq expects (t,state) where state = (x, logdet, jacint)"""
        x, logdet, _ = state
        vfield = self.model(x, t)
        
        if self.use_fd:
            div = self._fd_div_at(x, t, f_m=vfield)
        else:
            raise NotImplementedError("Only FD divergence is implemented.")

        dlogdet_dt = -div
        djac_dt = torch.zeros_like(logdet)
        return (vfield, dlogdet_dt, djac_dt)
    
    def push_forward(self, x0, t=0.0, reverse=False, full_traj=False):
        t_start, t_end = float(t), float(t) + float(self.h_k)
        t_grid = torch.linspace(
            t_start, t_end, self.h_steps + 1, dtype=x0.dtype, device=self.device
        )
        if reverse:
            t_grid = t_grid.flip(0)
        B, D = x0.shape
        logdet0 = torch.zeros(B, device=self.device, dtype=x0.dtype)
        jac0 = torch.zeros(B, device=self.device, dtype=x0.dtype)

        xT, logdetT, jacT = tdeq.odeint(
            self._rhs, (x0, logdet0, jac0), t_grid, method=self.ode_solver
        )

        if full_traj:
            return xT, logdetT, jacT
        return xT[-1], logdetT[-1], jacT[-1]

