
from __future__ import annotations
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from layers.RevIN import RevIN
from layers.Embed import PatchEmbedding


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.config = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.revin = RevIN(configs.enc_in)
        self.k = configs.top_k
        self.n_layer = configs.n_layer
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.period_list = configs.period_list

        self.patch_embedding_list = nn.ModuleList([
            PatchEmbedding(
                configs.d_model,
                configs.period_list[i],
                int(configs.period_list[i] / 2),
                int(configs.period_list[i] / 2),
                configs.dropout
            )
            for i in range(configs.top_k)
        ])

        self.layers = nn.ModuleList([ParticleBlock(configs, configs.period_list[i]) for i in range(configs.top_k)])

        # Prediction Head
        self.head_nf_list = [(configs.d_model * \
                       int((configs.seq_len - configs.period_list[i]) / int(configs.period_list[i] / 2) + 2))
                             for i in range(configs.top_k)]
        self.head_list = nn.ModuleList([
            Flatten_Head(
                configs.enc_in,
                self.head_nf_list[i],
                configs.pred_len,
                head_dropout=configs.dropout
            )
            for i in range(configs.top_k)
        ])


        self.channel = nn.Sequential(
            nn.Linear(configs.enc_in, configs.d_model),
            nn.GELU(),
            nn.Linear(configs.d_model, configs.enc_in),
            nn.Dropout(configs.dropout)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):

        # Normalization
        x_enc = self.revin(x_enc, 'norm')

        x_old = x_enc
        final_output = None

        """Multi-scale patching"""
        for i in range(self.k):
            x_enc = x_old


            x_enc = x_enc.permute(0, 2, 1)

            x_enc, n_vars = self.patch_embedding_list[i](x_enc)

            for _ in range(self.n_layer):
                x_enc = self.layers[i](x_enc)   # Quantum Mamba

            x_enc = torch.reshape(
                x_enc, (-1, n_vars, x_enc.shape[-2], x_enc.shape[-1]))
            x_enc = x_enc.permute(0, 1, 3, 2)

            x_enc = self.head_list[i](x_enc)

            x_enc = x_enc.permute(0, 2, 1)

            x_enc = x_enc + self.channel(x_enc)

            if final_output is None:
                final_output = self.period_list[i] * x_enc
            else:
                final_output = self.period_list[i] * x_enc + final_output

        out = final_output / sum(self.period_list[:self.k])

        # De-Normalization
        out = self.revin(out, 'denorm')

        return out


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        potential_energy = moving_mean
        return potential_energy


class QuantumSeries(nn.Module):
    def __init__(self, mass, kernel_size, dt):
        super(QuantumSeries, self).__init__()
        self.decomposition = series_decomp(kernel_size)
        self.dt = dt
        self.mass = mass
        self.hbar = 1.0  # Planck's constant (reduced)

    def euler_step(self, psi, dt, hamiltonian_func):
        return psi + dt * hamiltonian_func(psi)

    def apply_hamiltonian(self, psi, kinetic, potential_energy):
        return -1j * (kinetic + potential_energy) * psi

    def forward(self, psi):
        (b, l, d_in) = psi.shape  # b: batch size, l: sequence length, d_in: feature size
        steps = l


        potential_energy = self.decomposition(psi)
        kinetic = -0.5 * (self.hbar ** 2 / self.mass) * (
                    torch.roll(psi, 1, dims=-1) - 2 * psi + torch.roll(psi, -1, dims=-1))

        # Euler
        for _ in range(steps):
            psi = self.euler_step(psi, self.dt, lambda psi: self.apply_hamiltonian(psi, kinetic, potential_energy))

        # 计算概率密度
        probability_density = torch.abs(psi) ** 2
        probability_density = F.softmax(probability_density, dim=-1)

        return probability_density


class Flatten_Head(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class ParticleBlock(nn.Module):
    def __init__(self, configs, mass):
        super().__init__()

        self.mixer = MambaBlock(configs, mass)
        self.norm = RMSNorm(configs.d_model)

    def forward(self, x):
        output = self.mixer(self.norm(x)) + x
        return output


class MambaBlock(nn.Module):
    def __init__(self, configs, mass):
        super().__init__()

        self.d_inner = configs.expand * configs.d_model
        self.dt_rank = math.ceil(configs.d_model / 16)

        self.in_proj = nn.Linear(configs.d_model, self.d_inner * 2, bias=configs.bias)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=configs.conv_bias,
            kernel_size=configs.d_conv,
            groups=self.d_inner,
            padding=configs.d_conv - 1,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + configs.d_state, bias=False)

        self.C_proj = nn.Linear(self.d_inner, configs.d_state, bias=False)

        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = repeat(torch.arange(1, configs.d_state + 1), 'n -> d n', d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))

        self.out_proj = nn.Linear(self.d_inner, configs.d_model, bias=configs.bias)

        kernel_size = 25
        self.C_schro = QuantumSeries(mass, kernel_size, 1)


    def forward(self, x):
        (b, l, d) = x.shape

        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        
        x = F.silu(x)

        y = self.ssm(x)
        
        y = y * F.silu(res)
        
        output = self.out_proj(y)

        return output

    def ssm(self, x):
        (d_in, n) = self.A_log.shape

        A = -torch.exp(self.A_log.float())  # shape (d_in, n)

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        (delta, B) = x_dbl.split(split_size=[self.dt_rank, n], dim=-1)

        C = self.C_schro(x)
        C = self.C_proj(C)

        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)

        y = self.selective_scan(x, delta, A, B, C)
        
        return y

    def selective_scan(self, u, dt, A, B, C):
        dA = torch.einsum('bld,dn->bldn', dt, A)
        dB_u = torch.einsum('bld,bld,bln->bldn', dt, u, B)
        dA_cumsum = F.pad(dA[:, 1:], (0, 0, 0, 0, 0, 1)).flip(1).cumsum(1).exp().flip(1)
        x = dB_u * dA_cumsum
        x = x.cumsum(1) / (dA_cumsum + 1e-12)
        y = torch.einsum('bldn,bln->bld', x, C)
        return y


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output
