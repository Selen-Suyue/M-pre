import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

USE_MAMBA = 1
DIFFERENT_H_STATES_RECURRENT_UPDATE_MECHANISM = 0

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.projector = nn.Linear(configs.pred_len, configs.pred_len, bias=True)
        self.mamba = Mamba(configs.batch_size, configs.feature_dim, configs.pred_len, configs.pred_len, 
                           device='cuda' if torch.cuda.is_available() else 'cpu')

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        _, _, N = x_enc.shape
        x = torch.cat([x_enc, x_mark_enc], dim=2)
        x = x.permute(0, 2, 1)
        mamba_out = self.mamba(x)
        dec = self.projector(mamba_out)
        dec = dec.permute(0, 2, 1)
        dec_out = dec[:, :, :N]
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return out[:, -self.pred_len:, :]

class S6(nn.Module):
    def __init__(self,bs, seq_len, d_model, state_size, device):
        super(S6, self).__init__()
        self.fc1 = nn.Linear(d_model, d_model, device=device)
        self.fc2 = nn.Linear(d_model, state_size, device=device)
        self.fc3 = nn.Linear(d_model, state_size, device=device)
        self.seq_len = seq_len
        self.d_model = d_model
        self.state_size = state_size
        self.A = nn.Parameter(F.normalize(torch.ones(d_model, state_size, device=device), p=2, dim=-1))
        nn.init.xavier_uniform_(self.A)
        self.B = torch.zeros(bs, self.seq_len, self.state_size, device=device)
        self.C = torch.zeros(bs, self.seq_len, self.state_size, device=device)
        self.delta = torch.zeros(bs, self.seq_len, self.d_model, device=device)
        self.dA = torch.zeros(bs, self.seq_len, self.d_model, self.state_size, device=device)
        self.dB = torch.zeros(bs, self.seq_len, self.d_model, self.state_size, device=device)
        self.h = torch.zeros(bs, self.seq_len, self.d_model, self.state_size, device=device)
        self.y = torch.zeros(bs, self.seq_len, self.d_model, device=device)

    def discretization(self):
        self.dB = torch.einsum("bld,bln->bldn", self.delta, self.B)
        self.dA = torch.exp(torch.einsum("bld,dn->bldn", self.delta, self.A))
        return self.dA, self.dB

    def forward(self, x):
        self.B = self.fc2(x)
        self.C = self.fc3(x)
        self.delta = F.softplus(self.fc1(x))
        self.discretization()
        if DIFFERENT_H_STATES_RECURRENT_UPDATE_MECHANISM:
            global current_batch_size
            current_batch_size = x.shape[0]
            if self.h.shape[0] != current_batch_size:
                h_new = torch.einsum('bldn,bldn->bldn', self.dA, self.h[:current_batch_size, ...]) + rearrange(x, "b l d -> b l d 1") * self.dB
            else:
                h_new = torch.einsum('bldn,bldn->bldn', self.dA, self.h) + rearrange(x, "b l d -> b l d 1") * self.dB
            self.y = torch.einsum('bln,bldn->bld', self.C, h_new)
            global temp_buffer
            temp_buffer = h_new.detach().clone() if not self.h.requires_grad else h_new.clone()
            return self.y
        else:
            h = torch.zeros(x.size(0), self.seq_len, self.d_model, self.state_size, device=x.device)
            h = torch.einsum('bldn,bldn->bldn', self.dA, h) + rearrange(x, "b l d -> b l d 1") * self.dB
            y = torch.einsum('bln,bldn->bld', self.C, h)
            return y
        
class MambaBlock(nn.Module):
    def __init__(self, bs, seq_len, d_model, state_size, device):
        super(MambaBlock, self).__init__()
        self.inp_proj = nn.Linear(d_model, 2*d_model, device=device)
        self.out_proj = nn.Linear(2*d_model, d_model, device=device)
        self.D = nn.Linear(d_model, 2*d_model, device=device)
        self.out_proj.bias._no_weight_decay = True
        nn.init.constant_(self.out_proj.bias, 1.0)
        self.S6 = S6(bs, seq_len, 2*d_model, state_size, device)
        self.conv = nn.Conv1d(seq_len, seq_len, kernel_size=3, padding=1, device=device)
        self.conv_linear = nn.Linear(2*d_model, 2*d_model, device=device)
        self.norm = RMSNorm(d_model, device=device)

    def forward(self, x):
        x = self.norm(x)
        x_proj = self.inp_proj(x)
        x_conv = self.conv(x_proj)
        x_conv_act = F.silu(x_conv)
        x_conv_out = self.conv_linear(x_conv_act)
        x_ssm = self.S6(x_conv_out)
        x_act = F.silu(x_ssm)
        x_residual = F.silu(self.D(x))
        x_combined = x_act * x_residual
        x_out = self.out_proj(x_combined)
        return x_out

class Mamba(nn.Module):
    def __init__(self, bs, seq_len, d_model, state_size, device):
        super(Mamba, self).__init__()
        self.mamba_block1 = MambaBlock(bs, seq_len, d_model, state_size, device)
        self.mamba_block2 = MambaBlock(bs, seq_len, d_model, state_size, device)
        self.mamba_block3 = MambaBlock(bs, seq_len, d_model, state_size, device)

    def forward(self, x):
        x = self.mamba_block1(x)
        x = self.mamba_block2(x)
        x = self.mamba_block3(x)
        return x

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: str = 'cuda'):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight