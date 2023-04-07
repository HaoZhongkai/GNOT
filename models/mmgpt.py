#!/usr/bin/env python
#-*- coding:utf-8 _*-
import math
import numpy as np
import torch
import torch.nn as nn
import dgl
from einops import repeat, rearrange
from torch.nn import functional as F
from torch.nn import GELU, ReLU, Tanh, Sigmoid
from torch.nn.utils.rnn import pad_sequence

from gno_transformer.utils import MultipleTensors
from gno_transformer.models.mlp import MLP




class MoEGPTConfig():
    """ base GPT config, params common to all GPT versions """
    def __init__(self,attn_type='linear', embd_pdrop=0.0, resid_pdrop=0.0,attn_pdrop=0.0, n_embd=128, n_head=1, n_layer=3, block_size=128, n_inner=4,act='gelu',n_experts=2,space_dim=1,branch_sizes=None,n_inputs=1):
        self.attn_type = attn_type
        self.embd_pdrop = embd_pdrop
        self.resid_pdrop = resid_pdrop
        self.attn_pdrop = attn_pdrop
        self.n_embd = n_embd  # 64
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size
        self.n_inner = n_inner * self.n_embd
        self.act = act
        self.n_experts = n_experts
        self.space_dim = space_dim
        self.branch_sizes = branch_sizes
        self.n_inputs = n_inputs






class LinearAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super(LinearAttention, self).__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head

        self.attn_type = 'l1'

    '''
        Linear Attention and Linear Cross Attention (if y is provided)
    '''
    def forward(self, x, y=None, layer_past=None):
        y = x if y is None else y
        B, T1, C = x.size()
        _, T2, _ = y.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.query(x).view(B, T1, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = self.key(y).view(B, T2, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(y).view(B, T2, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)


        if self.attn_type == 'l1':
            q = q.softmax(dim=-1)
            k = k.softmax(dim=-1)   #
            k_cumsum = k.sum(dim=-2, keepdim=True)
            D_inv = 1. / (q * k_cumsum).sum(dim=-1, keepdim=True)       # normalized
        elif self.attn_type == "galerkin":
            q = q.softmax(dim=-1)
            k = k.softmax(dim=-1)  #
            D_inv = 1. / T2                                           # galerkin
        elif self.attn_type == "l2":                                   # still use l1 normalization
            q = q / q.norm(dim=-1,keepdim=True, p=1)
            k = k / k.norm(dim=-1,keepdim=True, p=1)
            k_cumsum = k.sum(dim=-2, keepdim=True)
            D_inv = 1. / (q * k_cumsum).abs().sum(dim=-1, keepdim=True)  # normalized
        else:
            raise NotImplementedError

        context = k.transpose(-2, -1) @ v
        y = self.attn_drop((q @ context) * D_inv + q)

        # output projection
        y = rearrange(y, 'b h n d -> b n (h d)')
        y = self.proj(y)
        return y



class LinearCrossAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super(LinearCrossAttention, self).__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.keys = nn.ModuleList([nn.Linear(config.n_embd, config.n_embd) for _ in range(config.n_inputs)])
        self.values = nn.ModuleList([nn.Linear(config.n_embd, config.n_embd) for _ in range(config.n_inputs)])
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.n_inputs = config.n_inputs

        self.attn_type = 'l1'

    '''
        Linear Attention and Linear Cross Attention (if y is provided)
    '''
    def forward(self, x, y=None, layer_past=None):
        y = x if y is None else y
        B, T1, C = x.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.query(x).view(B, T1, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.softmax(dim=-1)
        out = q
        for i in range(self.n_inputs):
            _, T2, _ = y[i].size()
            k = self.keys[i](y[i]).view(B, T2, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
            v = self.values[i](y[i]).view(B, T2, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
            k = k.softmax(dim=-1)  #
            k_cumsum = k.sum(dim=-2, keepdim=True)
            D_inv = 1. / (q * k_cumsum).sum(dim=-1, keepdim=True)  # normalized
            out = out +  1 * (q @ (k.transpose(-2, -1) @ v)) * D_inv


        # output projection
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj(out)
        return out


'''
    X: N*T*C --> N*(4*n + 3)*C 
'''
def horizontal_fourier_embedding(X, n=3):
    freqs = 2**torch.linspace(-n, n, 2*n+1).to(X.device)
    freqs = freqs[None,None,None,...]
    X_ = X.unsqueeze(-1).repeat([1,1,1,2*n+1])
    X_cos = torch.cos(freqs * X_)
    X_sin = torch.sin(freqs * X_)
    X = torch.cat([X.unsqueeze(-1), X_cos, X_sin],dim=-1).view(X.shape[0],X.shape[1],-1)
    return X





'''
    Self and Cross Attention block for CGPT, contains  a cross attention block and a self attention block
'''
class MIOECrossAttentionBlock(nn.Module):
    def __init__(self, config):
        super(MIOECrossAttentionBlock, self).__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2_branch = nn.ModuleList([nn.LayerNorm(config.n_embd) for _ in range(config.n_inputs)])
        self.ln3 = nn.LayerNorm(config.n_embd)
        self.ln4 = nn.LayerNorm(config.n_embd)
        self.ln5 = nn.LayerNorm(config.n_embd)
        if config.attn_type == 'linear':
            print('Using Linear Attention')
            self.selfattn = LinearAttention(config)
            self.crossattn = LinearCrossAttention(config)
        else:
            raise NotImplementedError

        if config.act == 'gelu':
            self.act = GELU
        elif config.act == "tanh":
            self.act = Tanh
        elif config.act == 'relu':
            self.act = ReLU
        elif config.act == 'sigmoid':
            self.act = Sigmoid

        self.resid_drop1 = nn.Dropout(config.resid_pdrop)
        self.resid_drop2 = nn.Dropout(config.resid_pdrop)

        self.n_experts = config.n_experts
        self.n_inputs = config.n_inputs

        self.moe_mlp1 = nn.ModuleList([nn.Sequential(
            nn.Linear(config.n_embd, config.n_inner),
            self.act(),
            nn.Linear(config.n_inner, config.n_embd),
        ) for _ in range(self.n_experts)])

        self.moe_mlp2 = nn.ModuleList([nn.Sequential(
            nn.Linear(config.n_embd, config.n_inner),
            self.act(),
            nn.Linear(config.n_inner, config.n_embd),
        ) for _ in range(self.n_experts)])

        self.gatenet = nn.Sequential(
            nn.Linear(config.space_dim, config.n_inner),
            self.act(),
            nn.Linear(config.n_inner, config.n_inner),
            self.act(),
            nn.Linear(config.n_inner, self.n_experts)
        )


    def ln_branchs(self, y):
        return MultipleTensors([self.ln2_branch[i](y[i]) for i in range(self.n_inputs)])

    '''
        x: [B, T1, C], y:[B, T2, C], pos:[B, T1, n]
    '''
    def forward(self, x, y, pos):
        gate_score = F.softmax(self.gatenet(pos),dim=-1).unsqueeze(2)    # B, T1, 1, m
        x = x + self.resid_drop1(self.crossattn(self.ln1(x), self.ln_branchs(y)))
        x_moe1 = torch.stack([self.moe_mlp1[i](x) for i in range(self.n_experts)],dim=-1) # B, T1, C, m
        x_moe1 = (gate_score*x_moe1).sum(dim=-1,keepdim=False)
        x = x + self.ln3(x_moe1)
        x = x + self.resid_drop2(self.selfattn(self.ln4(x)))
        x_moe2 = torch.stack([self.moe_mlp2[i](x) for i in range(self.n_experts)],dim=-1) # B, T1, C, m
        x_moe2 = (gate_score*x_moe2).sum(dim=-1,keepdim=False)
        x = x + self.ln5(x_moe2)
        return x

    #### No layernorm
    # def forward(self, x, y):
    #     # y = self.selfattn_branch(self.ln5(y))
    #     x = x + self.resid_drop1(self.crossattn(x, y))
    #     x = x + self.mlp1(x)
    #     x = x + self.resid_drop2(self.selfattn(x))
    #     x = x + self.mlp2(x)
    #
    #     return x





'''
    Cross Attention GPT neural operator
    Trunck Net: geom
'''
class GNOT(nn.Module):
    def __init__(self,
                 trunk_size=2,
                 branch_sizes=None,
                 space_dim=2,
                 output_size=3,
                 n_layers=2,
                 n_hidden=64,
                 n_head=1,
                 n_experts = 2,
                 n_inner = 4,
                 mlp_layers=2,
                 attn_type='linear',
                 act = 'gelu',
                 ffn_dropout=0.0,
                 attn_dropout=0.0,
                 horiz_fourier_dim = 0,
                 ):
        super(GNOT, self).__init__()

        self.horiz_fourier_dim = horiz_fourier_dim
        self.trunk_size = trunk_size * (4*horiz_fourier_dim + 3) if horiz_fourier_dim>0 else trunk_size
        self.branch_sizes = [bsize * (4*horiz_fourier_dim + 3) for bsize in branch_sizes] if horiz_fourier_dim > 0 else branch_sizes
        self.n_inputs = len(self.branch_sizes)
        self.output_size = output_size
        self.space_dim = space_dim
        self.gpt_config = MoEGPTConfig(attn_type=attn_type,embd_pdrop=ffn_dropout, resid_pdrop=ffn_dropout, attn_pdrop=attn_dropout,n_embd=n_hidden, n_head=n_head, n_layer=n_layers,
                                       block_size=128,act=act, n_experts=n_experts,space_dim=space_dim, branch_sizes=branch_sizes,n_inputs=len(branch_sizes),n_inner=n_inner)

        self.trunk_mlp = MLP(self.trunk_size, n_hidden, n_hidden, n_layers=mlp_layers,act=act)
        self.branch_mlps = nn.ModuleList([MLP(bsize, n_hidden, n_hidden, n_layers=mlp_layers,act=act) for bsize in self.branch_sizes])


        self.blocks = nn.Sequential(*[MIOECrossAttentionBlock(self.gpt_config) for _ in range(self.gpt_config.n_layer)])

        self.out_mlp = MLP(n_hidden, n_hidden, output_size, n_layers=mlp_layers)

        # self.apply(self._init_weights)

        self.__name__ = 'MIOEGPT'



    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.0002)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)



    def forward(self, g, u_p, inputs):
        gs = dgl.unbatch(g)
        x = pad_sequence([_g.ndata['x'] for _g in gs]).permute(1, 0, 2)  # B, T1, F

        pos = x[:,:,0:self.space_dim]


        x = torch.cat([x, u_p.unsqueeze(1).repeat([1, x.shape[1], 1])], dim=-1)

        # if self.horiz_fourier_dim > 0:
        #     x = horizontal_fourier_embedding(x, self.horiz_fourier_dim)
        #     z = horizontal_fourier_embedding(z, self.horiz_fourier_dim)

        x = self.trunk_mlp(x)
        z = MultipleTensors([self.branch_mlps[i](inputs[i]) for i in range(self.n_inputs)])

        for block in self.blocks:
            x = block(x, z, pos)
        x = self.out_mlp(x)

        x_out = torch.cat([x[i, :num] for i, num in enumerate(g.batch_num_nodes())],dim=0)
        return x_out