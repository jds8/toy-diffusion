# provenance: https://github.com/jannerm/diffuser/blob/main/diffuser/models/diffusion.py

import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import pdb

from models.toy_helpers import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
    Residual,
    PreNorm,
    LinearAttention,
    Attention,
)


class ResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, embed_dim, traj_length, kernel_size=5):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )

        self.cond_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )

        self.bool_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t, cemb, bool_emb):
        '''
            x : [ batch_size x inp_channels x traj_length ]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x traj_length ]
        '''
        out = self.blocks[0](x) + self.time_mlp(t) + self.cond_mlp(cemb) + self.bool_mlp(bool_emb)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class TemporalUnet(nn.Module):

    def __init__(
        self,
        traj_length,
        d_model,
        cond_dim,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        attention=False,
        device=None,
    ):
        super().__init__()

        dims = [d_model, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        self.traj_length = traj_length
        self.d_model = d_model

        time_dim = dim
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.cond_dim = cond_dim
        self.cond_mlp = nn.Sequential(
            nn.Linear(self.cond_dim, dim),
            nn.Mish(),
            nn.Linear(dim, dim),
        )

        self.bool_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim),
            nn.Mish(),
            nn.Linear(dim, dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim, traj_length=traj_length),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim, traj_length=traj_length),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))) if attention else nn.Identity(),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                traj_length = traj_length // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, traj_length=traj_length)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim))) if attention else nn.Identity()
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, traj_length=traj_length)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim, traj_length=traj_length),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim, traj_length=traj_length),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))) if attention else nn.Identity(),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

            if not is_last:
                traj_length = traj_length * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=5),
            nn.Conv1d(dim, d_model, 1),
        )

    def forward(self, x, time, cond):
        '''
            x : [ batch x traj_length x transition ]
        '''

        b_dim, h_dim, t_dim = x.shape
        x = einops.rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)
        cond = cond if cond is not None else torch.ones(t.shape[0], 1, device=x.device) * -1
        cemb = self.cond_mlp(cond).reshape(cond.shape[0], -1)

        use_cond = (cond > 0.).to(torch.float).reshape(cond.shape)
        bool_emb = self.bool_mlp(use_cond).reshape(cemb.shape)
        h = []

        for idx, (resnet, resnet2, attn, downsample) in enumerate(self.downs):
            x = resnet(x, t, cemb, bool_emb)
            x = resnet2(x, t, cemb, bool_emb)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t, cemb, bool_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, cemb, bool_emb)

        for idx, (resnet, resnet2, attn, upsample) in enumerate(self.ups):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t, cemb, bool_emb)
            x = resnet2(x, t, cemb, bool_emb)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')
        return x[:b_dim, :h_dim, :t_dim]


class TemporalRegressor(nn.Module):

    def __init__(
        self,
        traj_length,
        d_model,
        cond_dim,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        attention=False,
        device=None,
    ):
        super().__init__()

        dims = [d_model, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        self.traj_length = traj_length
        self.d_model = d_model

        time_dim = dim
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.cond_dim = cond_dim
        self.cond_mlp = nn.Sequential(
            nn.Linear(self.cond_dim, dim),
            nn.Mish(),
            nn.Linear(dim, dim),
        )

        self.bool_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim),
            nn.Mish(),
            nn.Linear(dim, dim),
        )

        self.downs = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim, traj_length=traj_length),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim, traj_length=traj_length),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))) if attention else nn.Identity(),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                traj_length = traj_length // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, traj_length=traj_length)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim))) if attention else nn.Identity()
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, traj_length=traj_length)

    def forward(self, x, time, cond):
        '''
            x : [ batch x traj_length x transition ]
        '''

        b_dim, h_dim, t_dim = x.shape
        x = einops.rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)
        cond = cond if cond is not None else torch.ones(t.shape[0], 1, device=x.device) * -1
        cemb = self.cond_mlp(cond).reshape(cond.shape[0], -1)

        use_cond = (cond > 0.).to(torch.float).reshape(cond.shape)
        bool_emb = self.bool_mlp(use_cond).reshape(cemb.shape)
        h = []

        for idx, (resnet, resnet2, attn, downsample) in enumerate(self.downs):
            x = resnet(x, t, cemb, bool_emb)
            x = resnet2(x, t, cemb, bool_emb)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t, cemb, bool_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, cemb, bool_emb)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')
        return y


class TemporalTransformerUnet(nn.Module):

    def __init__(
        self,
        traj_length,
        d_model,
        cond_dim,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        attention=True,
        device=None,
    ):
        super().__init__()

        dims = [d_model, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        self.traj_length = traj_length
        self.d_model = d_model
        self.attention = attention

        time_dim = dim
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.cond_dim = cond_dim
        self.cond_mlp = nn.Sequential(
            nn.Linear(self.cond_dim, dim),
            nn.Mish(),
            nn.Linear(dim, dim),
        )

        self.bool_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim),
            nn.Mish(),
            nn.Linear(dim, dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim, traj_length=traj_length),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim, traj_length=traj_length),
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim, traj_length=traj_length),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim, traj_length=traj_length),
                Residual(PreNorm(dim_out, Attention(dim_out, permute=True))) if attention else nn.Identity(),
                Downsample1d(dim_out) if not is_last else nn.Identity(),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                traj_length = traj_length // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, traj_length=traj_length)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, traj_length=traj_length)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, permute=True))) if attention else nn.Identity()
        self.mid_block3 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, traj_length=traj_length)
        self.mid_block4 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, traj_length=traj_length)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim, traj_length=traj_length),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim, traj_length=traj_length),
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim, traj_length=traj_length),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim, traj_length=traj_length),
                Residual(PreNorm(dim_in, Attention(dim_in, permute=True))) if attention else nn.Identity(),
                Upsample1d(dim_in) if not is_last else nn.Identity(),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

            if not is_last:
                traj_length = traj_length * 2

        self.final_conv1 = Conv1dBlock(dim, dim, kernel_size=5)

        self.final_attn = Residual(PreNorm(dim, Attention(dim, permute=True)))

        self.final_conv2 = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=5),
            nn.Conv1d(dim, d_model, 1),
        )


    def forward(self, x, time, cond_traj, cond):
        '''
            x : [ batch x traj_length x transition ]
        '''

        cond_traj = cond_traj if cond_traj is not None else torch.zeros_like(x)
        b_dim, h_dim, t_dim = x.shape
        x = einops.rearrange(x, 'b h t -> b t h')
        cond_traj = einops.rearrange(cond_traj, 'b h t -> b t h')

        t = self.time_mlp(time)
        cond = cond if cond is not None else torch.ones(t.shape[0], 1) * -1
        cemb = self.cond_mlp(cond).reshape(cond.shape[0], -1)

        use_cond = (cond > 0.).to(torch.float).reshape(cond.shape)
        bool_emb = self.bool_mlp(use_cond).reshape(cemb.shape)

        h = []
        h_cond = []

        for resnet, resnet2, resnet3, resnet4, attn, downsample, downsample2 in self.downs:
            x = resnet(x, t, cemb, bool_emb)
            x = resnet2(x, t, cemb, bool_emb)
            cond_traj = resnet3(cond_traj, t, cemb, bool_emb)
            cond_traj = resnet4(cond_traj, t, cemb, bool_emb)
            x = attn(x, cond_traj)
            h.append(x)
            h_cond.append(cond_traj)
            x = downsample(x)
            cond_traj = downsample2(cond_traj)

        x = self.mid_block1(x, t, cemb, bool_emb)
        cond_traj = self.mid_block2(cond_traj, t, cemb, bool_emb)
        x = self.mid_attn(x, cond_traj)
        x = self.mid_block3(x, t, cemb, bool_emb)
        cond_traj = self.mid_block4(cond_traj, t, cemb, bool_emb)

        for idx, (resnet, resnet2, resnet3, resnet4, attn, upsample, upsample2) in enumerate(self.ups):
            x = torch.cat((x, h.pop()), dim=1)
            cond_traj = torch.cat((cond_traj, h_cond.pop()), dim=1)
            x = resnet(x, t, cemb, bool_emb)
            x = resnet2(x, t, cemb, bool_emb)
            cond_traj = resnet3(cond_traj, t, cemb, bool_emb)
            cond_traj = resnet4(cond_traj, t, cemb, bool_emb)
            x = attn(x, cond_traj)
            x = upsample(x)
            cond_traj = upsample2(cond_traj)

        cond_traj = self.final_conv1(cond_traj)
        x = self.final_attn(x, cond_traj)
        x = self.final_conv2(x)

        x = einops.rearrange(x, 'b t h -> b h t')
        return x[:b_dim, :h_dim, :t_dim]
