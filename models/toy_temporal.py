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


class ResidualBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, embed_dim, kernel_size=5):
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

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        '''
            x : [ batch_size x inp_channels x traj_length ]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x traj_length ]
        '''
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class ResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, embed_dim, kernel_size=5):
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
        try:
            out = self.blocks[0](x) + self.time_mlp(t) + self.cond_mlp(cemb) + self.bool_mlp(bool_emb)
        except:
            import pdb; pdb.set_trace()
            out = self.blocks[0](x) + self.time_mlp(t) + self.cond_mlp(cemb) + self.bool_mlp(bool_emb)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class TemporalIDK(nn.Module):

    def __init__(
        self,
        device=None,
    ):
        super().__init__()

        dim = 32
        time_dim = dim
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )
        self.x_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.Mish(),
            nn.Linear(64, 1),
        )

        dim_in = 1
        self.resnet1 = ResidualBlock(dim_in, time_dim, embed_dim=time_dim)

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=5),
            nn.Conv1d(dim, 1, 1),
        )

    def forward(self, x, time, cond=None):
        t = self.time_mlp(time)
        x = self.resnet1(x, t)
        x = self.x_mlp(x) + x
        return self.final_conv(x)


class TemporalUnet(nn.Module):

    def __init__(
        self,
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
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))) if attention else nn.Identity(),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim))) if attention else nn.Identity()
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))) if attention else nn.Identity(),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

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
            hpop = h.pop()
            try:
                x = torch.cat((x, hpop), dim=1)
            except:
                import pdb; pdb.set_trace()
                x = torch.cat((x, hpop), dim=1)
            x = resnet(x, t, cemb, bool_emb)
            x = resnet2(x, t, cemb, bool_emb)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')
        return x[:b_dim, :h_dim, :t_dim]


class TemporalNNet(nn.Module):
    def __init__(
        self,
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

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))) if attention else nn.Identity(),
                Upsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim))) if attention else nn.Identity()
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))) if attention else nn.Identity(),
                Downsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=5),
            nn.Conv1d(dim, d_model, 1),
        )

    def forward(self, x, time, cond):
        b_dim, h_dim, t_dim = x.shape
        x = einops.rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)
        cond = cond if cond is not None else torch.ones(t.shape[0], 1, device=x.device) * -1
        cemb = self.cond_mlp(cond).reshape(cond.shape[0], -1)

        use_cond = (cond > 0.).to(torch.float).reshape(cond.shape)
        bool_emb = self.bool_mlp(use_cond).reshape(cemb.shape)
        h = []

        for idx, (resnet, resnet2, attn, upsample) in enumerate(self.ups):
            x = resnet(x, t, cemb, bool_emb)
            x = resnet2(x, t, cemb, bool_emb)
            x = attn(x)
            h.append(x)
            x = upsample(x)

        x = self.mid_block1(x, t, cemb, bool_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, cemb, bool_emb)

        for idx, (resnet, resnet2, attn, downsample) in enumerate(self.downs):
            hpop = h.pop()
            try:
                x = torch.cat((x, hpop), dim=1)
            except:
                import pdb; pdb.set_trace()
                x = torch.cat((x, hpop), dim=1)
            x = resnet(x, t, cemb, bool_emb)
            x = resnet2(x, t, cemb, bool_emb)
            x = attn(x)
            x = downsample(x)
        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')
        return x[:b_dim, :h_dim, :t_dim]


class TemporalClassifier(nn.Module):

    def __init__(
        self,
        traj_length,
        d_model,
        cond_dim,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        attention=False,
        device=None,
        num_classes=4,
    ):
        super().__init__()

        dims = [d_model, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.in_out = in_out
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        self.d_model = d_model
        self.num_classes = num_classes

        time_dim = dim
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.downs = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualBlock(dim_in, dim_out, embed_dim=time_dim),
                ResidualBlock(dim_out, dim_out, embed_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))) if attention else nn.Identity(),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResidualBlock(mid_dim, mid_dim, embed_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim))) if attention else nn.Identity()
        self.mid_block2 = ResidualBlock(mid_dim, mid_dim, embed_dim=time_dim)

        self.final_conv = nn.Sequential(
            nn.Linear(mid_dim * traj_length // 8, self.num_classes),
        )

        self.other_layer = nn.Linear(1000, 2)

    def forward(self, x, time):
        '''
            x : [ batch x traj_length x transition ]
        '''

        # b_dim, h_dim, t_dim = x.shape
        # x = einops.rearrange(x, 'b h t -> b t h')

        # t = self.time_mlp(time)
        # h = []

        # import pdb; pdb.set_trace()
        # for idx, (resnet, resnet2, attn, downsample) in enumerate(self.downs):
        #     x = resnet(x, t)
        #     x = resnet2(x, t)
        #     x = attn(x)
        #     h.append(x)
        #     x = downsample(x)

        # x = self.mid_block1(x, t)
        # x = self.mid_attn(x)
        # x = self.mid_block2(x, t)

        # x = x.flatten(1)

        # y = self.final_conv(x)

        y = self.other_layer(x[..., 0])
        return y


class NewTemporalClassifier(nn.Module):

    def __init__(
        self,
        traj_length,
        d_model,
        cond_dim,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        attention=True,
        device=None,
        num_classes=4,
    ):
        super().__init__()

        dims = [d_model, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.in_out = in_out
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        self.d_model = d_model
        self.num_classes = num_classes

        time_dim = dim
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        # for ind, (dim_in, dim_out) in enumerate(in_out):
        #     is_last = ind >= (num_resolutions - 1)

        #     self.downs.append(nn.ModuleList([
        #         ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim),
        #         ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim),
        #         Residual(PreNorm(dim_out, Attention(dim_out, permute=True))) if attention else nn.Identity(),
        #         Downsample1d(dim_out) if not is_last else nn.Identity()
        #     ]))

        # mid_dim = dims[-1]
        # self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim)
        # self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim)
        # self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, permute=True))) if attention else nn.Identity()
        # self.mid_block3 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim)
        # self.mid_block4 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim)

        self.residual1 = ResidualBlock(1, dim, embed_dim=time_dim)
        self.residual2 = ResidualBlock(dim, dim, embed_dim=time_dim)
        self.residual3 = nn.Conv1d(dim, 1, 1)

        self.hidden_dim = 2048
        self.final_layer = nn.Sequential(
            nn.Linear(1000, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, num_classes)
        )

    def forward(self, x, time):
        '''
            x : [ batch x traj_length x transition ]
        '''

        b_dim, h_dim, t_dim = x.shape
        # x = einops.rearrange(x, 'b h t -> b t h')

        # t = self.time_mlp(time)

        # y = self.residual1(x, t)
        # y = self.residual2(y, t)
        # y = self.residual3(y)

        # y = einops.rearrange(y, 'b t h -> b h t')
        # y = self.final_layer(y[..., 0])
        # return y

        return self.final_layer(x[..., 0])


class TemporalTransformerUnet(nn.Module):

    def __init__(
        self,
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
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim),
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim),
                Residual(PreNorm(dim_out, Attention(dim_out, permute=True))) if attention else nn.Identity(),
                Downsample1d(dim_out) if not is_last else nn.Identity(),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, permute=True))) if attention else nn.Identity()
        self.mid_block3 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim)
        self.mid_block4 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim),
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim),
                Residual(PreNorm(dim_in, Attention(dim_in, permute=True))) if attention else nn.Identity(),
                Upsample1d(dim_in) if not is_last else nn.Identity(),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

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
        cond = cond if cond is not None else torch.ones(t.shape[0], 1, device=x.device) * -1
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
