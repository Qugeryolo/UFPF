# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Vision Transformer with Deformable Attention
# Modified by Zhuofan Xia
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np

from torch.nn.functional import pad
from timm.models.layers import DropPath, to_2tuple
from timm.models.layers import trunc_normal_
from natten.functional import NATTEN2DQKRPBFunction, NATTEN2DAVFunction


class SlideAttention(nn.Module):

    def __init__(
            self, dim, num_heads, ka, qkv_bias=True, qk_scale=None, attn_drop=0.,
            proj_drop=0., dim_reduction=4, rpb=True, padding_mode='zeros',
            share_dwc_kernel=True, share_qkv=False):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        if share_qkv:
            self.qkv_scale = 1
        else:
            self.qkv_scale = 3
        self.rpb = rpb
        self.share_dwc_kernel = share_dwc_kernel
        self.padding_mode = padding_mode
        self.share_qkv = share_qkv
        self.ka = ka
        self.dim_reduction = dim_reduction
        self.qkv = nn.Linear(dim, dim * self.qkv_scale // self.dim_reduction, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim // self.dim_reduction, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.dep_conv = nn.Conv2d(dim // self.dim_reduction // self.num_heads,
                                  self.ka * self.ka * dim // self.dim_reduction // self.num_heads, kernel_size=self.ka,
                                  bias=True, groups=dim // self.dim_reduction // self.num_heads, padding=self.ka // 2,
                                  padding_mode=padding_mode)
        self.dep_conv1 = nn.Conv2d(dim // self.dim_reduction // self.num_heads,
                                   self.ka * self.ka * dim // self.dim_reduction // self.num_heads, kernel_size=self.ka,
                                   bias=True, groups=dim // self.dim_reduction // self.num_heads, padding=self.ka // 2,
                                   padding_mode=padding_mode)
        if not share_dwc_kernel:
            self.dep_conv2 = nn.Conv2d(dim // self.dim_reduction // self.num_heads,
                                       self.ka * self.ka * dim // self.dim_reduction // self.num_heads,
                                       kernel_size=self.ka, bias=True,
                                       groups=dim // self.dim_reduction // self.num_heads, padding=self.ka // 2,
                                       padding_mode=padding_mode)

        self.reset_parameters()

        # define a parameter table of relative position bias
        if self.rpb:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(1, self.num_heads, 1, self.ka * self.ka, 1, 1))
            trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=3)

    def reset_parameters(self):
        # shift initialization for group convolution
        kernel = torch.zeros(self.ka * self.ka, self.ka, self.ka)
        for i in range(self.ka * self.ka):
            kernel[i, i // self.ka, i % self.ka] = 1.
        kernel = kernel.unsqueeze(1).repeat(self.dim // self.dim_reduction // self.num_heads, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):

        x = einops.rearrange(x, 'b c h w -> b h w c')
        B, H, W, C = x.shape
        qkv = self.qkv(x)

        f_conv = qkv.permute(0, 3, 1, 2).reshape(B * self.num_heads,
                                                 self.qkv_scale * C // self.dim_reduction // self.num_heads, H, W)

        if self.qkv_scale == 3:
            q = (f_conv[:, :C // self.dim_reduction // self.num_heads, :, :] * self.scale).reshape(B, self.num_heads,
                                                                                                   C // self.dim_reduction // self.num_heads,
                                                                                                   1, H, W)
            k = f_conv[:, C // self.dim_reduction // self.num_heads:2 * C // self.dim_reduction // self.num_heads, :,
                :]  # B*self.nhead, C//self.nhead, H, W
            v = f_conv[:, 2 * C // self.dim_reduction // self.num_heads:, :, :]
        elif self.qkv_scale == 1:
            q = (f_conv * self.scale).reshape(B, self.num_heads, C // self.dim_reduction // self.num_heads, 1, H, W)
            k = v = f_conv

        if self.share_dwc_kernel:
            k = (self.dep_conv(k) + self.dep_conv1(k)).reshape(B, self.num_heads,
                                                               C // self.dim_reduction // self.num_heads,
                                                               self.ka * self.ka, H, W)
            v = (self.dep_conv(v) + self.dep_conv1(v)).reshape(B, self.num_heads,
                                                               C // self.dim_reduction // self.num_heads,
                                                               self.ka * self.ka, H, W)
        else:
            k = (self.dep_conv(k) + self.dep_conv1(k)).reshape(B, self.num_heads,
                                                               C // self.dim_reduction // self.num_heads,
                                                               self.ka * self.ka, H, W)
            v = (self.dep_conv(v) + self.dep_conv2(v)).reshape(B, self.num_heads,
                                                               C // self.dim_reduction // self.num_heads,
                                                               self.ka * self.ka, H, W)

        if self.rpb:
            k = k + self.relative_position_bias_table
        attn = (q * k).sum(2, keepdim=True)  # B, self.nhead, 1, k^2, H, W

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn * v).sum(3).reshape(B, C // self.dim_reduction, H, W).permute(0, 2, 3, 1)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = einops.rearrange(x, 'b h w c -> b c h w')
        return x, None, None


class LocalAttention(nn.Module):

    def __init__(self, dim, heads, window_size, attn_drop, proj_drop):
        super().__init__()

        window_size = to_2tuple(window_size)

        self.proj_qkv = nn.Linear(dim, 3 * dim)
        self.heads = heads
        assert dim % heads == 0
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.proj_out = nn.Linear(dim, dim)
        self.window_size = window_size
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        Wh, Ww = self.window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * Wh - 1) * (2 * Ww - 1), heads)
        )
        trunc_normal_(self.relative_position_bias_table, std=0.01)

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x, mask=None):
        B, C, H, W = x.size()
        r1, r2 = H // self.window_size[0], W // self.window_size[1]

        x_total = einops.rearrange(x, 'b c (r1 h1) (r2 w1) -> b (r1 r2) (h1 w1) c', h1=self.window_size[0],
                                   w1=self.window_size[1])  # B x Nr x Ws x C

        x_total = einops.rearrange(x_total, 'b m n c -> (b m) n c')

        qkv = self.proj_qkv(x_total)  # B' x N x 3C
        q, k, v = torch.chunk(qkv, 3, dim=2)

        q = q * self.scale
        q, k, v = [einops.rearrange(t, 'b n (h c1) -> b h n c1', h=self.heads) for t in [q, k, v]]
        attn = torch.einsum('b h m c, b h n c -> b h m n', q, k)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn_bias = relative_position_bias
        attn = attn + attn_bias.unsqueeze(0)

        if mask is not None:
            # attn : (b * nW) h w w
            # mask : nW ww ww
            nW, ww, _ = mask.size()
            attn = einops.rearrange(attn, '(b n) h w1 w2 -> b n h w1 w2', n=nW, h=self.heads, w1=ww,
                                    w2=ww) + mask.reshape(1, nW, 1, ww, ww)
            attn = einops.rearrange(attn, 'b n h w1 w2 -> (b n) h w1 w2')
        attn = self.attn_drop(attn.softmax(dim=3))

        x = torch.einsum('b h m n, b h n c -> b h m c', attn, v)
        x = einops.rearrange(x, 'b h n c1 -> b n (h c1)')
        x = self.proj_drop(self.proj_out(x))  # B' x N x C
        x = einops.rearrange(x, '(b r1 r2) (h1 w1) c -> b c (r1 h1) (r2 w1)', r1=r1, r2=r2, h1=self.window_size[0],
                             w1=self.window_size[1])  # B x C x H x W

        return x, None, None


class ShiftWindowAttention(LocalAttention):

    def __init__(self, dim, heads, window_size, attn_drop, proj_drop, shift_size, fmap_size):

        super().__init__(dim, heads, window_size, attn_drop, proj_drop)

        self.fmap_size = to_2tuple(fmap_size)
        self.shift_size = shift_size

        assert 0 < self.shift_size < min(self.window_size), "wrong shift size."

        img_mask = torch.zeros(*self.fmap_size)  # H W
        h_slices = (slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size[1]),
                    slice(-self.window_size[1], -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[h, w] = cnt
                cnt += 1
        mask_windows = einops.rearrange(img_mask, '(r1 h1) (r2 w1) -> (r1 r2) (h1 w1)', h1=self.window_size[0],
                                        w1=self.window_size[1])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # nW ww ww
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):

        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        sw_x, _, _ = super().forward(shifted_x, self.attn_mask)
        x = torch.roll(sw_x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))

        return x, None, None


class DAttentionBaseline(nn.Module):

    def __init__(
            self, q_size, kv_size, n_heads, n_head_channels, n_groups,
            attn_drop, proj_drop, stride,
            offset_range_factor, use_pe, dwc_pe,
            no_off, fixed_pe, ksize, log_cpb
    ):

        super().__init__()
        self.dwc_pe = dwc_pe
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.q_h, self.q_w = q_size
        # self.kv_h, self.kv_w = kv_size
        self.kv_h, self.kv_w = self.q_h // stride, self.q_w // stride
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor
        self.ksize = ksize
        self.log_cpb = log_cpb
        self.stride = stride
        kk = self.ksize
        pad_size = kk // 2 if kk != stride else 0

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )
        if self.no_off:
            for m in self.conv_offset.parameters():
                m.requires_grad_(False)

        self.proj_q = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_k = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_v = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_out = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        if self.use_pe and not self.no_off:
            if self.dwc_pe:
                self.rpe_table = nn.Conv2d(
                    self.nc, self.nc, kernel_size=3, stride=1, padding=1, groups=self.nc)
            elif self.fixed_pe:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_h * self.q_w, self.kv_h * self.kv_w)
                )
                trunc_normal_(self.rpe_table, std=0.01)
            elif self.log_cpb:
                # Borrowed from Swin-V2
                self.rpe_table = nn.Sequential(
                    nn.Linear(2, 32, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(32, self.n_group_heads, bias=False)
                )
            else:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_h * 2 - 1, self.q_w * 2 - 1)
                )
                trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B * g H W 2

        return ref

    @torch.no_grad()
    def _get_q_grid(self, H, W, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.arange(0, H, dtype=dtype, device=device),
            torch.arange(0, W, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B * g H W 2

        return ref

    def forward(self, x):

        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device

        q = self.proj_q(x)
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        offset = self.conv_offset(q_off).contiguous()  # B * g 2 Hg Wg
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk

        if self.offset_range_factor >= 0 and not self.no_off:
            offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        if self.no_off:
            offset = offset.fill_(0.0)

        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).clamp(-1., +1.)

        if self.no_off:
            x_sampled = F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)
            assert x_sampled.size(2) == Hk and x_sampled.size(3) == Wk, f"Size is {x_sampled.size()}"
        else:
            x_sampled = F.grid_sample(
                input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),
                grid=pos[..., (1, 0)],  # y, x -> x, y
                mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg

        x_sampled = x_sampled.reshape(B, C, 1, n_sample)

        q = q.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)

        attn = torch.einsum('b c m, b c n -> b m n', q, k)  # B * h, HW, Ns
        attn = attn.mul(self.scale)

        if self.use_pe and (not self.no_off):

            if self.dwc_pe:
                residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(B * self.n_heads, self.n_head_channels,
                                                                              H * W)
            elif self.fixed_pe:
                rpe_table = self.rpe_table
                attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                attn = attn + attn_bias.reshape(B * self.n_heads, H * W, n_sample)
            elif self.log_cpb:
                q_grid = self._get_q_grid(H, W, B, dtype, device)
                displacement = (
                            q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups,
                                                                                                   n_sample,
                                                                                                   2).unsqueeze(1)).mul(
                    4.0)  # d_y, d_x [-8, +8]
                displacement = torch.sign(displacement) * torch.log2(torch.abs(displacement) + 1.0) / np.log2(8.0)
                attn_bias = self.rpe_table(displacement)  # B * g, H * W, n_sample, h_g
                attn = attn + einops.rearrange(attn_bias, 'b m n h -> (b h) m n', h=self.n_group_heads)
            else:
                rpe_table = self.rpe_table
                rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                q_grid = self._get_q_grid(H, W, B, dtype, device)
                displacement = (
                            q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups,
                                                                                                   n_sample,
                                                                                                   2).unsqueeze(1)).mul(
                    0.5)
                attn_bias = F.grid_sample(
                    input=einops.rearrange(rpe_bias, 'b (g c) h w -> (b g) c h w', c=self.n_group_heads,
                                           g=self.n_groups),
                    grid=displacement[..., (1, 0)],
                    mode='bilinear', align_corners=True)  # B * g, h_g, HW, Ns

                attn_bias = attn_bias.reshape(B * self.n_heads, H * W, n_sample)
                attn = attn + attn_bias

        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)

        out = torch.einsum('b m n, b c n -> b c m', attn, v)

        if self.use_pe and self.dwc_pe:
            out = out + residual_lepe
        out = out.reshape(B, C, H, W)

        y = self.proj_drop(self.proj_out(out))

        return y, pos.reshape(B, self.n_groups, Hk, Wk, 2), reference.reshape(B, self.n_groups, Hk, Wk, 2)


class PyramidAttention(nn.Module):

    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0., sr_ratio=1):

        super().__init__()

        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Conv2d(dim, dim, 1, 1, 0)
        self.kv = nn.Conv2d(dim, dim * 2, 1, 1, 0)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1, 1, 0)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.proj_ds = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio),
                LayerNormProxy(dim)
            )

    def forward(self, x):

        B, C, H, W = x.size()
        Nq = H * W
        q = self.q(x)

        if self.sr_ratio > 1:
            x_ds = self.proj_ds(x)
            kv = self.kv(x_ds)
        else:
            kv = self.kv(x)

        k, v = torch.chunk(kv, 2, dim=1)
        Nk = (H // self.sr_ratio) * (W // self.sr_ratio)
        q = q.reshape(B * self.num_heads, self.head_dim, Nq).mul(self.scale)
        k = k.reshape(B * self.num_heads, self.head_dim, Nk)
        v = v.reshape(B * self.num_heads, self.head_dim, Nk)
        attn = torch.einsum('b c m, b c n -> b m n', q, k)
        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)

        x = torch.einsum('b m n, b c n -> b c m', attn, v)
        x = x.reshape(B, C, H, W)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, None, None


class TransformerMLP(nn.Module):

    def __init__(self, channels, expansion, drop):
        super().__init__()

        self.dim1 = channels
        self.dim2 = channels * expansion
        self.chunk = nn.Sequential()
        self.chunk.add_module('linear1', nn.Linear(self.dim1, self.dim2))
        self.chunk.add_module('act', nn.GELU())
        self.chunk.add_module('drop1', nn.Dropout(drop, inplace=True))
        self.chunk.add_module('linear2', nn.Linear(self.dim2, self.dim1))
        self.chunk.add_module('drop2', nn.Dropout(drop, inplace=True))

    def forward(self, x):
        _, _, H, W = x.size()
        x = einops.rearrange(x, 'b c h w -> b (h w) c')
        x = self.chunk(x)
        x = einops.rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x


class LayerNormProxy(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')


class TransformerMLPWithConv(nn.Module):

    def __init__(self, channels, expansion, drop):
        super().__init__()

        self.dim1 = channels
        self.dim2 = channels * expansion
        self.linear1 = nn.Sequential(
            nn.Conv2d(self.dim1, self.dim2, 1, 1, 0),
            # nn.GELU(),
            # nn.BatchNorm2d(self.dim2, eps=1e-5)
        )
        self.drop1 = nn.Dropout(drop, inplace=True)
        self.act = nn.GELU()
        # self.bn = nn.BatchNorm2d(self.dim2, eps=1e-5)
        self.linear2 = nn.Sequential(
            nn.Conv2d(self.dim2, self.dim1, 1, 1, 0),
            # nn.BatchNorm2d(self.dim1, eps=1e-5)
        )
        self.drop2 = nn.Dropout(drop, inplace=True)
        self.dwc = nn.Conv2d(self.dim2, self.dim2, 3, 1, 1, groups=self.dim2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.drop1(x)
        x = x + self.dwc(x)
        x = self.act(x)
        # x = self.bn(x)
        x = self.linear2(x)
        x = self.drop2(x)

        return x


class FusedKQnA(nn.Module):

    def __init__(self, n_q, n_channels, n_heads, ksize, stride, padding, qna_activation):

        super().__init__()
        self.n_q = n_q
        self.n_channels = n_channels
        self.n_heads = n_heads
        self.ksize = ksize
        self.stride = stride
        self.padding = padding
        self.head_channels = n_channels // n_heads
        self.qna_activation = qna_activation

        self.proj_k = nn.Linear(self.n_channels, self.n_channels * stride, bias=False)
        self.proj_v = nn.Linear(self.n_channels, self.n_channels * stride, bias=False)
        self.proj_out = nn.Conv2d(self.n_channels * stride, self.n_channels * stride, 1, 1, 0, bias=False)
        self.scale = self.head_channels ** -0.5
        self.q_param = nn.Parameter(torch.empty(self.n_q, self.n_channels * stride))
        trunc_normal_(self.q_param, std=math.sqrt(1.0 / self.head_channels))

        self.attn_scale = nn.Parameter(torch.empty(1, 1, self.ksize ** 2, self.n_q * self.n_heads * stride))
        nn.init.normal_(self.attn_scale, std=0.02)

        self.rpb_table = nn.Parameter(torch.empty(self.ksize ** 2, self.n_q * self.n_heads * stride))
        trunc_normal_(self.rpb_table, std=0.02)

    def forward(self, x):
        # assert not torch.isnan(x).any(), 'NaN in x'
        B, C, H, W = x.size()
        x = einops.rearrange(x, 'b c h w -> b (h w) c')
        N = H * W
        q = self.q_param[None, ...].expand(B, self.n_q, C * self.stride)
        # assert not torch.isnan(q).any(), 'NaN in q'
        q = einops.rearrange(q, 'b q (h c) -> b h q c', h=self.n_heads * self.stride, c=self.head_channels, q=self.n_q)
        # q_norm_l2 = torch.linalg.vector_norm(q, ord=2, dim=3, keepdim=True)
        # q = q / (q_norm_l2 + 1e-6)
        k = self.proj_k(x)
        B, N, C = k.size()
        # assert not torch.isnan(k).any(), 'NaN in k'
        k = einops.rearrange(k, 'b k (h c) -> b h k c', h=self.n_heads * self.stride, c=self.head_channels, k=N)
        q = q * self.scale
        qkT = torch.einsum('b h q c, b h k c -> b h q k', q, k)
        qkT = einops.rearrange(qkT, 'b h q k -> b k q h')

        # assert not torch.isnan(qkT).any(), 'NaN in qkT'

        v = self.proj_v(x)
        attn_scale = self.attn_scale.reshape(self.ksize, self.ksize, 1, self.n_q * self.n_heads * self.stride)
        rpb = self.rpb_table.reshape(self.ksize, self.ksize, 1, self.n_q * self.n_heads * self.stride)

        if self.qna_activation == 'exp':
            cost_exp = torch.exp(qkT - qkT.max().detach())
        elif self.qna_activation == 'sigmoid':
            cost_exp = qkT.sigmoid()
        elif self.qna_activation == 'linear':
            cost_exp = qkT

        # assert not torch.isnan(cost_exp).any(), 'NaN in cost_exp'

        v_cost_exp = cost_exp[..., None] * v.reshape(B, N, 1, self.n_heads * self.stride, self.head_channels)
        # v_cost_exp : B N n_q h ch
        v_cost_exp = v_cost_exp.reshape(B, N, self.n_q * self.n_heads * self.stride * self.head_channels)

        if self.qna_activation == 'exp':
            rpb_exp = torch.exp(rpb - rpb.max().detach())
        elif self.qna_activation == 'sigmoid':
            rpb_exp = rpb.sigmoid()
        elif self.qna_activation == 'linear':
            rpb_exp = rpb

        summation_kernel = (rpb_exp * attn_scale).repeat_interleave(self.head_channels, dim=3)
        # summation_kernel : [kh, kw, 1, n_q * h * Ch] HWIO
        v_cost_exp_ = einops.rearrange(v_cost_exp, 'b (h w) c -> b c h w', h=H, w=W)
        v_kern_ = einops.rearrange(summation_kernel, 'h w i o -> o i h w')
        sum_num = F.conv2d(
            v_cost_exp_,
            v_kern_,
            stride=self.stride,
            padding=self.padding,
            groups=self.n_q * self.n_channels * self.stride
        )  # B Nq*h*Ch H
        sum_num = einops.rearrange(
            sum_num,
            'b (q g c) h w -> b q g c h w',
            q=self.n_q, g=self.n_heads * self.stride, c=self.head_channels)
        cost_exp_ = einops.rearrange(
            cost_exp,
            'b (h w) q c -> b (q c) h w',
            h=H,
            w=W,
            q=self.n_q,
            c=self.n_heads * self.stride)  # B Nq*h H W
        kern_ = einops.rearrange(rpb_exp, 'h w i o -> o i h w')

        sum_den = F.conv2d(
            cost_exp_,
            kern_,
            stride=self.stride,
            padding=self.padding,
            groups=self.n_q * self.n_heads * self.stride
        )  # B Nq*h H W
        sum_den = einops.rearrange(
            sum_den,
            'b (q g c) h w -> b q g c h w',
            q=self.n_q, g=self.n_heads * self.stride, c=1)
        H = H // self.stride
        W = W // self.stride
        out = (sum_num / sum_den).sum(dim=1).reshape(B, C, H, W)
        out = self.proj_out(out)
        return out, None, None


class NeighborhoodAttention2D(nn.Module):
    """
    Neighborhood Attention 2D Module
    """
    def __init__(self, dim, kernel_size, num_heads, attn_drop=0., proj_drop=0.,
                 dilation=None):
        super().__init__()
        self.fp16_enabled = False
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11, 13], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
        self.kernel_size = kernel_size
        if type(dilation) is str:
            self.dilation = None
            self.window_size = None
        else:
            assert dilation is None or dilation >= 1, \
                f"Dilation must be greater than or equal to 1, got {dilation}."
            self.dilation = dilation or 1
            self.window_size = self.kernel_size * self.dilation

        self.qkv = nn.Linear(dim, dim * 3)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # assert x.dtype == torch.float16, f"AMP failed!, dtype={x.dtype}"
        x = x.permute(0, 2, 3, 1)
        B, Hp, Wp, C = x.shape
        H, W = int(Hp), int(Wp)
        pad_l = pad_t = pad_r = pad_b = 0
        dilation = self.dilation
        window_size = self.window_size
        if window_size is None:
            dilation = max(min(H, W) // self.kernel_size, 1)
            window_size = dilation * self.kernel_size
        if H < window_size or W < window_size:
            pad_l = pad_t = 0
            pad_r = max(0, window_size - W)
            pad_b = max(0, window_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        # breakpoint()
        attn = NATTEN2DQKRPBFunction.apply(q, k, self.rpb, self.kernel_size, dilation)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = NATTEN2DAVFunction.apply(attn, v, self.kernel_size, dilation)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x)).permute(0, 3, 1, 2), None, None



class LayerScale(nn.Module):

    def __init__(self,
                 dim: int,
                 inplace: bool = False,
                 init_values: float = 1e-5):
        super().__init__()
        self.inplace = inplace
        self.weight = nn.Parameter(torch.ones(dim) * init_values)

    def forward(self, x):
        if self.inplace:
            return x.mul_(self.weight.view(-1, 1, 1))
        else:
            return x * self.weight.view(-1, 1, 1)


class TransformerStage(nn.Module):

    def __init__(self, fmap_size, window_size, ns_per_pt,
                 dim_in, dim_embed, depths, stage_spec, n_groups,
                 use_pe, sr_ratio,
                 heads, heads_q, stride,
                 offset_range_factor,
                 dwc_pe, no_off, fixed_pe,
                 attn_drop, proj_drop, expansion, drop, drop_path_rate,
                 use_dwc_mlp, ksize, nat_ksize,
                 k_qna, nq_qna, qna_activation,
                 layer_scale_value, use_lpu, log_cpb):

        super().__init__()
        fmap_size = to_2tuple(fmap_size)
        self.depths = depths
        hc = dim_embed // heads
        assert dim_embed == heads * hc
        self.proj = nn.Conv2d(dim_in, dim_embed, 1, 1, 0) if dim_in != dim_embed else nn.Identity()
        self.stage_spec = stage_spec
        self.use_lpu = use_lpu

        self.ln_cnvnxt = nn.ModuleDict(
            {str(d): LayerNormProxy(dim_embed) for d in range(depths) if stage_spec[d] == 'X'}
        )
        self.layer_norms = nn.ModuleList(
            [LayerNormProxy(dim_embed) if stage_spec[d // 2] != 'X' else nn.Identity() for d in range(2 * depths)]
        )

        mlp_fn = TransformerMLPWithConv if use_dwc_mlp else TransformerMLP

        self.mlps = nn.ModuleList(
            [
                mlp_fn(dim_embed, expansion, drop) for _ in range(depths)
            ]
        )
        self.attns = nn.ModuleList()
        self.drop_path = nn.ModuleList()
        self.layer_scales = nn.ModuleList(
            [
                LayerScale(dim_embed, init_values=layer_scale_value) if layer_scale_value > 0.0 else nn.Identity()
                for _ in range(2 * depths)
            ]
        )
        self.local_perception_units = nn.ModuleList(
            [
                nn.Conv2d(dim_embed, dim_embed, kernel_size=3, stride=1, padding=1,
                          groups=dim_embed) if use_lpu else nn.Identity()
                for _ in range(depths)
            ]
        )

        for i in range(depths):
            if stage_spec[i] == 'L':
                self.attns.append(
                    LocalAttention(dim_embed, heads, window_size, attn_drop, proj_drop)
                )
            elif stage_spec[i] == 'D':
                self.attns.append(
                    DAttentionBaseline(fmap_size, fmap_size, heads,
                                       hc, n_groups, attn_drop, proj_drop,
                                       stride, offset_range_factor, use_pe, dwc_pe,
                                       no_off, fixed_pe, ksize, log_cpb)
                )
            elif stage_spec[i] == 'S':
                shift_size = math.ceil(window_size / 2)
                self.attns.append(
                    ShiftWindowAttention(dim_embed, heads, window_size, attn_drop, proj_drop, shift_size, fmap_size)
                )
            elif stage_spec[i] == 'N':
                self.attns.append(
                    NeighborhoodAttention2D(dim_embed, nat_ksize, heads, attn_drop, proj_drop)
                )
            elif stage_spec[i] == 'P':
                self.attns.append(
                    PyramidAttention(dim_embed, heads, attn_drop, proj_drop, sr_ratio)
                )
            elif stage_spec[i] == 'Q':
                self.attns.append(
                    FusedKQnA(nq_qna, dim_embed, heads_q, k_qna, 1, 0, qna_activation)
                )
            elif self.stage_spec[i] == 'X':
                self.attns.append(
                    nn.Conv2d(dim_embed, dim_embed, kernel_size=window_size, padding=window_size // 2, groups=dim_embed)
                )
            elif self.stage_spec[i] == 'E':
                self.attns.append(
                    SlideAttention(dim_embed, heads, 3)
                )
            else:
                raise NotImplementedError(f'Spec: {stage_spec[i]} is not supported.')

            self.drop_path.append(DropPath(drop_path_rate[i]) if drop_path_rate[i] > 0.0 else nn.Identity())

    def forward(self, x):

        x = self.proj(x)

        for d in range(self.depths):

            if self.use_lpu:
                x0 = x
                x = self.local_perception_units[d](x.contiguous())
                x = x + x0

            if self.stage_spec[d] == 'X':
                x0 = x
                x = self.attns[d](x)
                x = self.mlps[d](self.ln_cnvnxt[str(d)](x))
                x = self.drop_path[d](x) + x0
            else:
                x0 = x
                x, pos, ref = self.attns[d](self.layer_norms[2 * d](x))
                x = self.layer_scales[2 * d](x)
                x = self.drop_path[d](x) + x0
                x0 = x
                x = self.mlps[d](self.layer_norms[2 * d + 1](x))
                x = self.layer_scales[2 * d + 1](x)
                x = self.drop_path[d](x) + x0

        return x


class DAT(nn.Module):

    def __init__(self, img_size=256, patch_size=4, num_classes=2, expansion=4,
                 dim_stem=96, dims=[96, 192, 384, 768], depths=[2, 2, 6, 2],
                 heads=[3, 6, 12, 24], heads_q=[6, 12, 24, 48],
                 window_sizes=[7, 7, 7, 7],
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
                 strides=[-1, -1, -1, -1],
                 offset_range_factor=[1, 2, 3, 4],
                 stage_spec=[['L', 'D'], ['L', 'D'], ['L', 'D', 'L', 'D', 'L', 'D'], ['L', 'D']],
                 groups=[-1, -1, 3, 6],
                 use_pes=[False, False, False, False],
                 dwc_pes=[False, False, False, False],
                 sr_ratios=[8, 4, 2, 1],
                 lower_lr_kvs={},
                 fixed_pes=[False, False, False, False],
                 no_offs=[False, False, False, False],
                 ns_per_pts=[4, 4, 4, 4],
                 use_dwc_mlps=[False, False, False, False],
                 use_conv_patches=False,
                 ksizes=[9, 7, 5, 3],
                 ksize_qnas=[3, 3, 3, 3],
                 nqs=[2, 2, 2, 2],
                 qna_activation='exp',
                 nat_ksizes=[3, 3, 3, 3],
                 layer_scale_values=[-1, -1, -1, -1],
                 use_lpus=[False, False, False, False],
                 log_cpb=[False, False, False, False],
                 **kwargs):
        super().__init__()

        self.patch_proj = nn.Sequential(
            nn.Conv2d(3, dim_stem // 2, 3, patch_size // 2, 1),
            LayerNormProxy(dim_stem // 2),
            nn.GELU(),
            nn.Conv2d(dim_stem // 2, dim_stem, 3, patch_size // 2, 1),
            LayerNormProxy(dim_stem)
        ) if use_conv_patches else nn.Sequential(
            nn.Conv2d(3, dim_stem, patch_size, patch_size, 0),
            LayerNormProxy(dim_stem)
        )

        img_size = img_size // patch_size
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.stages = nn.ModuleList()
        for i in range(4):
            dim1 = dim_stem if i == 0 else dims[i - 1] * 2
            dim2 = dims[i]
            self.stages.append(
                TransformerStage(
                    img_size, window_sizes[i], ns_per_pts[i],
                    dim1, dim2, depths[i],
                    stage_spec[i], groups[i], use_pes[i],
                    sr_ratios[i], heads[i], heads_q[i], strides[i],
                    offset_range_factor[i],
                    dwc_pes[i], no_offs[i], fixed_pes[i],
                    attn_drop_rate, drop_rate, expansion, drop_rate,
                    dpr[sum(depths[:i]):sum(depths[:i + 1])], use_dwc_mlps[i],
                    ksizes[i], nat_ksizes[i], ksize_qnas[i], nqs[i], qna_activation,
                    layer_scale_values[i], use_lpus[i], log_cpb[i]
                )
            )
            img_size = img_size // 2

        self.down_projs = nn.ModuleList()
        for i in range(3):
            self.down_projs.append(
                nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], 3, 2, 1, bias=False),
                    LayerNormProxy(dims[i + 1])
                ) if use_conv_patches else nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], 2, 2, 0, bias=False),
                    LayerNormProxy(dims[i + 1])
                )
            )

        self.cls_norm = LayerNormProxy(dims[-1])
        self.cls_head = nn.Linear(dims[-1], num_classes)

        self.lower_lr_kvs = lower_lr_kvs

        self.reset_parameters()

    def reset_parameters(self):

        for m in self.parameters():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    @torch.no_grad()
    def load_pretrained(self, state_dict, lookup_22k):

        new_state_dict = {}
        for state_key, state_value in state_dict.items():
            keys = state_key.split('.')
            m = self
            for key in keys:
                if key.isdigit():
                    m = m[int(key)]
                else:
                    m = getattr(m, key)
            if m.shape == state_value.shape:
                new_state_dict[state_key] = state_value
            else:
                # Ignore different shapes
                if 'relative_position_index' in keys:
                    new_state_dict[state_key] = m.data
                if 'q_grid' in keys:
                    new_state_dict[state_key] = m.data
                if 'reference' in keys:
                    new_state_dict[state_key] = m.data
                # Bicubic Interpolation
                if 'relative_position_bias_table' in keys:
                    n, c = state_value.size()
                    l_side = int(math.sqrt(n))
                    assert n == l_side ** 2
                    L = int(math.sqrt(m.shape[0]))
                    pre_interp = state_value.reshape(1, l_side, l_side, c).permute(0, 3, 1, 2)
                    post_interp = F.interpolate(pre_interp, (L, L), mode='bicubic')
                    new_state_dict[state_key] = post_interp.reshape(c, L ** 2).permute(1, 0)
                if 'rpe_table' in keys:
                    c, h, w = state_value.size()
                    C, H, W = m.data.size()
                    pre_interp = state_value.unsqueeze(0)
                    post_interp = F.interpolate(pre_interp, (H, W), mode='bicubic')
                    new_state_dict[state_key] = post_interp.squeeze(0)
                if 'cls_head' in keys:
                    new_state_dict[state_key] = state_value[lookup_22k]

        msg = self.load_state_dict(new_state_dict, strict=False)
        return msg

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table', 'rpe_table'}

    def forward(self, x):
        x = self.patch_proj(x)
        for i in range(4):
            x = self.stages[i](x)
            if i < 3:
                x = self.down_projs[i](x)
        x = self.cls_norm(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.cls_head(x)
        return x, None, None