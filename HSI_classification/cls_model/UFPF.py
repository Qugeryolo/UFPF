# encoding: utf-8
# @Time    : 2024/5/16 下午8:41
# @Author  : Geng Qin
# @File    : UFPF.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import logging
import math
from os.path import join as pjoin
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
import ml_collections
from collections import OrderedDict
from mamba_ssm import Mamba
from torch.cuda.amp import autocast
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
import time
"""
===============================================================================================
    Args:
        Configs
    :return dict
===============================================================================================
"""

def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.classifier = 'seg'
    config.representation_size = None
    config.resnet_pretrained_path = None
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_16.npz'
    config.patch_size = 16

    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2
    config.activation = 'softmax'
    return config


def get_testing():
    """Returns a minimal configuration for testing."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1
    config.transformer.num_heads = 1
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config


def get_r50_b16_config():
    """Returns the Resnet50 + ViT-B/16 configuration."""
    config = get_b16_config()
    config.patches.grid = (16, 16)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1

    config.classifier = 'seg'
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 2
    config.n_skip = 3
    config.activation = 'softmax'

    return config


def get_b32_config():
    """Returns the ViT-B/32 configuration."""
    config = get_b16_config()
    config.patches.size = (32, 32)
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_32.npz'
    return config


def get_l16_config():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1024
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 4096
    config.transformer.num_heads = 16
    config.transformer.num_layers = 24
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.representation_size = None

    # custom
    config.classifier = 'seg'
    config.resnet_pretrained_path = None
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-L_16.npz'
    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2
    config.activation = 'softmax'
    return config


def get_r50_l16_config():
    """Returns the Resnet50 + ViT-L/16 configuration. customized """
    config = get_l16_config()
    config.patches.grid = (16, 16)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1

    config.classifier = 'seg'
    config.resnet_pretrained_path = '../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 2
    config.activation = 'softmax'
    return config


def get_l32_config():
    """Returns the ViT-L/32 configuration."""
    config = get_l16_config()
    config.patches.size = (32, 32)
    return config


def get_h14_config():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (14, 14)})
    config.hidden_size = 1280
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 5120
    config.transformer.num_heads = 16
    config.transformer.num_layers = 32
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None

    return config


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )

    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)

        return out


class BiPixelMambaLayer(nn.Module):
    def __init__(self, dim, p, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)

        self.mamba_forw = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            use_fast_path=False,
        )

        self.out_proj = copy.deepcopy(self.mamba_forw.out_proj)

        self.mamba_backw = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            use_fast_path=False,

        )

        # adjust the window size here to fit the feature map
        self.p = p * 5
        self.p1 = 5 * p
        self.p2 = 7 * p
        self.p3 = 6 * p
        self.mamba_forw.out_proj = nn.Identity()
        self.mamba_backw.out_proj = nn.Identity()
        self.conv = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)

    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)

        ll = len(x.shape)

        B, C = x.shape[:2]

        assert C == self.dim
        img_dims = x.shape[2:]


        H, W = x.shape[2:]

        if H % self.p == 0 and W % self.p == 0:
            x_div = x.reshape(B, C, H // self.p, self.p, W // self.p, self.p).permute(0, 3, 5, 1, 2,
                                                                                          4).contiguous().view(B * self.p * self.p, C, H // self.p, W // self.p)
        else:
            x_div = x

        NB = x_div.shape[0]

        NH, NW = x_div.shape[2:]
        # print(NH, NW)

        """fo corner"""
        B, C, H, W = x_div.shape
        H_half, W_half = H // 2, W // 2
        
        x_tl = x_div[:, :, :H_half, :W_half]  
        x_tr = x_div[:, :, :H_half, W_half:]  
        x_bl = x_div[:, :, H_half:, :W_half]  
        x_br = x_div[:, :, H_half:, W_half:] 
        
        x_tr_rot = torch.rot90(x_tr, dims=[-2, -1], k=3)
        x_bl_rot = torch.rot90(x_bl, dims=[-2, -1], k=1)
        x_br_rot = torch.rot90(x_br, dims=[-2, -1], k=2)

        n_tokens = x_tl.shape[2:].numel()
        
        x_tl_flat = x_tl.reshape(B, C, n_tokens).transpose(-1, -2)
        x_tr_flat = x_tr_rot.reshape(B, C, n_tokens).transpose(-1, -2)
        x_bl_flat = x_bl_rot.reshape(B, C, n_tokens).transpose(-1, -2)
        x_br_flat = x_br_rot.reshape(B, C, n_tokens).transpose(-1, -2)
        
        x_tl_norm = self.norm(x_tl_flat)
        x_tr_norm = self.norm(x_tr_flat)
        x_bl_norm = self.norm(x_bl_flat)
        x_br_norm = self.norm(x_br_flat)
        
        x_tl_mamba = self.mamba_forw(x_tl_norm)
        x_tr_mamba = self.mamba_forw(x_tr_norm).permute(0, 2, 1)
        x_bl_mamba = self.mamba_forw(x_bl_norm).permute(0, 2, 1)
        x_br_mamba = self.mamba_forw(x_br_norm).permute(0, 2, 1)
        
        b, hw, c = x_tl_mamba.size()
        sqrt_hw = int(hw ** 0.5)
        
        x_tl_out = x_tl_mamba

        x_tr_mamba = x_tr_mamba.view(b, c, sqrt_hw, sqrt_hw)
        x_bl_mamba = x_bl_mamba.view(b, c, sqrt_hw, sqrt_hw)
        x_br_mamba = x_br_mamba.view(b, c, sqrt_hw, sqrt_hw)
        
        x_tr_mamba = torch.rot90(x_tr_mamba, dims=[-2, -1], k=1)
        x_bl_mamba = torch.rot90(x_bl_mamba, dims=[-2, -1], k=3)
        x_br_mamba = torch.rot90(x_br_mamba, dims=[-2, -1], k=2)  
        
        x_tr_out = x_tr_mamba.reshape(b, c, hw).transpose(-1, -2)
        x_bl_out = x_bl_mamba.reshape(b, c, hw).transpose(-1, -2)
        x_br_out = x_br_mamba.reshape(b, c, hw).transpose(-1, -2)
        
        x_tl_out = x_tl_out.transpose(-1, -2).reshape(b, c, sqrt_hw, sqrt_hw)
        x_tr_out = x_tr_out.transpose(-1, -2).reshape(b, c, sqrt_hw, sqrt_hw)
        x_bl_out = x_bl_out.transpose(-1, -2).reshape(b, c, sqrt_hw, sqrt_hw)
        x_br_out = x_br_out.transpose(-1, -2).reshape(b, c, sqrt_hw, sqrt_hw)
        
        top_out = torch.cat([x_tl_out, x_tr_out], dim=-1)
        bottom_out = torch.cat([x_bl_out, x_br_out], dim=-1)
        x_out = torch.cat([top_out, bottom_out], dim=-2)
        
        # x_out = self.out_proj(x_out)
        x_out = self.conv(x_out)


        out = x_out + x

        return out


class BiWindowMambaLayer(nn.Module):
    def __init__(self, dim, p, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)

        self.p = p
        self.mamba_forw1 = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            use_fast_path=False,
        )

        self.mamba_forw2 = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            use_fast_path=False,
        )

        self.mamba_forw3 = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            use_fast_path=False,
        )

        self.mamba_forw4 = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            use_fast_path=False,
        )

        # self.out_proj = copy.deepcopy(self.mamba_forw1.out_proj)
        self.out_proj = nn.Identity()
        self.conv = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)


    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)

        ll = len(x.shape)

        B, C = x.shape[:2]

        assert C == self.dim

        img_dims = x.shape[2:]

        H, W = x.shape[2:]
        if self.p == 0:
            self.p = 1
        if H % self.p == 0 and W % self.p == 0:
            pool_layer = nn.AvgPool2d(self.p, stride=self.p)
            x_div = pool_layer(x)
        else:
            x_div = x

        NH, NW = x_div.shape[2:]

        """fo corner"""
        B, C, H, W = x_div.shape
        H_half, W_half = H // 2, W // 2
        
        x_tl = x_div[:, :, :H_half, :W_half]  
        x_tr = x_div[:, :, :H_half, W_half:]  
        x_bl = x_div[:, :, H_half:, :W_half]  
        x_br = x_div[:, :, H_half:, W_half:] 
        
        x_tr_rot = torch.rot90(x_tr, dims=[-2, -1], k=3)
        x_bl_rot = torch.rot90(x_bl, dims=[-2, -1], k=1)
        x_br_rot = torch.rot90(x_br, dims=[-2, -1], k=2)

        n_tokens = x_tl.shape[2:].numel()
        
        x_tl_flat = x_tl.reshape(B, C, n_tokens).transpose(-1, -2)
        x_tr_flat = x_tr_rot.reshape(B, C, n_tokens).transpose(-1, -2)
        x_bl_flat = x_bl_rot.reshape(B, C, n_tokens).transpose(-1, -2)
        x_br_flat = x_br_rot.reshape(B, C, n_tokens).transpose(-1, -2)
        
        x_tl_norm = self.norm(x_tl_flat)
        x_tr_norm = self.norm(x_tr_flat)
        x_bl_norm = self.norm(x_bl_flat)
        x_br_norm = self.norm(x_br_flat)
        
        x_tl_mamba = self.mamba_forw1(x_tl_norm)
        x_tr_mamba = self.mamba_forw2(x_tr_norm).permute(0, 2, 1)
        x_bl_mamba = self.mamba_forw3(x_bl_norm).permute(0, 2, 1)
        x_br_mamba = self.mamba_forw4(x_br_norm).permute(0, 2, 1)
        
        b, hw, c = x_tl_mamba.size()
        sqrt_hw = int(hw ** 0.5)
        
        x_tl_out = x_tl_mamba

        x_tr_mamba = x_tr_mamba.view(b, c, sqrt_hw, sqrt_hw)
        x_bl_mamba = x_bl_mamba.view(b, c, sqrt_hw, sqrt_hw)
        x_br_mamba = x_br_mamba.view(b, c, sqrt_hw, sqrt_hw)
        
        x_tr_mamba = torch.rot90(x_tr_mamba, dims=[-2, -1], k=1)
        x_bl_mamba = torch.rot90(x_bl_mamba, dims=[-2, -1], k=3)
        x_br_mamba = torch.rot90(x_br_mamba, dims=[-2, -1], k=2)  
        
        x_tr_out = x_tr_mamba.reshape(b, c, hw).transpose(-1, -2)
        x_bl_out = x_bl_mamba.reshape(b, c, hw).transpose(-1, -2)
        x_br_out = x_br_mamba.reshape(b, c, hw).transpose(-1, -2)
        
        x_tl_out = x_tl_out.transpose(-1, -2).reshape(b, c, sqrt_hw, sqrt_hw)
        x_tr_out = x_tr_out.transpose(-1, -2).reshape(b, c, sqrt_hw, sqrt_hw)
        x_bl_out = x_bl_out.transpose(-1, -2).reshape(b, c, sqrt_hw, sqrt_hw)
        x_br_out = x_br_out.transpose(-1, -2).reshape(b, c, sqrt_hw, sqrt_hw)
        
        top_out = torch.cat([x_tl_out, x_tr_out], dim=-1)
        bottom_out = torch.cat([x_bl_out, x_br_out], dim=-1)
        x_out = torch.cat([top_out, bottom_out], dim=-2)
        # x_out = self.conv(x_out)

        out = x_out + x

        return out


class Combinmambalayer(nn.Module):
    def __init__(self, dim):
        super(Combinmambalayer, self).__init__()
        self.pixelmamba = BiPixelMambaLayer(dim=dim//2, p=4)
        self.winmamba = BiWindowMambaLayer(dim=dim//2, p=1)
        self.conv = nn.Conv2d(dim, dim // 2, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x_p = self.pixelmamba(x)
        x_w = self.win_mamba(x)
        out = torch.cat([x_p, x_w], dim=1)
        return out



"""
======================================================================================================
"""
class ChannelAttentionCBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttentionCBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


# Squeeze-and-excitation
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = self.squeeze(x).view(batch_size, channels)
        y = self.excitation(y).view(batch_size, channels, 1, 1)
        return x * y


class SpatialAttentionCBAM(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionCBAM, self).__init__()

        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


class SSAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(SSAM, self).__init__()
        self.channel_attention = ChannelAttentionCBAM(in_channels, reduction_ratio)
        self.SE = SEBlock(in_channels, reduction_ratio=16)
        self.spatial_attention = SpatialAttentionCBAM(kernel_size)

    def forward(self, x):
        se_out = x * self.channel_attention(x)
        sa_out = x * self.spatial_attention(x)
        return torch.add(se_out, sa_out)



"""
======================================================================================================
"""
from functools import partial
from timm.models.helpers import named_apply


def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


class MGAG(nn.Module):
    def __init__(self, F_g, F_int, kernel_size=3, groups=1, activation='gelu'):
        super(MGAG, self).__init__()

        if kernel_size == 1:
            groups = 1
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=kernel_size, stride=1, padding=1, groups=groups,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, groups=groups,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.activation = act_layer(activation, inplace=True)

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        g1 = self.W_g(x)
        x1 = self.W_x(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)
        x = x * psi
        # x = x * self.channel(x)

        return x


"""
======================================================================================================
"""
def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class DWConv2d(nn.Module):

    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        '''
        x = x.permute(0, 3, 1, 2)  # (b c h w)
        x = self.conv(x)  # (b c h w)
        x = x.permute(0, 2, 3, 1)  # (b h w c)
        return x


class Layernorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        '''
        x = x.permute(0, 2, 3, 1)  # (b h w c)
        x = self.ln(x)  # (b c h w)
        x = x.permute(0, 3, 1, 2)  # (b c h w)
        return x


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)
        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y


"""========================================================================================="""


class Mambabackbone(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(60, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width)) for i in range(2, block_units[0] + 1)] +
                [('unit_pm', BiPixelMambaLayer(dim=width * 4, p=4))] +
                [('win_mamba', BiWindowMambaLayer(dim=width*4, p=1))] +
                [('unit_f', PreActBottleneck(cin=width * 4, cout=width * 4, cmid=width))] +
                [('ssam', SSAM(in_channels=width*4))]
                ,
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2)) for i in range(2, block_units[1] + 1)] +
                [('unit_pm', BiPixelMambaLayer(dim=width * 8, p=4))] +
                [('win_mamba', BiWindowMambaLayer(dim=width*8, p=1))] +
                [('unit_f', PreActBottleneck(cin=width * 8, cout=width * 8, cmid=width))] +
                [('ssam', SSAM(in_channels=width*8))]
                ,
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) for i in range(2, block_units[2] + 1)] +
                [('unit_pm', BiPixelMambaLayer(dim=width * 16, p=4))] +
                [('win_mamba', BiWindowMambaLayer(dim=width*16, p=1))] +
                [('unit_f', PreActBottleneck(cin=width * 16, cout=width * 16, cmid=width))] +
                [('ssam', SSAM(in_channels=width*16))]
                ,
                ))),
            # ('retention', retentionlayer(embed_dim=width*16, num_heads=4, init_value=2, heads_range=4)) # v2
            # ('unit_pm', BiPixelMambaLayer(dim=width * 16, p=4))
        ]))

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)(x) # 3
        for i in range(len(self.body)-1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i+1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1]


logger = logging.getLogger(__name__)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:   # ResNet
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = Mambabackbone(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)
        self.mamba = Mamba(
            d_model=config.hidden_size,  # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor
            use_fast_path=False,
        )

    def forward(self, x):

        h = x
        x = self.mamba(x)
        x = self.ffn_norm(x)
        x = x + h

        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class Classifier(nn.Module):
    def __init__(self, config, in_channels, hiden_channels, num_classes):
        super(Classifier, self).__init__()

        head_channels = 384
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
        self.fc1 = nn.Linear(in_channels, hiden_channels)
        self.fc2 = nn.Linear(hiden_channels, num_classes)
        self.dropout = Dropout(0.5)

    def forward(self, x):
        B, n_patch, hidden = x.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x


class MambaVisionTransformer(nn.Module):
    def __init__(self, config, img_size=256, num_classes=3, zero_head=False, vis=False):
        super(MambaVisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head

        self.transformer = Transformer(config, img_size, vis)

        self.classifier = Classifier(
            config,
            in_channels=384,
            hiden_channels=768,
            num_classes=num_classes,
        )
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        logits = self.classifier(x)
        # logits = x

        return logits


CONFIGS = {
    'ViT-B_16': get_b16_config(),
    'ViT-B_32': get_b32_config(),
    'ViT-L_16': get_l16_config(),
    'ViT-L_32': get_l32_config(),
    'ViT-H_14': get_h14_config(),
    'R50-ViT-B_16': get_r50_b16_config(),
    'R50-ViT-L_16': get_r50_l16_config(),
    'testing': get_testing(),
}


@torch.no_grad()
def throughput(model):
    model.eval()

    model.cuda()

    images = torch.randn(1, 60, 256, 256).cuda()

    batch_size = images.shape[0]
    for i in range(50):
        model(images)
    torch.cuda.synchronize()
    print(f"throughput averaged with 30 times")
    tic1 = time.time()
    for i in range(30):
        model(images)
    torch.cuda.synchronize()
    tic2 = time.time()
    print(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
    return


def compute_para():
    total = sum([param.nelement() for param in model.parameters() if param.requires_grad])

    model.eval()
    input = torch.rand(1, 60, 256, 256)
    flops = FlopCountAnalysis(model, input.to("cuda:0"))
    print(flop_count_table(flops))
    print("Number of parameter: %.4fM" % (total / 1e6))

    # throughput(model)

    exit()


if __name__ == "__main__":
    model = MambaVisionTransformer(config=CONFIGS['R50-ViT-B_16'], img_size=256, num_classes=2)
    model.to("cuda:1")
    print(model)

    input = torch.randn(4, 60, 256, 256).to("cuda:1")
    output = model(input)
    print(output.size())

    # compute_para()



