# encoding: utf-8
# @Time    : 2023/11/26 下午5:41
# @Author  : Geng Qin
# @File    : get_networks.py
import sys
from cls_model.CMT import CmtB
from cls_model.CoAtNet import coatnet_0
from cls_model.ConvNeXt import convnext_base
from cls_model.CSwin import CSWin_96_24322_base_224
from cls_model.DeiT import deit_base_distilled_patch16_224
from cls_model.Focal_Transformer import FocalTransformer
from cls_model.MPViT import mpvit_xsmall
from cls_model.PVT import pvt_medium
from cls_model.ResNet import resnet50
from cls_model.STViT import STViT
from cls_model.Swin_transformer import swin_base_patch4_window7_224
from cls_model.UniFormer import uniformer_base
from cls_model.ViT import vit_base_patch16_224_in21k
from cls_model.WiKG import WiKG
from cls_model.v1 import *


def get_network(network, num_classes, **kwargs):

    # 2d networks
    if network == 'cmt':
        net = CmtB(num_classes=num_classes)
    elif network == 'coat':
        net = coatnet_0(img_size=256, in_channel=40, num_classes=num_classes)
    elif network == 'convnext':
        net = convnext_base(num_classes=num_classes)
    elif network == 'cswin':
        net = CSWin_96_24322_base_224()
    elif network == 'deit':
        net = deit_base_distilled_patch16_224()
    elif network == 'focal':
        net = FocalTransformer(in_chans=40, num_classes=num_classes,
                             img_size=256, embed_dim=128, depths=[2, 2, 18, 2], drop_path_rate=0.2,
                             focal_levels=[2, 2, 2, 2], expand_sizes=[3, 3, 3, 3], expand_layer="all",
                             num_heads=[4, 8, 16, 32],
                             focal_windows=[7, 5, 3, 1],
                             window_size=7,
                             use_conv_embed=True,
                             use_shift=False)
    elif network == 'mpvit':
        net = mpvit_xsmall()
    elif network == 'pvt':
        net = pvt_medium()
    elif network == "resnet":
        net = resnet50(num_classes=num_classes)
    elif network == 'stvit':
        net = STViT(
            embed_dim=[96, 192, 448, 640],  # 95M, 15.6G, 269 FPS
            depths=[4, 7, 19, 8],
            num_heads=[2, 3, 7, 10],
            n_iter=[1, 1, 1, 1],
            stoken_size=[16, 8, 2, 1],
            projection=1024,
            mlp_ratio=4,
            stoken_refine=True,
            stoken_refine_attention=True,
            hard_label=False,
            rpe=False,
            qkv_bias=True,
            qk_scale=None,
            use_checkpoint=False,
            checkpoint_num=[0, 0, 0, 0],
            layerscale=[False] * 4,
            init_values=1e-6, )

    elif network == 'swin_transformer':
        net = swin_base_patch4_window7_224(num_classes=num_classes)
    elif network == 'uniformer':
        net = uniformer_base()
    elif network == "vit":
        net = vit_base_patch16_224_in21k(num_classes=num_classes)
    elif network == 'wikg':
        net = WiKG(dim_in=384, dim_hidden=512, topk=6, n_classes=num_classes, agg_type='bi-interaction', dropout=0.3,
                 pool='attn')
    elif network == 'v1':
        net = MambaVisionTransformer(config=CONFIGS['R50-ViT-B_16'], img_size=256, num_classes=num_classes)

    else:
        print('the network you have entered is not supported yet')
        sys.exit()
    return net
