# encoding: utf-8
# @Time    : 2023/11/26 下午5:41
# @Author  : Geng Qin
# @File    : get_networks.py
import sys
# from model.Dinov2 import *
from model.transunet import *
from model.H_vmunet import *
from model.segformer import *
from model.FDnet import *
from model.ssm_tran import *
from model.v1 import *
from model.EMCAD import EMCADNet
from model.swin_unet import swinunet
from model.unet import UNet
from model.wavesnet import *
from model.nnunet import *
from model.resunet import *
from model.MADGNet import MFMSNet
# from model.BEFUnet import *


def get_network(network, num_classes, **kwargs):

    # 2d networks
    if network == 'unet':
        net = UNet(n_channels=60, n_classes=num_classes)
    elif network == 'transunet':
        net = VisionTransformer(config=CONFIGS['R50-ViT-B_16'], img_size=256, num_classes=2)
    elif network == 'H-vmunet':
        net = H_vmunet(num_classes=num_classes,
                     input_channels=60,
                     c_list=[8, 16, 32, 64, 128, 256],
                     split_att='fc',
                     bridge=True,
                     drop_path_rate=0.4)
    elif network == 'segformer':
        net = SegFormer2d(in_channels=60, num_classes=num_classes)
    elif network == 'FDnet':
        net = FD_Net()
    elif network == 'EMCAD':
        net = EMCADNet(num_classes=num_classes)
    elif network == 'wavesnet':
        net = wsegnet_vgg16_bn(in_channels=60, num_classes=num_classes)
    elif network == "swinunet":
        net = swinunet(2, 256)
    elif network == 'nnunet':
        net = initialize_network(threeD=False)
    elif network == 'resunet':
        net = res_unet(60, 2)
    # elif network == 'BEFUnet':
    #     net = BEFUnet()
    elif network == "MADGNet":
        net = MFMSNet()
    elif network == 'v1':
        net = MambaVisionTransformer(config=CONFIGS['R50-ViT-B_16'], img_size=256, num_classes=2)

    else:
        print('the network you have entered is not supported yet')
        sys.exit()
    return net
