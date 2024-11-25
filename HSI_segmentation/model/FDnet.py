# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class CALayer(nn.Module):
    def __init__(self, in_ch, reduction=8):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_1 = nn.Conv2d(in_ch, in_ch // reduction, kernel_size=1, stride=1)
        self.conv_2 = nn.Conv2d(in_ch // reduction, in_ch, kernel_size=1, stride=1)
        self.relu = nn.ReLU()
        self.sigmiod = nn.Sigmoid()

    def forward(self, input):
        y = self.avg_pool(input)
        y = self.sigmiod(self.conv_2(self.relu(self.conv_1(y))))
        return input * y

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class FDBlock_1th(nn.Module):
    def __init__(self, in_ch=60, out_ch=8, ):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 8, kernel_size=3, stride=1, padding=1)
        self.BN = nn.BatchNorm2d(out_ch * 8)
        self.relu = nn.ReLU()
        self.conv_1 = nn.Conv2d(out_ch * 8, out_ch * 8, kernel_size=3, stride=1, padding=1)
        self.BN_1 = nn.BatchNorm2d(out_ch * 8)
        self.conv_2 = nn.Conv2d(out_ch * 8, out_ch * 8, kernel_size=3, stride=1, padding=1)
        self.BN_2 = nn.BatchNorm2d(out_ch * 8)
        self.conv_3 = nn.Conv2d(out_ch * 8, out_ch * 8, kernel_size=3, stride=1, padding=1)
        self.BN_3 = nn.BatchNorm2d(out_ch * 8)
        self.conv_4 = nn.Conv2d(out_ch * 6, out_ch * 8, kernel_size=3, stride=1, padding=1)
        self.BN_4 = nn.BatchNorm2d(out_ch * 8)
        self.conv_5 = nn.Conv2d(out_ch * 8, out_ch * 8, kernel_size=3, stride=1, padding=1)
        self.BN_5 = nn.BatchNorm2d(out_ch * 8)
        self.conv_6 = nn.Conv2d(out_ch * 8, out_ch * 10, kernel_size=3, stride=1, padding=1)
        self.BN_6 = nn.BatchNorm2d(out_ch * 10)
        self.ca = CALayer(out_ch * 10)
        self.conv_end = nn.Conv2d(out_ch * 10, out_ch * 8, kernel_size=1, stride=1)
        self.BN_end = nn.BatchNorm2d(out_ch * 8)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, input):
        conv = self.relu(self.BN(self.conv(input)))  
        conv_1 = self.relu(self.BN_1(self.conv_1(conv)))
        conv_2 = self.relu(self.BN_2(self.conv_2(conv_1)))
        conv_3 = self.relu(self.BN_3(self.conv_3(conv_2)))     
        conv_3_up = conv_3[:, :16, :, :]
        conv_3_down = conv_3[:, 16:, :, :]
        conv_4 = self.relu(self.BN_4(self.conv_4(conv_3_down))) 
        conv_5 = self.relu(self.BN_5(self.conv_5(conv_4)))
        conv_6 = self.relu(self.BN_6(self.conv_6(conv_5)))
        cat = torch.cat([conv, conv_3_up], dim=1)      
        conv_6_in = cat + conv_6
        ca = self.ca(conv_6_in)
        conv_end = self.relu(self.BN_end(self.conv_end(ca)))

        down_1 = self.downsample(conv_end)
        return conv_end, down_1

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class FDBlock_2th(nn.Module):
    def __init__(self, out_ch=8):
        super().__init__()
        self.conv = nn.Conv2d(out_ch * 8, out_ch * 16, kernel_size=3, stride=1, padding=1)
        self.BN = nn.BatchNorm2d(out_ch * 16)
        self.relu = nn.ReLU()
        self.conv_1 = nn.Conv2d(out_ch * 16, out_ch * 16, kernel_size=3, stride=1, padding=1)
        self.BN_1 = nn.BatchNorm2d(out_ch * 16)
        self.conv_2 = nn.Conv2d(out_ch * 16, out_ch * 16, kernel_size=3, stride=1, padding=1)
        self.BN_2 = nn.BatchNorm2d(out_ch * 16)
        self.conv_3 = nn.Conv2d(out_ch * 16, out_ch * 16, kernel_size=3, stride=1, padding=1)
        self.BN_3 = nn.BatchNorm2d(out_ch * 16)
        self.conv_4 = nn.Conv2d(out_ch * 12, out_ch * 16, kernel_size=3, stride=1, padding=1)
        self.BN_4 = nn.BatchNorm2d(out_ch * 16)
        self.conv_5 = nn.Conv2d(out_ch * 16, out_ch * 16, kernel_size=3, stride=1, padding=1)
        self.BN_5 = nn.BatchNorm2d(out_ch * 16)
        self.conv_6 = nn.Conv2d(out_ch * 16, out_ch * 20, kernel_size=3, stride=1, padding=1)
        self.BN_6 = nn.BatchNorm2d(out_ch * 20)
        self.ca = CALayer(out_ch * 20)
        self.conv_end = nn.Conv2d(out_ch * 20, out_ch * 16, kernel_size=1, stride=1)
        self.BN_end = nn.BatchNorm2d(out_ch * 16)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, input):
        conv = self.relu(self.BN(self.conv(input)))  
        conv_1 = self.relu(self.BN_1(self.conv_1(conv)))
        conv_2 = self.relu(self.BN_2(self.conv_2(conv_1)))
        conv_3 = self.relu(self.BN_3(self.conv_3(conv_2)))     
        conv_3_up = conv_3[:, :32, :, :]
        conv_3_down = conv_3[:, 32:, :, :]
        conv_4 = self.relu(self.BN_4(self.conv_4(conv_3_down))) 
        conv_5 = self.relu(self.BN_5(self.conv_5(conv_4)))
        conv_6 = self.relu(self.BN_6(self.conv_6(conv_5)))
        cat = torch.cat([conv, conv_3_up], dim=1)      
        conv_6_in = cat + conv_6
        ca = self.ca(conv_6_in)
        conv_end = self.relu(self.BN_end(self.conv_end(ca)))

        down_2 = self.downsample(conv_end)
        return conv_end, down_2

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class FDBlock_3th(nn.Module):
    def __init__(self, out_ch=8):
        super().__init__()
        self.conv = nn.Conv2d(out_ch * 16, out_ch * 32, kernel_size=3, stride=1, padding=1)
        self.BN = nn.BatchNorm2d(out_ch * 32)
        self.relu = nn.ReLU()
        self.conv_1 = nn.Conv2d(out_ch * 32, out_ch * 32, kernel_size=3, stride=1, padding=1)
        self.BN_1 = nn.BatchNorm2d(out_ch * 32)
        self.conv_2 = nn.Conv2d(out_ch * 32, out_ch * 32, kernel_size=3, stride=1, padding=1)
        self.BN_2 = nn.BatchNorm2d(out_ch * 32)
        self.conv_3 = nn.Conv2d(out_ch * 32, out_ch * 32, kernel_size=3, stride=1, padding=1)
        self.BN_3 = nn.BatchNorm2d(out_ch * 32)
        self.conv_4 = nn.Conv2d(out_ch * 24, out_ch * 32, kernel_size=3, stride=1, padding=1)
        self.BN_4 = nn.BatchNorm2d(out_ch * 32)
        self.conv_5 = nn.Conv2d(out_ch * 32, out_ch * 32, kernel_size=3, stride=1, padding=1)
        self.BN_5 = nn.BatchNorm2d(out_ch * 32)
        self.conv_6 = nn.Conv2d(out_ch * 32, out_ch * 40, kernel_size=3, stride=1, padding=1)
        self.BN_6 = nn.BatchNorm2d(out_ch * 40)
        self.ca = CALayer(out_ch * 40)
        self.conv_end = nn.Conv2d(out_ch * 40, out_ch * 32, kernel_size=1, stride=1)
        self.BN_end = nn.BatchNorm2d(out_ch * 32)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, input):
        conv = self.relu(self.BN(self.conv(input)))  
        conv_1 = self.relu(self.BN_1(self.conv_1(conv)))
        conv_2 = self.relu(self.BN_2(self.conv_2(conv_1)))
        conv_3 = self.relu(self.BN_3(self.conv_3(conv_2)))     
        conv_3_up = conv_3[:, :64, :, :]
        conv_3_down = conv_3[:, 64:, :, :]
        conv_4 = self.relu(self.BN_4(self.conv_4(conv_3_down))) 
        conv_5 = self.relu(self.BN_5(self.conv_5(conv_4)))
        conv_6 = self.relu(self.BN_6(self.conv_6(conv_5)))
        cat = torch.cat([conv, conv_3_up], dim=1)      
        conv_6_in = cat + conv_6
        ca = self.ca(conv_6_in)
        conv_end = self.relu(self.BN_end(self.conv_end(ca)))

        down_3 = self.downsample(conv_end)
        return conv_end, down_3

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class FDBlock_4th(nn.Module):
    def __init__(self, out_ch=8):
        super().__init__()
        self.conv = nn.Conv2d(out_ch * 32, out_ch * 64, kernel_size=3, stride=1, padding=1)
        self.BN = nn.BatchNorm2d(out_ch * 64)
        self.relu = nn.ReLU()
        self.conv_1 = nn.Conv2d(out_ch * 64, out_ch * 64, kernel_size=3, stride=1, padding=1)
        self.BN_1 = nn.BatchNorm2d(out_ch * 64)
        self.conv_2 = nn.Conv2d(out_ch * 64, out_ch * 64, kernel_size=3, stride=1, padding=1)
        self.BN_2 = nn.BatchNorm2d(out_ch * 64)
        self.conv_3 = nn.Conv2d(out_ch * 64, out_ch * 64, kernel_size=3, stride=1, padding=1)
        self.BN_3 = nn.BatchNorm2d(out_ch * 64)
        self.conv_4 = nn.Conv2d(out_ch * 48, out_ch * 64, kernel_size=3, stride=1, padding=1)
        self.BN_4 = nn.BatchNorm2d(out_ch * 64)
        self.conv_5 = nn.Conv2d(out_ch * 64, out_ch * 64, kernel_size=3, stride=1, padding=1)
        self.BN_5 = nn.BatchNorm2d(out_ch * 64)
        self.conv_6 = nn.Conv2d(out_ch * 64, out_ch * 80, kernel_size=3, stride=1, padding=1)
        self.BN_6 = nn.BatchNorm2d(out_ch * 80)
        self.ca = CALayer(out_ch * 80)
        self.conv_end = nn.Conv2d(out_ch * 80, out_ch * 64, kernel_size=1, stride=1)
        self.BN_end = nn.BatchNorm2d(out_ch * 64)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, input):
        conv = self.relu(self.BN(self.conv(input)))  
        conv_1 = self.relu(self.BN_1(self.conv_1(conv)))
        conv_2 = self.relu(self.BN_2(self.conv_2(conv_1)))
        conv_3 = self.relu(self.BN_3(self.conv_3(conv_2)))     
        conv_3_up = conv_3[:, :128, :, :]
        conv_3_down = conv_3[:, 128:, :, :]
        conv_4 = self.relu(self.BN_4(self.conv_4(conv_3_down))) 
        conv_5 = self.relu(self.BN_5(self.conv_5(conv_4)))
        conv_6 = self.relu(self.BN_6(self.conv_6(conv_5)))
        cat = torch.cat([conv, conv_3_up], dim=1)      
        conv_6_in = cat + conv_6
        ca = self.ca(conv_6_in)
        conv_end = self.relu(self.BN_end(self.conv_end(ca)))

        down_4 = self.downsample(conv_end)
        return conv_end, down_4

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class FDBlock_end(nn.Module):
    def __init__(self, out_ch=8):
        super().__init__()
        self.conv = nn.Conv2d(out_ch * 64, out_ch * 128, kernel_size=3, stride=1, padding=1)
        self.BN = nn.BatchNorm2d(out_ch * 128)
        self.relu = nn.ReLU()
        self.conv_1 = nn.Conv2d(out_ch * 128, out_ch * 128, kernel_size=3, stride=1, padding=1)
        self.BN_1 = nn.BatchNorm2d(out_ch * 128)
        self.conv_2 = nn.Conv2d(out_ch * 128, out_ch * 128, kernel_size=3, stride=1, padding=1)
        self.BN_2 = nn.BatchNorm2d(out_ch * 128)
        self.conv_3 = nn.Conv2d(out_ch * 128, out_ch * 128, kernel_size=3, stride=1, padding=1)
        self.BN_3 = nn.BatchNorm2d(out_ch * 128)
        self.conv_4 = nn.Conv2d(out_ch * 96, out_ch * 128, kernel_size=3, stride=1, padding=1)
        self.BN_4 = nn.BatchNorm2d(out_ch * 128)
        self.conv_5 = nn.Conv2d(out_ch * 128, out_ch * 128, kernel_size=3, stride=1, padding=1)
        self.BN_5 = nn.BatchNorm2d(out_ch * 128)
        self.conv_6 = nn.Conv2d(out_ch * 128, out_ch * 160, kernel_size=3, stride=1, padding=1)
        self.BN_6 = nn.BatchNorm2d(out_ch * 160)
        self.ca = CALayer(out_ch * 160)
        self.conv_end = nn.Conv2d(out_ch * 160, out_ch * 128, kernel_size=1, stride=1)
        self.BN_end = nn.BatchNorm2d(out_ch * 128)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, input):
        conv = self.relu(self.BN(self.conv(input)))  
        conv_1 = self.relu(self.BN_1(self.conv_1(conv)))
        conv_2 = self.relu(self.BN_2(self.conv_2(conv_1)))
        conv_3 = self.relu(self.BN_3(self.conv_3(conv_2)))     
        conv_3_up = conv_3[:, :256, :, :]
        conv_3_down = conv_3[:, 256:, :, :]
        conv_4 = self.relu(self.BN_4(self.conv_4(conv_3_down))) 
        conv_5 = self.relu(self.BN_5(self.conv_5(conv_4)))
        conv_6 = self.relu(self.BN_6(self.conv_6(conv_5)))
        cat = torch.cat([conv, conv_3_up], dim=1)      
        conv_6_in = cat + conv_6
        ca = self.ca(conv_6_in)
        conv_end = self.relu(self.BN_end(self.conv_end(ca)))

        return conv_end

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Decoder_1th(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.conv_1 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.BN_1 = nn.BatchNorm2d(num_features=512)
        self.relu = nn.ReLU()
        self.conv_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.BN_2 = nn.BatchNorm2d(num_features=512)
    def forward(self, input, map):
        up = self.up(input)
        cat = torch.cat((up, map), dim=1)
        conv_1 = self.relu(self.BN_1(self.conv_1(cat)))
        conv_2 = self.relu(self.BN_2(self.conv_2(conv_1)))
        return conv_2

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Decoder_2th(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.conv_1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.BN_1 = nn.BatchNorm2d(num_features=256)
        self.relu = nn.ReLU()
        self.conv_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.BN_2 = nn.BatchNorm2d(num_features=256)
    def forward(self, input, map):
        up = self.up(input)
        cat = torch.cat((up, map), dim=1)
        conv_1 = self.relu(self.BN_1(self.conv_1(cat)))
        conv_2 = self.relu(self.BN_2(self.conv_2(conv_1)))
        return conv_2

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Decoder_3th(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.conv_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.BN_1 = nn.BatchNorm2d(num_features=128)
        self.relu = nn.ReLU()
        self.conv_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.BN_2 = nn.BatchNorm2d(num_features=128)
    def forward(self, input, map):
        up = self.up(input)
        cat = torch.cat((up, map), dim=1)
        conv_1 = self.relu(self.BN_1(self.conv_1(cat)))
        conv_2 = self.relu(self.BN_2(self.conv_2(conv_1)))
        return conv_2

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Decoder_4th(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.conv_1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.BN_1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU()
        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.BN_2 = nn.BatchNorm2d(num_features=64)
        self.conv = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, stride=1, padding=0)

    def forward(self, input, map):
        up = self.up(input)
        cat = torch.cat((up, map), dim=1)
        conv_1 = self.relu(self.BN_1(self.conv_1(cat)))
        conv_2 = self.relu(self.BN_2(self.conv_2(conv_1)))
        conv = self.conv(conv_2)
        return conv

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class FD_Net(nn.Module):
    def __init__(self):
        super(FD_Net, self).__init__()
        
        self.FDBlock_1 = FDBlock_1th()
        self.FDBlock_2 = FDBlock_2th()
        self.FDBlock_3 = FDBlock_3th()
        self.FDBlock_4 = FDBlock_4th()
        self.FDBlock_end = FDBlock_end()
        self.Decoder_1 = Decoder_1th()
        self.Decoder_2 = Decoder_2th()
        self.Decoder_3 = Decoder_3th()
        self.Decoder_4 = Decoder_4th()

    def forward(self, input):

        encoder1, down1 = self.FDBlock_1(input)
        encoder2, down2 = self.FDBlock_2(down1)
        encoder3, down3 = self.FDBlock_3(down2)
        encoder4, down4 = self.FDBlock_4(down3)
        encoder5 = self.FDBlock_end(down4)
        decoder1 = self.Decoder_1(encoder5, encoder4)
        decoder2 = self.Decoder_2(decoder1, encoder3)
        decoder3 = self.Decoder_3(decoder2, encoder2)
        decoder4 = self.Decoder_4(decoder3, encoder1)

        return decoder4


if __name__ == "__main__":
    x = torch.ones(1, 60, 256, 256)
    x = x.to(0)
    model = FD_Net().to(0)
    output = model(x)
    print(output.size())

