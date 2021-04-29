import sys
sys.path.append('..')
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.lib import SynchronizedBatchNorm2d
BatchNorm2d = SynchronizedBatchNorm2d



class DoubleConv2D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, norm_layer=None):
        super(DoubleConv2D,self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = norm_layer
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

#-------------------------------------------

class Down2D(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, conv_builder, norm_layer=None):
        super(Down2D,self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            conv_builder(in_channels, out_channels, norm_layer=norm_layer)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


#-------------------------------------------

class SimpleNet(nn.Module):
    def __init__(self, down, width, conv_builder, n_channels=1, norm_layer=BatchNorm2d):
        super(SimpleNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = norm_layer

        self.n_channels = n_channels
        self.width = width

        self.inc = nn.Sequential(
            nn.Conv2d(self.n_channels, width[0], kernel_size=3, padding=1),
            norm_layer(width[0]),
            nn.ReLU(inplace=True)
        )
        self.down1 = down(width[0], width[1], conv_builder, norm_layer)
        self.down2 = down(width[1], width[2], conv_builder, norm_layer)
        self.down3 = down(width[2], width[3], conv_builder, norm_layer)
        self.down4 = down(width[3], width[4], conv_builder, norm_layer)
       

    def forward(self, x):
        out_x = []
        x1 = self.inc(x)
        out_x.append(x1)
        x2 = self.down1(x1)
        out_x.append(x2)
        x3 = self.down2(x2)
        out_x.append(x3)
        x4 = self.down3(x3)
        out_x.append(x4)
        x5 = self.down4(x4)
        out_x.append(x5)

        return out_x



def simplenet(**kwargs):
    return SimpleNet(down=Down2D,
                    width=[32,64,128,256,512],
                    conv_builder=DoubleConv2D,
                    **kwargs)
