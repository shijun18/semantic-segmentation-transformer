from typing import Optional, Union, List

import torch.nn as nn
from decoder.unet import UnetDecoder
from get_encoder import build_encoder
from base_model import SegmentationModel

from lib import SynchronizedBatchNorm2d
BatchNorm2d = SynchronizedBatchNorm2d


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class Unet(SegmentationModel):
    """Unet is a fully convolution neural network for image semantic segmentation. Consist of *encoder* 
    and *decoder* parts connected with *skip connections*. Encoder extract features of different spatial 
    resolution (skip connections) which are used by decoder to define accurate segmentation mask. Use *concatenation*
    for fusing decoder blocks with skip connections.
    Args:
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and 
            other pretrained weights (see table with available weights for each encoder_name)
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features 
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_channels: List of integers which specify **out_channels** parameter for convolutions used in encoder.
            Length of the list should be the same as **encoder_depth**
        decoder_use_batchnorm: If **True**, BatchNormalization layer between Conv2D and Activation layers is used.
            Available options are **True, False**.
        decoder_attention_type: Attention module used in decoder of the model. Available options are **None** and **scse**.
            SCSE paper - https://arxiv.org/abs/1808.08127
        decoder_channels: List of integers which specify **in_channels** parameter for convolutions used in decoder.
            Length of the list should be the same as **encoder_depth**
        upsampling: Int number of upsampling factor for segmentation head, default=1 
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        aux_classifier: If **True**, add a classification branch based the last feature of the encoder.
            Available options are **True, False**.
    Returns:
        ``torch.nn.Module``: Unet
    """

    def __init__(
        self,
        in_channels: int = 3,
        encoder_name: str = "simplenet",
        encoder_weights: Optional[str] = None,
        encoder_depth: int = 5,
        encoder_channels: List[int] = [32,64,128,256,512],
        decoder_use_batchnorm: bool = True,
        decoder_attention_type: Optional[str] = None,
        decoder_channels: List[int] = (256,128,64,32),
        upsampling: int = 1,
        classes: int = 1,
        aux_classifier: bool = False,
    ):
        super().__init__()

        self.encoder_depth = encoder_depth
        self.encoder_channels = encoder_channels

        self.encoder = build_encoder(
            encoder_name,
            weights=encoder_weights,
            n_channels=in_channels
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder_channels,
            decoder_channels=decoder_channels,
            n_blocks=self.encoder_depth - 1,      # the number of decoder block, = encoder_depth - 1 
            use_batchnorm=decoder_use_batchnorm,
            norm_layer=BatchNorm2d,
            center=False,
            attention_type=decoder_attention_type
        )

        self.segmentation_head = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity(),
            nn.Conv2d(decoder_channels[-1], classes, kernel_size=3, padding=1)
        )

        if aux_classifier:
            self.classification_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Flatten(),
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(self.encoder_channels[-1], classes, bias=True)
            )
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()



if __name__ == '__main__':

    from torchsummary import summary
    import torch
    from config import MODEL_CONFIG
    import os 
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # net = Unet(**MODEL_CONFIG['unet'])
    # net = Unet(**MODEL_CONFIG['swin_trans_unet'])
    net = Unet(**MODEL_CONFIG['resnet18_unet'])

      
    summary(net.cuda(),input_size=(1,512,512),batch_size=1,device='cuda')
    
    # net = net.cuda()
    # net.train()
    # input = torch.randn((1,1,512,512)).cuda()
    # output = net(input)
    # print(output.size())
    

    import sys
    sys.path.append('..')
    from utils import count_params_and_macs
    count_params_and_macs(net.cuda(),(1,1,512,512))