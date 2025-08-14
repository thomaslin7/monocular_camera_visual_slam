import os
import glob
import json
import h5py
import random

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models

from sklearn.metrics import mean_absolute_error, mean_squared_error

#ssim is used for the depth loss function, which is hybrid of l1 loss and ssim
#ssim provides better detail around the edges and better object boundaries
from pytorch_msssim import ssim

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

#import model
vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

#break encoder up to implement skip connections to help preserve info
class VGG16Encoder(nn.Module):
    def __init__(self):
        super(VGG16Encoder, self).__init__()

        #gets the model up to before the fc layer(s)
        features = vgg16.features

        #vgg16 blocks
        self.block1 = features[:5]      #conv1
        self.block2 = features[5:10]    #conv2
        self.block3 = features[10:17]   #conv3
        self.block4 = features[17:24]   #conv4
        self.block5 = features[24:31]   #conv5 -> 512 channels
        
        #freeze first 2 blocks to help speed up training and improve stability
        for param in self.block1.parameters():
            param.requires_grad = False
        for param in self.block2.parameters():
            param.requires_grad = False

        #last 3 blocks stay unfrozen to help adapt encoder to dataset (unfrozen by deafult); helps w fine-tuning

    #(channel, w, h)
    def forward(self, x):
        x1 = self.block1(x) # [64, 112, 112]
        x2 = self.block2(x1) # [128, 56, 56]
        x3 = self.block3(x2) # [256, 28, 28]
        x4 = self.block4(x3) # [512, 14, 14]
        x5 = self.block5(x4) # [512, 7, 7]
        return x1, x2, x3, x4, x5


#decoder w skip connections
class DecoderUNet(nn.Module):
    def __init__(self):
        super(DecoderUNet, self).__init__()

        # upsampling blocks: in_ch accounts for concat (upsampled + skip)
        self.up1 = self.upSamp(512, 512)   # 7x7 -> 14x14
        #double channels b/c of concatenate decoder + skip
        self.up2 = self.upSamp(512 + 512, 256)   # 14x14 -> 28x28
        self.up3 = self.upSamp(256 + 256, 128)   # 28x28 -> 56x56
        self.up4 = self.upSamp(128 + 128, 64)    # 56x56 -> 112x112
        self.final_up = nn.Sequential(              # 112x112 -> 224x224
            nn.ConvTranspose2d(64 + 64, 1, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.Sigmoid()
        )

    #chIn: channels in, chOut: channels out after upsampling
    def upSamp(self, chIn, chOut):
        return nn.Sequential(
            #k=3 and s=2 doubles the h,w 
            nn.ConvTranspose2d(chIn, chOut, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(chOut),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3)
        )

    def forward(self, x1, x2, x3, x4, x5):
        d1 = self.up1(x5)
        #concat skip and decoder
        d1 = torch.cat([d1, x4], dim=1)  # 512 + 512 

        d2 = self.up2(d1)
        d2 = torch.cat([d2, x3], dim=1)  # 256 + 256

        d3 = self.up3(d2)
        d3 = torch.cat([d3, x2], dim=1)  # 128 + 128

        d4 = self.up4(d3)
        d4 = torch.cat([d4, x1], dim=1)  # 64 + 64

        out = self.final_up(d4)
        return out

#combine encoder and decoder
class DepthEstimationUNet(nn.Module):
    def __init__(self):
        super(DepthEstimationUNet, self).__init__()
        self.encoder = VGG16Encoder()
        self.decoder = DecoderUNet()

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)
        out = self.decoder(x1, x2, x3, x4, x5)
        return out
