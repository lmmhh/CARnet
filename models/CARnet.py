#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
Introduce Unet structure to supplement context information
"""
import torch
from torch import nn
import numpy as np
import cv2
import torch.nn.functional as F


def get_model_spec():
    model = Generator()
    print('# of parameters: ', sum([p.numel() for p in model.parameters()]))
    return model


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.attention = ChannelAttentionBlock()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.PReLU(init=0.1)
        )

        self.block2 = ResidualBlock(32)
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.PReLU(init=0.1)
        )
        self.block4 = ResidualBlock(64)
        self.block5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.PReLU(init=0.1)
        )
        self.block6 = ResidualBlock(128)
        self.block7 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.PReLU(init=0.1)
        )
        self.block8 = ResidualBlock(64)
        self.block9 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.PReLU(init=0.1)
        )
        self.block10 = ResidualBlock(32)
        self.block11 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.PReLU(init=0.1)
        )

        self.blockadd1 = ResidualBlock(64)

        self.blockadd2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.PReLU(init=0.1)
        )

        self.blockadd3 = ResidualBlock(128)

        self.blockadd4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.PReLU(init=0.1)
        )

        self.blockadd5 = ResidualBlock(64)

        self.blockadd6 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.PReLU(init=0.1)
        )

        self.block12 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.PReLU(init=0.1),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.PReLU(init=0.1),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        a32 = self.attention(x)
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block7)
        block9 = self.block9(block8)
        block10 = self.block10(block9)
        block11 = self.block11(block10)
        blockadd1 = self.blockadd1(torch.cat([block11, a32], dim=1))
        blockadd2 = self.blockadd2(blockadd1)
        blockadd3 = self.blockadd3(blockadd2)
        blockadd4 = self.blockadd4(blockadd3)
        blockadd5 = self.blockadd5(blockadd4)
        blockadd6 = self.blockadd6(blockadd5)
        block12 = self.block12(block1 + blockadd6)

        return block12


class ChannelAttentionBlock(nn.Module):
    def __init__(self):
        super(ChannelAttentionBlock, self).__init__()
        self.convG = GradientLayer()
        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=3),
            nn.PReLU(init=0.1),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.PReLU(init=0.1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.PReLU(init=0.1),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.PReLU(init=0.1),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.PReLU(init=0.1),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.PReLU(init=0.1),
        )
        self.diconv1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 2, dilation=2),
            nn.PReLU(init=0.1)
        )
        self.diconv2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 4, dilation=4),
            nn.PReLU(init=0.1)
        )
        self.diconv3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 8, dilation=8),
            nn.PReLU(init=0.1)
        )

        self.deconv1 = UpsampleBLock(128, 2)
        self.conv6 = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            nn.PReLU(init=0.1)
        )
        self.deconv2 = UpsampleBLock(64, 2)
        self.conv7 = nn.Sequential(
            nn.Conv2d(32, 32, 1),
            nn.PReLU(init=0.1)
        )
        self.deconv3 = UpsampleBLock(32, 2)
        self.conv8 = nn.Sequential(
            nn.Conv2d(16, 32, 1),
            nn.PReLU(init=0.1)
        )

    def forward(self, x):
        a, v = self.convG(x)
        x = torch.cat([x, a, v], dim=1)
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.diconv1(x)
        x = self.diconv2(x)
        x = self.diconv3(x)
        x = self.deconv1(x)
        x = self.conv6(x)
        x = self.deconv2(x)
        x = self.conv7(x)
        x = self.deconv3(x)
        x = self.conv8(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2 // 2,
                              kernel_size=1)  # V0  kernel_size=3 looks like no influence
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class GradientLayer(nn.Module):
    def __init__(self):
        super(GradientLayer, self).__init__()

    def forward(self, x):
        x = x.cpu().detach().numpy()
        value = np.zeros_like(x)
        angle = np.zeros_like(x)
        batch = x.shape[0]
        for i in range(batch):
            gx = cv2.Sobel(x[i][0], cv2.CV_32F, 1, 0, ksize=-1)
            gy = cv2.Sobel(x[i][0], cv2.CV_32F, 0, 1, ksize=-1)
            value[i][0], angle[i][0] = cv2.cartToPolar(gx, gy, angleInDegrees=False)
        value = torch.from_numpy(value)
        angle = torch.from_numpy(angle)
        return value.cuda(), angle.cuda()
