import math
import torch
import numpy as np
from torch import nn
from PIL import JpegPresets
from models import loss

Q = 20

JpegPresets.presets['matlab20'] = {
    'quantization': [
        [
            40,28,25,40,60,100,128,153,
            30,30,35,48,65,145,150,138,
            35,33,40,60,100,143,173,140,
            35,43,55,73,128,218,200,155,
            45,55,93,140,170,255,255,193,
            60,88,138,160,203,255,255,230,
            123,160,195,218,255,255,255,253,
            180,230,238,245,255,250,255,248
        ], [
            40,28,25,40,60,100,128,153,
            30,30,35,48,65,145,150,138,
            35,33,40,60,100,143,173,140,
            35,43,55,73,128,218,200,155,
            45,55,93,140,170,255,255,193,
            60,88,138,160,203,255,255,230,
            123,160,195,218,255,255,255,253,
            180,230,238,245,255,250,255,248
        ]],
    'subsampling': 0
}

JpegPresets.presets['matlab10'] = {
    'quantization': [
        [
            80,55,50,80,120,200,255,255,
            60,60,70,95,130,255,255,255,
            70,65,80,120,200,255,255,255,
            70,85,110,145,255,255,255,255,
            90,110,185,255,255,255,255,255,
            120,175,255,255,255,255,255,255,
            245,255,255,255,255,255,255,255,
            255,255,255,255,255,255,255,255
        ], [
            80,55,50,80,120,200,255,255,
            60,60,70,95,130,255,255,255,
            70,65,80,120,200,255,255,255,
            70,85,110,145,255,255,255,255,
            90,110,185,255,255,255,255,255,
            120,175,255,255,255,255,255,255,
            245,255,255,255,255,255,255,255,
            255,255,255,255,255,255,255,255
        ]],
    'subsampling': 0
}

def get_model_spec():
    model = DMCNN(1, 'matlab' + str(Q))
    print('# of parameters: ', sum([p.numel() for p in model.parameters()]))
    # loss_fn = loss.dmcnnLoss()
    return model


def softRound(x):
    r = torch.round(x).detach_()
    return r + (x - r) ** 3


def initDctCoeff():
    dm = []
    for i in range(8):
        col = []
        for j in range(8):
            if i == 0:
                c = math.sqrt(2) / 4
            else:
                c = 1 / 2
            col.append(c * math.cos(math.pi * (j + 0.5) * i / 8))
        dm.append(col)
    dm = torch.FloatTensor(dm)
    DCT = nn.Parameter(dm, requires_grad=False)
    iDCT = nn.Parameter(dm.t(), requires_grad=False)
    return DCT, iDCT


class dctLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.DCT, self.iDCT = initDctCoeff()

    def forward(self, inputs):
        b, c, w, h = inputs.shape
        nw, nh = w // 8, h // 8

        x = softRound(inputs * 255)
        x = x - 128
        x = x.contiguous().view(b, c, nw, 8, nh, 8)
        x = x.transpose(3, 4)
        x = self.DCT @ x @ self.iDCT
        x = x.transpose(3, 4)
        x = x.contiguous().view(b, c, w, h)

        return x


class iDctLayer(nn.Module):
    def __init__(self, q, alpha=0):
        super().__init__()
        self.DCT, self.iDCT = initDctCoeff()
        self.qtable = nn.Parameter(torch.FloatTensor(np.array(JpegPresets.presets[q]['quantization'][0]).reshape((8, 8))), requires_grad=False)
        self.alpha = nn.Parameter(torch.FloatTensor([alpha]), requires_grad=False)

    def forward(self, inputs, target):
        b, c, w, h = inputs.shape
        nw, nh = w // 8, h // 8

        x = inputs.contiguous().view(b, c, nw, 8, nh, 8)
        x = x.transpose(3, 4)

        # Clamping the DCT coefficients
        target = softRound(target * 255)
        target -= 128
        target = target.contiguous().view(b, c, nw, 8, nh, 8)
        target = target.transpose(3, 4)
        target = self.DCT @ target @ self.iDCT

        low = target - (self.qtable / 2)
        high = target + (self.qtable / 2)

        nx = torch.min(x, high)
        nx = torch.max(nx, low)

        #         from IPython import embed
        #         embed()

        x = (1 - self.alpha) * nx + self.alpha * x

        x = self.iDCT @ x @ self.DCT
        x = x.transpose(3, 4)
        x = x.contiguous().view(b, c, w, h)
        x = x + 128
        x = x / 255

        return x


class dctSubnet(nn.Module):
    def __init__(self, C, q):
        super().__init__()
        self.DCTlayer = dctLayer()
        self.conv0 = nn.Sequential(
            nn.Conv2d(C, 32, 3, 1, 1),
            nn.PReLU(init=0.1)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.PReLU(init=0.1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.PReLU(init=0.1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 2, dilation=2),
            nn.PReLU(init=0.1)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 4, dilation=4),
            nn.PReLU(init=0.1)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 8, dilation=8),
            nn.PReLU(init=0.1)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.PReLU(init=0.1)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.PReLU(init=0.1)
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(32, C, 3, 1, 1),
        )
        #         self.iDCTlayer = iDCTlayer2()
        self.iDCTlayer = iDctLayer(q)

    def forward(self, inputs):
        x = self.DCTlayer(inputs)
        dct_in = x
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        dct_out = x + dct_in
        x = self.iDCTlayer(dct_out, inputs)
        return x, dct_out


class DMCNN(nn.Module):
    def __init__(self, C, q):
        super().__init__()
        self.dct_branch = dctSubnet(C, q)
        self.conv0 = nn.Sequential(
            nn.Conv2d(2 * C, 32, 3, 1, 1),
            nn.PReLU(init=0.1)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.PReLU(init=0.1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.PReLU(init=0.1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.PReLU(init=0.1)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.PReLU(init=0.1)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.PReLU(init=0.1)
        )
        self.diconv1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 2, dilation=2),
            nn.PReLU(init=0.1)
        )
        self.diconv2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 4, dilation=4),
            nn.PReLU(init=0.1)
        )
        self.diconv3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 8, dilation=8),
            nn.PReLU(init=0.1)
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.PReLU(init=0.1)
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride=1),
            nn.PReLU(init=0.1)
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.PReLU(init=0.1)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride=1),
            nn.PReLU(init=0.1)
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(128, 32, 3, 1, 1),
            nn.PReLU(init=0.1)
        )
        self.output = nn.Sequential(
            nn.Conv2d(32, C, 5, 1, 2)
        )
        self.outframe1 = nn.Sequential(
            nn.Conv2d(256, C, 3, 1, 1),
        )
        self.outframe2 = nn.Sequential(
            nn.Conv2d(128, C, 3, 1, 1),
        )

    def forward(self, inputs):
        i_4, i_2, i = inputs

        d, dct_out = self.dct_branch(i)

        x = torch.cat([i, d], dim=1)
        x = self.conv0(x)
        x = self.conv1(x)
        res1 = x
        x = self.conv2(x)
        x = self.conv3(x)
        res2 = x
        x = self.conv4(x)
        x = self.conv6(x)
        x = self.diconv1(x)
        x = self.diconv2(x)
        x = self.diconv3(x)
        x = self.conv8(x)
        frame1 = self.outframe1(x) + i_4
        x = self.deconv1(x)
        x = torch.cat([x, res2], dim=1)
        x = self.conv9(x)
        frame2 = self.outframe2(x) + i_2
        x = self.deconv2(x)
        x = torch.cat([x, res1], dim=1)
        x = self.conv10(x)
        x = self.output(x) + i

        return frame1, frame2, x, d


class dmcnnLoss(nn.Module):
    def __init__(self, theta=0.618, lambd=0.9):
        super(dmcnnLoss, self).__init__()
        self.theta = theta
        self.lambd = lambd
        self.MSE1 = nn.MSELoss()
        self.MSE2 = nn.MSELoss()
        self.MSE3 = nn.MSELoss()
        self.MSE4 = nn.MSELoss()

    def forward(self, x, target):
        c4, c2, c, d = x
        o4, o2, o = target
        MSE4 = self.MSE1(c4, o4)
        MSE2 = self.MSE2(c2, o2)
        MSEp = self.MSE3(c, o)
        MSEd = self.MSE4(d, o)

        return self.theta * (self.theta * MSE4 + MSE2) + MSEp + self.lambd * MSEd, \
               MSE4, MSE2, MSEp, MSEd
