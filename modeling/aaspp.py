import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class _AtrousConvBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_AtrousConvBlock, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class _AASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, dilations, BatchNorm):
        super(_AASPPModule, self).__init__()

        self.atrous_list = nn.ModuleList([_AtrousConvBlock(inplanes, planes, kernel_size, dilations[0], dilations[0], BatchNorm)] +
                                         [_AtrousConvBlock(planes,planes, kernel_size, dilation, dilation, BatchNorm) for dilation in dilations[1:]])
        self.atrous_blocks = nn.Sequential(*self.atrous_list) 
        self.trace = []

    def forward(self, x):
        self.atrous_blocks(x)
        return x

# class Flatten(nn.Module):
#     def forward(self, x):
#         N, C, H, W = x.size() # read in N, C, H, W
#         return x.view(N, -1)


# class _FusionModule(nn.Module):
#     def __init__(self, inplanes, planes, kernel_size, BatchNorm):
#         super(_FusionModule, self).__init__()

#         # Standard Convolutions
#         self.conv_list = nn.ModuleList([nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1),
#                                           BatchNorm(planes),
#                                           nn.ReLU(),
#                                           nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1),
#                                           BatchNorm(planes),
#                                           nn.ReLU()])
#         self.conv_block = nn.Sequential(*self.conv_list)

#         #self.linear = nn.Sequential(nn.Conv2d(planes, planes, kernel_size=1)
#         #                            #nn.Linear(256*33*33, 1),
#         #                            Flatten(),
#         #                            )




#         self.trace = []

#     def forward(self, x):

#         xf = self.conv_block(x)
#         ###########################
#         xf = torch.flatten(xf, 1)
#         xf = self.linear(xf)
#         xf = 'blah'
#         ########################
#         # Reshape
#         xf = xf.view(-1, 256, 33, 33)

#         return x


class AASPP(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm, siamese=False):
        super(AASPP, self).__init__()
        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        else:
            inplanes = 2048

        if siamese is True:
            inplanes = inplanes * 2

        if output_stride == 16:
            dilations = [[1],
                         [3, 6, 4, 2],
                         [6, 12, 6, 4], 
                         [12, 18, 12, 6]]
        elif output_stride == 8:
            dilations = [[1],
                         [6, 12, 8, 4],
                         [12, 24, 12, 8], 
                         [24, 36, 24, 12]]
        else:
            raise NotImplementedError

        

        # ASPP 1x1 convolution
        self.aaspp1 = _AASPPModule(inplanes, 256, 1, dilations=dilations[0], BatchNorm=BatchNorm)
        self.aaspp2 = _AASPPModule(inplanes, 256, 3, dilations=dilations[1], BatchNorm=BatchNorm)
        self.aaspp3 = _AASPPModule(inplanes, 256, 3, dilations=dilations[2], BatchNorm=BatchNorm)
        self.aaspp4 = _AASPPModule(inplanes, 256, 3, dilations=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             BatchNorm(256),
                                             nn.ReLU())
        #self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.conv1 = nn.Conv2d(16640, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        #self.fusion = _FusionModule(inplanes, 256, 3, BatchNorm=BatchNorm)
        self._init_weight()

    def forward(self, x):
        #xf = self.fusion(x)
        x1 = self.aaspp1(x)
        x2 = self.aaspp2(x)
        x3 = self.aaspp3(x)
        x4 = self.aaspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        #assert x.shape == xf.shape
        #x.add_(xf) 

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_aaspp(backbone, output_stride, BatchNorm, siamese=False):
    return AASPP(backbone, output_stride, BatchNorm, siamese)