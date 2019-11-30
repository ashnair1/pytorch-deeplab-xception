import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class Fusion(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm, siamese=False):
        super(Fusion, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        if siamese is True:
            low_level_inplanes *= 2

        self.conv1a = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1a = BatchNorm(48)
        self.conv1b = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1b = BatchNorm(48)
        self.relu = nn.ReLU()

        self.last_conv = nn.Sequential(nn.Conv2d(48, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))

        self.channel_attention = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                               nn.Conv2d(256, 128, kernel_size=1, stride=1),
                                               nn.ReLU(),
                                               nn.Conv2d(128, 48, kernel_size=1, stride=1),
                                               nn.Sigmoid())
        self._init_weight()

    def forward(self, x, x1, sum1):
        x = self.conv1a(x)
        x = self.bn1a(x)
        x = self.relu(x)
        x = F.interpolate(x, size=x.size()[2:], mode='bilinear', align_corners=True)

        x1 = self.conv1b(x1)
        x1 = self.bn1b(x1)
        x1 = self.relu(x1)
        x1 = F.interpolate(x, size=x1.size()[2:], mode='bilinear', align_corners=True)

        sum2 = torch.add(x, x1)
        ############ Channel Attention #################
        #s = torch.cat((x, sum1), dim=1)
        #import pdb
        #pdb.set_trace()
        #sum1 = torch.mean(sum1.view(sum1.size(0),sum1.size(1), -1), dim=2)
        cs = self.channel_attention(sum1)
        ################################################
        sum3 = torch.mul(cs, sum2)
        #sum3 = torch.add(cs, sum2)
        sum3 = self.last_conv(sum3)

        return sum3

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


def build_fusion(num_classes, backbone, BatchNorm, siamese=False):
    return Fusion(num_classes, backbone, BatchNorm, siamese)

