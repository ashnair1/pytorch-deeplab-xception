import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp
from modeling.aaspp import build_aaspp
from modeling.decoder import build_decoder
from modeling.fusion import build_fusion
from modeling.backbone import build_backbone


class Siamese(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(Siamese, self).__init__()

        # Branch 1
        self.conv1a = nn.Conv2d(inplanes, 16, kernel_size=kernel_size,
                                   stride=1, padding=1, dilation=1, bias=False)
        self.bn1a = BatchNorm(16)
        self.conv1b = nn.Conv2d(16, 32, kernel_size=kernel_size,
                                   stride=1, padding=1, dilation=1, bias=False)
        self.bn1b = BatchNorm(32)

        # Branch 2
        self.conv2a = nn.Conv2d(inplanes, 16, kernel_size=kernel_size,
                                   stride=1, padding=1, dilation=1, bias=False)
        self.bn2a = BatchNorm(16)
        self.conv2b = nn.Conv2d(16, 32, kernel_size=kernel_size,
                                   stride=1, padding=1, dilation=1, bias=False)
        self.bn2b = BatchNorm(32)

        # 1x1 conv to match original image shape
        self.last_conv = nn.Conv2d(32, 3, kernel_size=1,
                                   stride=1, padding=0, dilation=1, bias=False)

        self.relu = nn.ReLU()
        self.dropout2d = nn.Dropout2d(0.5)

        self._init_weight()

    def forward(self, x, x1):
        x = self.conv1a(x)
        x = self.bn1a(x)
        x = self.relu(x)
        x = self.conv1b(x)
        x = self.bn1b(x)
        x = self.relu(x)

        x1 = self.conv2a(x1)
        x1 = self.bn2a(x1)
        x1 = self.relu(x1)
        x1 = self.conv2b(x1)
        x1 = self.bn2b(x1)
        x1 = self.relu(x1)

        # Concatenate
        #x = torch.cat((x, x1), dim=1)
        # Addition
        x.add_(x1)
        #x = self.dropout2d(x)
        x = self.last_conv(x)

        return x

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


class GCB(nn.Module):
    def __init__(self, inplanes, planes, kh=7, kw=7):
        super(GCB, self).__init__()
        self.conv_l1 = nn.Conv2d(inplanes, 1024, kernel_size=(kh, 1),
                                 padding=(int(kh/2), 0))
        self.conv_l2 = nn.Conv2d(1024, planes, kernel_size=(1, kw),
                                 padding=(0, int(kw/2)))
        self.conv_r1 = nn.Conv2d(inplanes, 1024, kernel_size=(1, kw),
                                 padding=(0, int(kw/2)))
        self.conv_r2 = nn.Conv2d(1024, planes, kernel_size=(kh, 1),
                                 padding=(int(kh/2), 0))
        self.conv_g1 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.conv_g2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)

    def forward(self, x):
        x_l = self.conv_l2(self.conv_l1(x))
        x_r = self.conv_r2(self.conv_r1(x))
        x = x_l + x_r
        # Residual Block
        r = self.conv_g1(F.relu(x))
        r = self.conv_g2(F.relu(r))
        return x + r

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


class SiameseDeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(SiameseDeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn is True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        #self.siamese = Siamese(3,3,3,1,1, BatchNorm)
        #self.gcb = GCB(1024, 1024)
        #self.aspp = build_aspp(backbone, output_stride, BatchNorm, siamese=True)
        #self.aaspp = build_aaspp(backbone, output_stride, BatchNorm, siamese=True)
        self.aspp1 = build_aspp(backbone, output_stride, BatchNorm, siamese=False)
        self.aspp2 = build_aspp(backbone, output_stride, BatchNorm, siamese=False)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm, siamese=True)
        self.fusion = build_fusion(num_classes, backbone, BatchNorm, siamese=False)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input, input1):

        #x = self.siamese(input, input1)
        #x, low_level_feat = self.backbone(input)

        # Push separately through backbone (encoder) and then concatenate
        # ######################################################################
        x, low_level_feat = self.backbone(input)
        x1, low_level_feat1 = self.backbone(input1)


        sum1 = torch.add(low_level_feat, low_level_feat1)
        #x = torch.cat((x, x1), dim=1)
        #low_level_feat = torch.cat((low_level_feat, low_level_feat1), dim=1)
        ########################################################################

        # Pass to Global Convolutional Block
        #x = self.gcb(x)

        x = self.aspp1(x)
        x1 = self.aspp2(x1)
        #x = self.aaspp(x)
        #x = self.decoder(x, low_level_feat)
        x = self.fusion(x, x1, sum1)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp1, self.aspp2, self.fusion]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


if __name__ == "__main__":
    model = SiameseDeepLab(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())
