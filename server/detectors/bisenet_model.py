"""BiSeNet face-parsing model architecture.

Vendored from https://github.com/zllrunning/face-parsing.PyTorch (MIT license).
Provides 19-class face segmentation using a ResNet-18 backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self._init_weight()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

    def _init_weight(self):
        nn.init.kaiming_normal_(self.conv.weight, a=1)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)


class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes):
        super().__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
        self._init_weight()

    def forward(self, x):
        return self.conv_out(self.conv(x))

    def _init_weight(self):
        nn.init.kaiming_normal_(self.conv_out.weight, a=1)


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()
        self._init_weight()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.adaptive_avg_pool2d(feat, 1)
        atten = self.sigmoid_atten(self.bn_atten(self.conv_atten(atten)))
        return feat * atten

    def _init_weight(self):
        nn.init.kaiming_normal_(self.conv_atten.weight, a=1)
        nn.init.constant_(self.bn_atten.weight, 1)
        nn.init.constant_(self.bn_atten.bias, 0)


class ContextPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet18(weights=None)
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)

    def forward(self, x):
        # Use resnet layers directly
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        feat8 = self.resnet.layer1(x)
        feat8 = self.resnet.layer2(feat8)
        feat16 = self.resnet.layer3(feat8)
        feat32 = self.resnet.layer4(feat16)

        avg = F.adaptive_avg_pool2d(feat32, 1)
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, size=feat32.shape[2:], mode="nearest")

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, size=feat16.shape[2:], mode="nearest")
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, size=feat8.shape[2:], mode="nearest")
        feat16_up = self.conv_head16(feat16_up)

        return feat8, feat16_up, feat32_up


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan, out_chan // 4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(out_chan // 4, out_chan, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self._init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = F.adaptive_avg_pool2d(feat, 1)
        atten = self.relu(self.conv1(atten))
        atten = self.sigmoid(self.conv2(atten))
        return feat + feat * atten

    def _init_weight(self):
        nn.init.kaiming_normal_(self.conv1.weight, a=1)
        nn.init.kaiming_normal_(self.conv2.weight, a=1)


class BiSeNet(nn.Module):
    def __init__(self, n_classes=19):
        super().__init__()
        self.cp = ContextPath()
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(128, 64, n_classes)

    def forward(self, x):
        h, w = x.shape[2:]
        feat_res8, feat_cp8, feat_cp16 = self.cp(x)
        feat_fuse = self.ffm(feat_res8, feat_cp8)

        feat_out = self.conv_out(feat_fuse)
        feat_out16 = self.conv_out16(feat_cp8)
        feat_out32 = self.conv_out32(feat_cp16)

        feat_out = F.interpolate(feat_out, size=(h, w), mode="bilinear", align_corners=True)
        feat_out16 = F.interpolate(feat_out16, size=(h, w), mode="bilinear", align_corners=True)
        feat_out32 = F.interpolate(feat_out32, size=(h, w), mode="bilinear", align_corners=True)

        return feat_out, feat_out16, feat_out32
