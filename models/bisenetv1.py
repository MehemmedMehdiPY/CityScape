"""
The source code was taken from the below repository:
    https://github.com/CoinCheung/BiSeNet/blob/master/lib/models/bisenetv1.py

Xception was integrated into context path as well.
"""

#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import Resnet18
from .xception import xception
from torch.nn import BatchNorm2d


class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class UpSample(nn.Module):

    def __init__(self, n_chan, factor=2):
        super(UpSample, self).__init__()
        out_chan = n_chan * factor * factor
        self.proj = nn.Conv2d(n_chan, out_chan, 1, 1, 0)
        self.up = nn.PixelShuffle(factor)
        self.init_weight()

    def forward(self, x):
        feat = self.proj(x)
        feat = self.up(feat)
        return feat

    def init_weight(self):
        nn.init.xavier_normal_(self.proj.weight, gain=1.)


class BiSeNetOutput(nn.Module):

    def __init__(self, in_chan, mid_chan, n_classes, up_factor=32, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.up_factor = up_factor
        out_chan = n_classes
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, out_chan, kernel_size=1, bias=True)
        self.up = nn.Upsample(scale_factor=up_factor,
                mode='bilinear', align_corners=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        x = self.up(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size= 1, bias=False)
        self.bn_atten = BatchNorm2d(out_chan)
        #  self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)
        atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        #  atten = self.sigmoid_atten(atten)
        atten = atten.sigmoid()
        out = torch.mul(feat, atten)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class ContextPath(nn.Module):
    def __init__(self, feature_extractor: str = 'xception', is_pretrained_extractor: bool = False, extractor_path: str = None):

        if feature_extractor not in ['xception', 'resnet']:
            raise Exception('Wrong feature_extractor {} defined. {} or {} should be chosen'.format(
                feature_extractor,
                'imagenet',
                'resnet'
            ))
        
        if feature_extractor == 'resnet' and is_pretrained_extractor and extractor_path is None:
            raise Exception('extractor_path is None while feature_extractor=\'resnet\' and is_pretrained_extractor=True.')
        
        super(ContextPath, self).__init__()

        if feature_extractor == 'xception' and is_pretrained_extractor:
            self.feature_extractor = xception(num_classes=1000, pretrained='imagenet', extractor_path=extractor_path)

        elif feature_extractor == 'xception' and not is_pretrained_extractor:
            self.feature_extractor = xception(num_classes=1000, pretrained=None, extractor_path=None)
        
        elif feature_extractor == 'resnet' and is_pretrained_extractor:
            self.feature_extractor = Resnet18(pretrained=True, extractor_path=extractor_path)
        
        elif feature_extractor == 'resnet' and not is_pretrained_extractor:
            self.feature_extractor = Resnet18(pretrained=False, extractor_path=None)
        
        self.is_pretrained_extractor = is_pretrained_extractor

        if feature_extractor == 'xception':
            self.arm16 = AttentionRefinementModule(728, 128)
            self.arm32 = AttentionRefinementModule(1024, 128)
        
        elif feature_extractor == 'resnet':
            self.arm16 = AttentionRefinementModule(256, 128)
            self.arm32 = AttentionRefinementModule(512, 128)

        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        
        if feature_extractor == 'xception':
            self.conv_avg = ConvBNReLU(1024, 128, ks=1, stride=1, padding=0)

        elif feature_extractor == 'resnet':
            self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)

        self.up32 = nn.Upsample(scale_factor=2.)
        self.up16 = nn.Upsample(scale_factor=2.)

    def forward(self, x):
        feat8, feat16, feat32 = self.feature_extractor(x)
        
        avg = torch.mean(feat32, dim=(2, 3), keepdim=True)
        avg = self.conv_avg(avg)

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg
        feat32_up = self.up32(feat32_sum)
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = self.up16(feat16_sum)
        feat16_up = self.conv_head16(feat16_up)

        return feat16_up, feat32_up # x8, x16

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class SpatialPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SpatialPath, self).__init__()
        self.conv1 = ConvBNReLU(3, 64, ks=7, stride=2, padding=3)
        self.conv2 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv_out = ConvBNReLU(64, 128, ks=1, stride=1, padding=0)
        self.init_weight()

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = self.conv_out(feat)
        return feat

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        ## use conv-bn instead of 2 layer mlp, so that tensorrt 7.2.3.4 can work for fp16
        self.conv = nn.Conv2d(out_chan,
                out_chan,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.bn = nn.BatchNorm2d(out_chan)
        #  self.conv1 = nn.Conv2d(out_chan,
        #          out_chan//4,
        #          kernel_size = 1,
        #          stride = 1,
        #          padding = 0,
        #          bias = False)
        #  self.conv2 = nn.Conv2d(out_chan//4,
        #          out_chan,
        #          kernel_size = 1,
        #          stride = 1,
        #          padding = 0,
        #          bias = False)
        #  self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.conv(atten)
        atten = self.bn(atten)
        #  atten = self.conv1(atten)
        #  atten = self.relu(atten)
        #  atten = self.conv2(atten)
        atten = atten.sigmoid()
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class BiSeNetV1(nn.Module):

    def __init__(self, n_classes: int, aux_mode: str = 'train', context_feature_extractor: str = 'xception', 
                 activation_function: str =  None, is_pretrained_extractor: bool = False, 
                 extractor_path: str = None
                 ):
        """
        Args:
            n_classes:                               Used to indicate the number of classes for classification problem.
            aux_mode:                                The mode of process. It is advised to initialize the model with the default
                                                     train model so that the mode can be changed later.
            context_feature_extractor:               The name of feature extractor in context path: xception or resnet
            activation_function:                     The name of activation function in final layer: softmax, sigmoid, or None
            is_pretrained_extractor:                 If False, feature_extractor in context path will be randomly initialized. 
                                                     If True, feature extractor states in context path will be loaded from extractor_path
            extractor_path:                          Path to feature extractor in context path. If is_pretrained_extractor=True, the path
                                                     must be supplied    
        """
        
        super(BiSeNetV1, self).__init__()

        if activation_function is None:
            self.activation_function = lambda x: x
        
        elif activation_function == 'sigmoid':
            self.activation_function = nn.Sigmoid()

        elif activation_function == 'softmax':
            self.activation_function = nn.Softmax(dim=1)

        else:
            raise ValueError(
                'No such function {} accepted'.format(activation_function)
                )
        
        self.cp = ContextPath(feature_extractor=context_feature_extractor, 
                              is_pretrained_extractor=is_pretrained_extractor,
                              extractor_path=extractor_path
                              )
        self.sp = SpatialPath()
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes, up_factor=8)
        self.aux_mode = aux_mode
        if self.aux_mode == 'train':
            self.conv_out16 = BiSeNetOutput(128, 64, n_classes, up_factor=8)
            self.conv_out32 = BiSeNetOutput(128, 64, n_classes, up_factor=16)
        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        feat_cp8, feat_cp16 = self.cp(x)
        feat_sp = self.sp(x)
        feat_fuse = self.ffm(feat_sp, feat_cp8)

        feat_out = self.conv_out(feat_fuse)
        feat_out = self.activation_function(feat_out)
        if self.aux_mode == 'train':
            feat_out16 = self.conv_out16(feat_cp8)
            feat_out32 = self.conv_out32(feat_cp16)

            feat_out16 = self.activation_function(feat_out16)
            feat_out32 = self.activation_function(feat_out32)
            return feat_out, feat_out16, feat_out32
        
        elif self.aux_mode == 'eval':
            return feat_out
        
        elif self.aux_mode == 'pred':
            feat_out = feat_out.argmax(dim=1)
            return feat_out
        else:
            raise NotImplementedError

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, (FeatureFusionModule, BiSeNetOutput)):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params


if __name__ == "__main__":
    net = BiSeNetV1(19)
    net.cuda()
    net.eval()
    in_ten = torch.randn(16, 3, 640, 480).cuda()
    out, out16, out32 = net(in_ten)
    print(out.shape)
    print(out16.shape)
    print(out32.shape)

    net.get_params()
