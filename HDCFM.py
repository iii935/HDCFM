import sys
import os
# sys.path.append()

from torch import nn




import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from torch.autograd import Variable

from torch.utils.checkpoint import checkpoint
from torch.nn import functional as F

import sys
sys.path.append('models/modules/ddf/')
from ddf import DDFPack

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

def DEconv(in_channels, kernel_size, bias=True):
    return DDFPack(in_channels,kernel_size)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res



import torch
from mmcv.cnn import constant_init, kaiming_init
from torch import nn


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
    else:
        constant_init(m, val=0)


class ContextBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 ratio,
                 pooling_type='att',
                 fusion_types=('channel_add', )):
        super(ContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                DEconv(self.planes,3),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                DEconv(self.planes,3),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pooling_type == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out    


class Non_Linear(nn.Module):
    def __init__(self,inch,):
        super(Non_Linear, self).__init__()
        self.inch=inch
    def forward(self, x,w):
        ws=[w[i,:,:,:,:] for i in range(w.shape[0])]
        xs=[x[i:i+1,:,:,:] for i in range(x.shape[0])]
        os=[F.conv2d(xs[i],ws[i]) for i in range(w.shape[0])]
        output =torch.cat(os,dim=0)
        # input_ = torch.cat(torch.split(input_, 1, dim=0), dim=1)
        # output_ = F.conv2d(input_,w,stride=1)
        # output_ = torch.cat(torch.split(output_, channel_out, dim=1), dim=0)

        # out=F.conv2d(x, w,stride=1)
        return output


class ContextBlockInstance(nn.Module):

    def __init__(self,
                 inplanes,
                 ratio,
                 pooling_type='att',
                 fusion_types=('channel_add', )):
        super(ContextBlockInstance, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                # nn.InstanceNorm2d(self.planes,affine=True),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                DEconv(self.planes,3),
                nn.LayerNorm([self.planes, 1, 1]),
                # nn.InstanceNorm2d(self.planes,affine=True),
                nn.ReLU(inplace=True),  # yapf: disable
                DEconv(self.planes,3),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pooling_type == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out    


def default_conv(in_channels, out_channels, kernel_size, stride=1, padding=None, bias=True, groups=1):
       if not padding and stride==1:
           padding = kernel_size // 2
       return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias, groups=groups)
       
class ESA(nn.Module):
     def __init__(self, n_feats, conv=DEconv):
         super(ESA, self).__init__()
         f = n_feats // 6
         self.conv1 = default_conv(n_feats, f, kernel_size=1)
        #  self.conv_f = default_conv(f, f, kernel_size=1)
         self.conv_max = conv(f, kernel_size=3)
         self.conv2 = conv(f, kernel_size=3)
        #  self.conv3 = conv(f,  kernel_size=3)
         self.conv3_ = conv(f, kernel_size=3)
         self.conv4 = default_conv(f, n_feats, kernel_size=1)
         self.sigmoid = nn.Sigmoid()
         self.relu = nn.ReLU(inplace=True)
  
     def forward(self, x):
         f=x
         c1_ = (self.conv1(f))
        #  c1 = self.conv2(c1_)
         v_max = F.max_pool2d(c1_, kernel_size=7, stride=3)
         v_range = self.relu(self.conv_max(v_max))
        #  c3 = self.relu(self.conv3(v_range))
         c3 = self.conv3_(v_range)
         c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear') 
        #  cf = self.conv_f(c1_)
         c4 = self.conv4(c3+c1_)
         m = self.sigmoid(c4)
         
         return x * m


# if __name__ == "__main__":
#     import os
#     # os.environ['CUDA_VISIBLE_DEVICES']='1'
#     device=torch.device('cuda:0')
    
#     model=_NLResGroup(DEconv,64,3,nn.LeakyReLU(inplace=True),res_scale=1).to(device)
#     summary(model,(64,128,128))

class ContextRes(nn.Module):
     def __init__(self, n_feats, conv=DEconv):
         super(ContextRes, self).__init__()
         self.conv1=conv(n_feats,3)
         self.relu=nn.ReLU(inplace=True)
         self.contextconv=ContextBlock(n_feats,0.2,'avg')
         self.conv2=conv(n_feats,3)
     def forward(self,x):
         x1=self.conv1(x)
         x2=self.relu(x1)
         x3=self.contextconv(x2)+x2
         x4=self.conv2(x3)
         out=x4+x
         return out

class ContextDense(nn.Module):
     def __init__(self, n_feats,nori,nb):
         super(ContextDense, self).__init__()
         self.nb=nb
         self.cbres=nn.ModuleList()
         self.first=nn.Conv2d(nori,n_feats,1,padding=0)
         for i in range(nb):
             self.cbres.append(ContextRes(n_feats))
         self.last=nn.Conv2d(n_feats,nori,1,padding=0)
         
     def forward(self,f):
         x=self.first(f)
         outs=[]
         for i in range(self.nb):
             x=self.cbres[i](x)
            #  x=checkpoint(self.cbres[i],x)
             outs.append(x)
         sum_1=sum(outs)
         output=self.last(sum_1)+f
         return output
         


def color_block(in_filters, out_filters, normalization=False):
    conv = nn.Conv2d(in_filters, out_filters, 1, stride=1, padding=0)
    pooling = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=True)
    act = nn.LeakyReLU(0.2)
    layers = [conv, pooling, act] 
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
    return layers

class CB(nn.Module):
    def __init__(self, in_filters, out_filters,normalization=False):
        super(CB, self).__init__()
        self.conv=nn.Conv2d(in_filters, out_filters, 1, stride=1, padding=0)
        self.pooling = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=True)
        self.act = nn.LeakyReLU(0.2)
        self.normalization=normalization
        if normalization:
            self.norm=nn.InstanceNorm2d(out_filters, affine=True)
    def forward(self,x):
        x=self.conv(x)
        x=self.pooling(x)
        x=self.act(x)
        if self.normalization:
            x=self.norm(x)
        return x
        


def color_blockDF(in_filters, out_filters, normalization=False):
    conv = nn.Conv2d(in_filters, out_filters, 1, stride=1, padding=0)
    conv1 = DDFPack(out_filters,3,1)
    pooling = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=True)
    act = nn.LeakyReLU(0.2)
    layers = [conv , conv1 , pooling, act] 
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
    return layers


class Color_Condition(nn.Module):
    def __init__(self, in_channels=3, out_c=3):
        super(Color_Condition, self).__init__()

        self.model = nn.Sequential(
            *color_block(3, 16, normalization=True),
            *color_block(16, 32, normalization=True),
            # ContextBlockInstance(32,0.25,pooling_type='avg',fusion_types=('channel_mul',)),
            *color_block(32, 64, normalization=True),
            # ContextBlockInstance(64,0.25,pooling_type='avg',fusion_types=('channel_mul',)),
            *color_block(64, 128, normalization=True),
            # ContextBlockInstance(128,0.25,pooling_type='avg',fusion_types=('channel_mul',)),
            *color_block(128, 128),
            nn.Dropout(p=0.5),
            nn.Conv2d(128, out_c, 1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, img_input):
        return self.model(img_input)


class Color_ConditionDF(nn.Module):
    def __init__(self, in_channels=3, out_c=3):
        super(Color_ConditionDF, self).__init__()

        self.model = nn.Sequential(
            *color_blockDF(3, 16, normalization=True),
            *color_blockDF(16, 32, normalization=True),
            # ContextBlockInstance(32,0.25,pooling_type='avg',fusion_types=('channel_mul',)),
            *color_blockDF(32, 64, normalization=True),
            # ContextBlockInstance(64,0.25,pooling_type='avg',fusion_types=('channel_mul',)),
            *color_blockDF(64, 128, normalization=True),
            # ContextBlockInstance(128,0.25,pooling_type='avg',fusion_types=('channel_mul',)),
            *color_blockDF(128, 128),
            nn.Dropout(p=0.5),
            nn.Conv2d(128, out_c, 1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, img_input):
        return self.model(img_input)


class ConditionNet(nn.Module):
    def __init__(self, nf=64, classifier='color_condition', cond_c=3):
        super(ConditionNet, self).__init__()

        if classifier=='color_condition':
            self.classifier = Color_Condition(out_c=cond_c)

        self.GFM_nf = 64

        self.cond_scale_first = nn.Linear(cond_c, nf)
        self.cond_scale_HR = nn.Linear(cond_c, nf)
        self.cond_scale_last = nn.Linear(cond_c, 3)

        self.cond_shift_first = nn.Linear(cond_c, nf)
        self.cond_shift_HR = nn.Linear(cond_c, nf)
        self.cond_shift_last = nn.Linear(cond_c, 3)

        self.conv_first = nn.Conv2d(3, nf, 1, 1)
        self.HRconv = nn.Conv2d(nf, nf, 1, 1)
        self.conv_last = nn.Conv2d(nf, 3, 1, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        content = x[0]
        condition = x[1]
        # b,3,h,w
        fea = self.classifier(condition).squeeze(2).squeeze(2)

        scale_first = self.cond_scale_first(fea)
        shift_first = self.cond_shift_first(fea)

        scale_HR = self.cond_scale_HR(fea)
        shift_HR = self.cond_shift_HR(fea)

        scale_last = self.cond_scale_last(fea)
        shift_last = self.cond_shift_last(fea)

        out = self.conv_first(content)
        out = out * scale_first.view(-1, self.GFM_nf, 1, 1) + shift_first.view(-1, self.GFM_nf, 1, 1) + out
        out = self.act(out)

        out = self.HRconv(out)
        out = out * scale_HR.view(-1, self.GFM_nf, 1, 1) + shift_HR.view(-1, self.GFM_nf, 1, 1) + out
        out = self.act(out)

        out = self.conv_last(out)
        out = out * scale_last.view(-1, 3, 1, 1) + shift_last.view(-1, 3, 1, 1) + out

        return out



class ConditionNetDDF(nn.Module): 
    def __init__(self, nf=64, classifier='color_condition', cond_c=3):
        super(ConditionNetDDF, self).__init__()

        if classifier=='color_condition':
            self.classifier = Color_Condition(out_c=cond_c)

        self.GFM_nf = 64

        self.cond_scale_first = nn.Linear(cond_c, nf)
        self.cond_scale_HR = nn.Linear(cond_c, nf)
        self.cond_scale_last = nn.Linear(cond_c, 3)

        self.cond_shift_first = nn.Linear(cond_c, nf)
        self.cond_shift_HR = nn.Linear(cond_c, nf)
        self.cond_shift_last = nn.Linear(cond_c, 3)
        self.conv_first = nn.Sequential(
            nn.Conv2d(3, nf, 1, 1),
            DDFPack(nf,version='f')
        )
        self.HRconv= nn.Sequential(
            DDFPack(nf,version='f'),
            nn.Conv2d(nf, nf, 1, 1)
        )
        self.conv_last = nn.Sequential(
            DDFPack(nf,version='f'),
            nn.Conv2d(nf, 3, 1, 1)
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        content = x[0]
        condition = x[1]
        # b,3,h,w
        fea = self.classifier(condition).squeeze(2).squeeze(2)

        scale_first = self.cond_scale_first(fea)
        shift_first = self.cond_shift_first(fea)

        scale_HR = self.cond_scale_HR(fea)
        shift_HR = self.cond_shift_HR(fea)

        scale_last = self.cond_scale_last(fea)
        shift_last = self.cond_shift_last(fea)

        out = self.conv_first(content)
        out = out * scale_first.view(-1, self.GFM_nf, 1, 1) + shift_first.view(-1, self.GFM_nf, 1, 1) + out
        out = self.act(out)

        out = self.HRconv(out)
        out = out * scale_HR.view(-1, self.GFM_nf, 1, 1) + shift_HR.view(-1, self.GFM_nf, 1, 1) + out
        out = self.act(out)

        out = self.conv_last(out)
        out = out * scale_last.view(-1, 3, 1, 1) + shift_last.view(-1, 3, 1, 1) + out

        return out



class ConditionNetDDFALL(nn.Module): 
    def __init__(self, nf=64, classifier='color_condition', cond_c=3):
        super(ConditionNetDDFALL, self).__init__()

        if classifier=='color_condition':
            self.classifier = Color_ConditionDF(out_c=cond_c)

        self.GFM_nf = 64

        self.cond_scale_first = nn.Linear(cond_c, nf)
        self.cond_scale_HR = nn.Linear(cond_c, nf)
        self.cond_scale_last = nn.Linear(cond_c, 3)

        self.cond_shift_first = nn.Linear(cond_c, nf)
        self.cond_shift_HR = nn.Linear(cond_c, nf)
        self.cond_shift_last = nn.Linear(cond_c, 3)
        self.conv_first = nn.Sequential(
            nn.Conv2d(3, nf, 1, 1),
            DDFPack(nf,version='f')
        )
        self.HRconv= nn.Sequential(
            DDFPack(nf,version='f'),
            nn.Conv2d(nf, nf, 1, 1)
        )
        self.conv_last = nn.Sequential(
            DDFPack(nf,version='f'),
            nn.Conv2d(nf, 3, 1, 1)
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        content = x[0]
        condition = x[1]
        # b,3,h,w
        fea = self.classifier(condition).squeeze(2).squeeze(2)

        scale_first = self.cond_scale_first(fea)
        shift_first = self.cond_shift_first(fea)

        scale_HR = self.cond_scale_HR(fea)
        shift_HR = self.cond_shift_HR(fea)

        scale_last = self.cond_scale_last(fea)
        shift_last = self.cond_shift_last(fea)

        out = self.conv_first(content)
        out = out * scale_first.view(-1, self.GFM_nf, 1, 1) + shift_first.view(-1, self.GFM_nf, 1, 1) + out
        out = self.act(out)

        out = self.HRconv(out)
        out = out * scale_HR.view(-1, self.GFM_nf, 1, 1) + shift_HR.view(-1, self.GFM_nf, 1, 1) + out
        out = self.act(out)

        out = self.conv_last(out)
        out = out * scale_last.view(-1, 3, 1, 1) + shift_last.view(-1, 3, 1, 1) + out

        return out



class Color_ConditionNL(nn.Module):
    def __init__(self, in_channels=3, out_c=3):
        super(Color_ConditionNL, self).__init__()

        self.model = nn.Sequential(
            *color_block(3, 16, normalization=True),
            *color_block(16, 32, normalization=True),
            # ContextBlockInstance(32,0.25,pooling_type='avg',fusion_types=('channel_mul',)),
            *color_block(32, 64, normalization=True),
            # ContextBlockInstance(64,0.25,pooling_type='avg',fusion_types=('channel_mul',)),
            *color_block(64, 128, normalization=True),
            # ContextBlockInstance(128,0.25,pooling_type='avg',fusion_types=('channel_mul',)),
            *color_block(128, 128),
            nn.Dropout(p=0.5)
        )
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.pool1=nn.AdaptiveAvgPool2d(1)
        self.linear=nn.Linear(128,64*64)
        self.conv=nn.Conv2d(128, out_c, 1, stride=1, padding=0)

    def forward(self, img_input):
        
        mid = self.model(img_input)
        midw=self.pool1(mid).squeeze(2).squeeze(2)
        
        tcond =self.pool(self.conv(mid))
        
        w=self.linear(midw).reshape(-1,64,64,1,1)
        return tcond,w
        



class ConditionNetNLDDF(nn.Module):
    def __init__(self, nf=64, classifier='color_condition', cond_c=3):
        super(ConditionNetNLDDF, self).__init__()

        if classifier=='color_condition':
            self.classifier = Color_ConditionNL(out_c=cond_c)

        self.GFM_nf = 64

        self.cond_scale_first = nn.Linear(cond_c, nf)
        self.cond_scale_HR = nn.Linear(cond_c, nf)
        self.cond_scale_last = nn.Linear(cond_c, 3)

        self.cond_shift_first = nn.Linear(cond_c, nf)
        self.cond_shift_HR = nn.Linear(cond_c, nf)
        self.cond_shift_last = nn.Linear(cond_c, 3)
        self.conv_first = nn.Sequential(
            nn.Conv2d(3, nf, 1, 1),
            # ContextBlock(nf,0.25,pooling_type='avg',fusion_types=('channel_mul')),
            DDFPack(nf,version='f')
        )
        self.HRconv= nn.Sequential(
            DDFPack(nf,version='f'),
            nn.Conv2d(nf, nf, 1, 1)
        )
        self.hrcontext=ContextBlock(nf,0.25,pooling_type='avg',fusion_types=('channel_mul',))
        self.firstcontext=ContextBlock(nf,0.25,pooling_type='avg',fusion_types=('channel_mul',))
        # self.contextblock=ContextBlock(nf,0.25,pooling_type='avg',fusion_types=('channel_mul'))
        
        self.conv_last = nn.Sequential(
            DDFPack(nf,version='f'),
            nn.Conv2d(nf, 3, 1, 1)
        )
        self.act = nn.ReLU(inplace=True)
        
        # self.non_linear = nn.Sequential(
        #     nn.Conv2d(nf,nf,1,1,0),
        #     nn.InstanceNorm2d(nf,affine=True),
        #     nn.Conv2d(nf,nf,1,1,0)
        # )
        self.non_linear = Non_Linear(nf)

    def forward(self, x):
        content = x[0]
        condition = x[1]
        # b,3,h,w
        fea,w = self.classifier(condition)
        fea=fea.squeeze(2).squeeze(2)

        scale_first = self.cond_scale_first(fea)
        shift_first = self.cond_shift_first(fea)

        scale_HR = self.cond_scale_HR(fea)
        shift_HR = self.cond_shift_HR(fea)

        scale_last = self.cond_scale_last(fea)
        shift_last = self.cond_shift_last(fea)


        out = self.conv_first(content)
        out=self.firstcontext(out)+out
        
        
        
        # # print(out.shape)
        out = out * scale_first.view(-1, self.GFM_nf, 1, 1) + shift_first.view(-1, self.GFM_nf, 1, 1) + out
        out = self.act(out)
        
        out = self.non_linear(out,w)

        out = self.HRconv(out)
        out=self.hrcontext(out)+out
        out = out * scale_HR.view(-1, self.GFM_nf, 1, 1) + shift_HR.view(-1, self.GFM_nf, 1, 1) + out
        out = self.act(out)

        out = self.conv_last(out)
        out = out * scale_last.view(-1, 3, 1, 1) + shift_last.view(-1, 3, 1, 1) + out

        return out




class ConditionNetContextDDF(nn.Module):
    def __init__(self, nf=64, classifier='color_condition', cond_c=3):
        super(ConditionNetContextDDF, self).__init__()

        if classifier=='color_condition':
            self.classifier = Color_Condition(out_c=cond_c)

        self.GFM_nf = 64

        self.cond_scale_first = nn.Linear(cond_c, nf)
        self.cond_scale_HR = nn.Linear(cond_c, nf)
        self.cond_scale_last = nn.Linear(cond_c, 3)

        self.cond_shift_first = nn.Linear(cond_c, nf)
        self.cond_shift_HR = nn.Linear(cond_c, nf)
        self.cond_shift_last = nn.Linear(cond_c, 3)
        self.conv_first = nn.Sequential(
            nn.Conv2d(3, nf, 1, 1),
            ContextBlock(nf,0.25,pooling_type='avg',fusion_types=('channel_mul',)),
            DDFPack(nf,version='f'),
            ContextBlock(nf,0.25,pooling_type='avg',fusion_types=('channel_mul',)),
            DDFPack(nf,version='f')
        )
        self.HRconv= nn.Sequential(
            DDFPack(nf,version='f'),
            ContextBlock(nf,0.25,pooling_type='avg',fusion_types=('channel_mul',)),
            nn.Conv2d(nf, nf, 1, 1)
        )
        self.conv_last = nn.Sequential(
            DDFPack(nf,version='f'),
            ContextBlock(nf,0.25,pooling_type='avg',fusion_types=('channel_mul',)),
            DDFPack(nf,version='f'),
            ContextBlock(nf,0.25,pooling_type='avg',fusion_types=('channel_mul',)),
            nn.Conv2d(nf, 3, 1, 1)
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        content = x[0]
        condition = x[1]
        # b,3,h,w
        fea = self.classifier(condition).squeeze(2).squeeze(2)

        scale_first = self.cond_scale_first(fea)
        shift_first = self.cond_shift_first(fea)

        scale_HR = self.cond_scale_HR(fea)
        shift_HR = self.cond_shift_HR(fea)

        scale_last = self.cond_scale_last(fea)
        shift_last = self.cond_shift_last(fea)

        out = self.conv_first(content)
        out = out * scale_first.view(-1, self.GFM_nf, 1, 1) + shift_first.view(-1, self.GFM_nf, 1, 1) + out
        out = self.act(out)

        out = self.HRconv(out)
        out = out * scale_HR.view(-1, self.GFM_nf, 1, 1) + shift_HR.view(-1, self.GFM_nf, 1, 1) + out
        out = self.act(out)

        out = self.conv_last(out)
        out = out * scale_last.view(-1, 3, 1, 1) + shift_last.view(-1, 3, 1, 1) + out

        return out



class LocalNetContextDDF(nn.Module):
    def __init__(self, nf=64, classifier='color_condition', cond_c=3):
        super(LocalNetContextDDF, self).__init__()

        if classifier=='color_condition':
            self.classifier = Color_Condition(out_c=cond_c)

        self.GFM_nf = 64
        self.nf=nf
        self.cond_scale_first = nn.Linear(cond_c, nf)
        self.cond_scale_HR = nn.Linear(cond_c, nf)
        self.cond_scale_last = nn.Linear(cond_c, 3)
        self.cond_c=cond_c
        self.cond_shift_first = nn.Linear(cond_c, nf)
        self.cond_shift_HR = nn.Linear(cond_c, nf)
        self.cond_shift_last = nn.Linear(cond_c, 3)
        self.conv_first = nn.Sequential(
            nn.Conv2d(3, nf, 1, 1),
            ContextBlock(nf,0.25,pooling_type='avg',fusion_types=('channel_mul',)),
            DDFPack(nf,version='f'),
            ContextBlock(nf,0.25,pooling_type='avg',fusion_types=('channel_mul',)),
            DDFPack(nf,version='f')
        )
        self.HRconv= nn.Sequential(
            DDFPack(nf,version='f'),
            ContextBlock(nf,0.25,pooling_type='avg',fusion_types=('channel_mul',)),
            nn.Conv2d(nf, nf, 1, 1)
        )
        self.conv_last = nn.Sequential(
            DDFPack(nf,version='f'),
            ContextBlock(nf,0.25,pooling_type='avg',fusion_types=('channel_mul',)),
            DDFPack(nf,version='f'),
            ContextBlock(nf,0.25,pooling_type='avg',fusion_types=('channel_mul',)),
            nn.Conv2d(nf, 3, 1, 1)
        )
        self.act = nn.ReLU(inplace=True)
        K=960
        self.K=K
        self.unfold=nn.Unfold((K,K), dilation=1, padding=0, stride=K)

    def forward(self, x):
        content = x[0]
        condition = x[1]
        hw=(x[1].shape[2]//self.K)*(x[1].shape[3]//self.K)*x[1].shape[0]
        condition1=self.unfold(condition).view(x[1].shape[0],3,self.K,self.K,x[1].shape[2]//self.K,x[1].shape[3]//self.K)
        condition1=condition1.permute(0,4,5,1,2,3).view(hw,3,self.K,self.K)
        # conditiongt=condition[:,:,:3,:3]
        # conditionpatch0=condition1[:,:,:,:,0,0]
        # mse=torch.mean((conditiongt-conditionpatch0)*(conditiongt-conditionpatch0))
        # b,3,h,w
        fea = self.classifier(condition1).squeeze(2).squeeze(2)#[:,:,0,0].view(x[1].shape[0],x[1].shape[2]//self.K,x[1].shape[3]//self.K,self.cond_c)


        scale_first = self.cond_scale_first(fea).view(x[1].shape[0],x[1].shape[2]//self.K,x[1].shape[3]//self.K,self.nf).permute(0,3,1,2)
        shift_first = self.cond_shift_first(fea).view(x[1].shape[0],x[1].shape[2]//self.K,x[1].shape[3]//self.K,self.nf).permute(0,3,1,2)
        scale_first=F.upsample_bilinear(scale_first,(x[1].shape[2],x[1].shape[3]))
        shift_first=F.upsample_bilinear(shift_first,(x[1].shape[2],x[1].shape[3]))

        out = self.conv_first(content)
        out = out * scale_first + shift_first + out
        out = self.act(out)
        scale_first=scale_first.cpu()
        shift_first=shift_first.cpu()
        content=content.cpu()
        torch.cuda.empty_cache()


        scale_HR = self.cond_scale_HR(fea).view(x[1].shape[0],x[1].shape[2]//self.K,x[1].shape[3]//self.K,self.nf).permute(0,3,1,2)
        shift_HR = self.cond_shift_HR(fea).view(x[1].shape[0],x[1].shape[2]//self.K,x[1].shape[3]//self.K,self.nf).permute(0,3,1,2)
        scale_HR=F.upsample_bilinear(scale_HR,(x[1].shape[2],x[1].shape[3]))
        shift_HR=F.upsample_bilinear(shift_HR,(x[1].shape[2],x[1].shape[3]))
        
        out = self.HRconv(out)
        out = out * scale_HR + shift_HR + out
        out = self.act(out)
        
        scale_HR=scale_HR.cpu()
        shift_HR=shift_HR.cpu()
        torch.cuda.empty_cache()
        
        


        scale_last = self.cond_scale_last(fea).view(x[1].shape[0],x[1].shape[2]//self.K,x[1].shape[3]//self.K,3).permute(0,3,1,2)
        shift_last = self.cond_shift_last(fea).view(x[1].shape[0],x[1].shape[2]//self.K,x[1].shape[3]//self.K,3).permute(0,3,1,2)
        scale_last=F.upsample_bilinear(scale_last,(x[1].shape[2],x[1].shape[3]))
        shift_last=F.upsample_bilinear(shift_last,(x[1].shape[2],x[1].shape[3]))
        

        out = self.conv_last(out)
        out = out * scale_last + shift_last + out

        return out




class Color_ConditionFT(nn.Module):
    def __init__(self, in_channels=3, out_c=64):
        super(Color_ConditionFT, self).__init__()

        self.model = nn.Sequential(
            *color_block(3, 16, normalization=True),
            *color_block(16, 32, normalization=True),
            # ContextBlockInstance(32,0.25,pooling_type='avg',fusion_types=('channel_mul',)),
            *color_block(32, 64, normalization=True),
            # ContextBlockInstance(64,0.25,pooling_type='avg',fusion_types=('channel_mul',)),
            *color_block(64, 128, normalization=True),
            # ContextBlockInstance(128,0.25,pooling_type='avg',fusion_types=('channel_mul',)),
            *color_block(128, 128),
            nn.Dropout(p=0.5),
            nn.Conv2d(128, out_c, 1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, img_input):
        return self.model(img_input)



class ConditionNetDDFT(nn.Module):
    def __init__(self, nf=64, classifier='color_condition', cond_c=64):
        super(ConditionNetDDFT, self).__init__()

        if classifier=='color_condition':
            self.classifier = Color_ConditionFT(out_c=cond_c)

        self.GFM_nf = 64
        self.KT=4
        self.SVDN=nf*self.KT
        self.nf=nf

        self.cond_scale_first = nn.Linear(cond_c, self.SVDN)
        self.cond_scale_first1 = nn.Linear(cond_c, self.SVDN)
        self.cond_scale_HR = nn.Linear(cond_c, self.SVDN)
        self.cond_scale_HR1 = nn.Linear(cond_c, self.SVDN)
        self.cond_scale_last = nn.Linear(cond_c, 3)

        self.cond_shift_first = nn.Linear(cond_c, nf)
        self.cond_shift_HR = nn.Linear(cond_c, nf)
        self.cond_shift_last = nn.Linear(cond_c, 3)

        self.conv_first = nn.Sequential(
            nn.Conv2d(3, nf, 1, 1),
            DDFPack(nf,version='f')
        )
        self.HRconv= nn.Sequential(
            DDFPack(nf,version='f'),
            nn.Conv2d(nf, nf, 1, 1)
        )
        self.conv_last = nn.Sequential(
            DDFPack(nf,version='f'),
            nn.Conv2d(nf, 3, 1, 1)
        )
        self.act = nn.ReLU(inplace=True)
        self.non_linear0=Non_Linear(nf)
        self.non_linear1=Non_Linear(nf)

    def forward(self, x):
        content = x[0]
        condition = x[1]
        # b,3,h,w
        fea = self.classifier(condition).squeeze(2).squeeze(2)

        scale_first0 = self.cond_scale_first(fea).view(fea.shape[0],self.nf,self.KT)
        scale_first1 = self.cond_scale_first1(fea).view(fea.shape[0],self.KT,self.nf)
        scale_first = torch.matmul(scale_first0,scale_first1).view(fea.shape[0],self.nf,self.nf,1,1)
        shift_first = self.cond_shift_first(fea)

        scale_HR0 = self.cond_scale_HR(fea).view(fea.shape[0],self.nf,self.KT)
        scale_HR1 = self.cond_scale_HR1(fea).view(fea.shape[0],self.KT,self.nf)
        scale_HR = torch.matmul(scale_HR0,scale_HR1).view(fea.shape[0],self.nf,self.nf,1,1)
        shift_HR = self.cond_shift_HR(fea)

        # TODO 明天添加自适应转换
        
        scale_last = self.cond_scale_last(fea)
        shift_last = self.cond_shift_last(fea)

        out = self.conv_first(content) # shape =[B,C,H,W] [B,C,H*W]
        out=self.non_linear0(out,scale_first) + shift_first.view(-1, self.GFM_nf, 1, 1) + out
        # out = out * scale_first.view(-1, self.GFM_nf, 1, 1) + shift_first.view(-1, self.GFM_nf, 1, 1) + out
        out = self.act(out)

        out = self.HRconv(out)
        out=self.non_linear1(out,scale_HR) + shift_HR.view(-1, self.GFM_nf, 1, 1) + out
        # out = out * scale_HR.view(-1, self.GFM_nf, 1, 1) + shift_HR.view(-1, self.GFM_nf, 1, 1) + out
        out = self.act(out)

        out = self.conv_last(out)
        out = out * scale_last.view(-1, 3, 1, 1) + shift_last.view(-1, 3, 1, 1) + out

        return out





class ConditionNetESADDF(nn.Module): 
    def __init__(self, nf=64, classifier='color_condition', cond_c=16):
        super(ConditionNetESADDF, self).__init__()

        if classifier=='color_condition':
            self.classifier = Color_Condition(out_c=cond_c)

        self.GFM_nf = 64
        
        self.cond_scale_first = nn.Linear(cond_c, nf)
        self.cond_scale_HR = nn.Linear(cond_c, nf)
        self.cond_scale_last = nn.Linear(cond_c, 3)

        self.cond_shift_first = nn.Linear(cond_c, nf)
        self.cond_shift_HR = nn.Linear(cond_c, nf)
        self.cond_shift_last = nn.Linear(cond_c, 3)
        self.conv_first = nn.Sequential(
            nn.Conv2d(3, nf, 1, 1),
            DDFPack(nf,version='f')
            # ESA(nf)
        )
        self.HRconv= nn.Sequential(
            DDFPack(nf,version='f'),
            
            # ContextRes(nf),
            nn.Conv2d(nf, nf, 1, 1)
        )
        self.conv_last = nn.Sequential(
            DDFPack(nf,version='f'),
            # ESA(nf),
            ContextDense(24,nf,4),
            nn.Conv2d(nf, 3, 1, 1)
        )
        
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        content = x[0]+torch.zeros(1, dtype=x[0].dtype, device=x[0].device, requires_grad=True)
        condition = x[1]+torch.zeros(1, dtype=x[0].dtype, device=x[0].device, requires_grad=True)
        # b,3,h,w
        fea = self.classifier(condition).squeeze(2).squeeze(2)

        scale_first = self.cond_scale_first(fea)
        shift_first = self.cond_shift_first(fea)

        scale_HR = self.cond_scale_HR(fea)
        shift_HR = self.cond_shift_HR(fea)

        scale_last = self.cond_scale_last(fea)
        shift_last = self.cond_shift_last(fea)

        out = self.conv_first(content)
        out = out * scale_first.view(-1, self.GFM_nf, 1, 1) + shift_first.view(-1, self.GFM_nf, 1, 1) + out
        out = self.act(out)

        out = self.HRconv(out)
        # out = checkpoint(self.HRconv,out)
        out = out * scale_HR.view(-1, self.GFM_nf, 1, 1) + shift_HR.view(-1, self.GFM_nf, 1, 1) + out
        out = self.act(out)

        out = self.conv_last(out)
        out = out * scale_last.view(-1, 3, 1, 1) + shift_last.view(-1, 3, 1, 1) + out

        return out


class Color_ConditionLT(nn.Module):
    def __init__(self, in_channels=3, out_c=3):
        super(Color_ConditionLT, self).__init__()

        self.model = nn.Sequential(
            *color_block(3, 16, normalization=True),
            *color_block(16, 32, normalization=True),
            *color_block(32, 64, normalization=True),
            *color_block(64, 128, normalization=True),
            *color_block(128, 128),
            nn.Dropout(p=0.5)
        )
        self.globalvector=nn.Sequential(
            nn.Conv2d(128, out_c, 1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d(1))
        self.localft=nn.Sequential(
            nn.Upsample(scale_factor=2)
        )
    def forward(self, img_input):
        feat = self.model(img_input)
        vector = self.globalvector(feat)
        localfeat = self.localft(feat)
        return vector,localfeat

class Color_ConditionUnet(nn.Module):
    def __init__(self, in_channels=3, out_c=6):
        super(Color_ConditionUnet, self).__init__()
        self.downblocks=nn.ModuleList()
        self.downblocks.append(CB(3, 16, normalization=True)) # /2
        self.downblocks.append(CB(16, 32, normalization=True)) # /4
        self.downblocks.append(CB(32, 64, normalization=True)) # /8
        self.downblocks.append(CB(64, 128, normalization=True)) # /16
        self.downblocks.append(CB(128, 128, normalization=False))# /32
        self.global_vector=nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(128, out_c, 1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d(1))
        self.drop=nn.Dropout(p=0.5)
        # self.upsample0=nn.Upsample(scale_factor=2)
        # self.upsample1=nn.Upsample(scale_factor=4)
        # self.upsample2=nn.Upsample(scale_factor=4)
        
        # self.conv0 = nn.Sequential(
        #     nn.Conv2d(128,128,1),
        #     nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=True),
        #     nn.LeakyReLU(0.2)        
        #     )
        
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(128,32,1),
        #     nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=True),
        #     nn.LeakyReLU(0.2)        
        #     )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128,out_c,1),
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=True),
            nn.LeakyReLU(0.2)        
            )
        # relu

    # pooling = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=True)
    # act = nn.LeakyReLU(0.2)        

        
    def forward(self,o):
        x=o
        # downfeats=[]
        for fe in self.downblocks:
            x=fe(x)
            # downfeats.append(x)
        vector=self.global_vector(x)
        feat1=F.upsample(x,(o.shape[2],o.shape[3])) 
        feat1=self.drop(feat1)
        feat1=self.conv2(feat1)   # 128
        
        # feat4 = self.upsample1(self.conv1(feat16))+downfeats[-4]  # 32
        # feat4=self.drop(feat4)
        # feat1 = self.conv2(self.upsample2(feat4))
        if False:
            x=x.cpu()
            torch.cuda.empty_cache()
        return vector.squeeze(2).squeeze(2),feat1
 





class HDCFM(nn.Module):
    def __init__(self, nf=64, classifier='color_condition', cond_c=6):
        super(HDCFM, self).__init__()

        if classifier=='color_condition':
            self.classifier = Color_ConditionUnet(out_c=cond_c)

        self.GFM_nf = nf

        self.cond_scale_first = nn.Linear(cond_c, nf)
        self.cond_scale_HR = nn.Linear(cond_c, nf)
        self.cond_scale_last = nn.Linear(cond_c, 3)
        self.conv_feat_first_scale=nn.Conv2d(cond_c, nf, 1)
        self.conv_feat_HR_scale=nn.Conv2d(cond_c, nf, 1)
        # self.conv_feat_last_scale=nn.Conv2d(cond_c, 3, 1)
        
        self.cond_shift_first = nn.Linear(cond_c, nf)
        self.cond_shift_HR = nn.Linear(cond_c, nf)
        self.cond_shift_last = nn.Linear(cond_c, 3)
        self.conv_feat_first_shift=nn.Conv2d(cond_c, nf, 1)
        self.conv_feat_HR_shift=nn.Conv2d(cond_c, nf, 1)
        # self.conv_feat_last_shift=nn.Conv2d(cond_c, 3, 1)
        
        self.conv_first = nn.Sequential(
            nn.Conv2d(3, nf, 1, 1),
            ContextBlock(nf,0.25,pooling_type='avg',fusion_types=('channel_mul',)),
            DDFPack(nf,version='f'),
            ContextBlock(nf,0.25,pooling_type='avg',fusion_types=('channel_mul',)),
            DDFPack(nf,version='f')
        )

        self.HRconv= nn.Sequential(
            DDFPack(nf,version='f'),
            ContextBlock(nf,0.25,pooling_type='avg',fusion_types=('channel_mul',)),
            nn.Conv2d(nf, nf, 1, 1)
        )
        
        self.conv_last = nn.Sequential(
            DDFPack(nf,version='f'),
            ContextBlock(nf,0.25,pooling_type='avg',fusion_types=('channel_mul',)),
            DDFPack(nf,version='f'),
            ContextBlock(nf,0.25,pooling_type='avg',fusion_types=('channel_mul',)),
            nn.Conv2d(nf, 3, 1, 1)
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        content = x
        condition = x
        # b,3,h,w
        fea,feat1 = self.classifier(condition)


        scale_first = self.cond_scale_first(fea)
        shift_first = self.cond_shift_first(fea)
        feat_scale_first=self.conv_feat_first_scale(feat1)
        feat_shift_first=self.conv_feat_first_shift(feat1)
        
        out = self.conv_first(content)
        
        out = out * scale_first.view(-1, self.GFM_nf, 1, 1) + shift_first.view(-1, self.GFM_nf, 1, 1) + out
        out = out * feat_scale_first +feat_shift_first + out
        out = self.act(out)

        
        scale_HR = self.cond_scale_HR(fea)
        shift_HR = self.cond_shift_HR(fea)
        feat_scale_HR=self.conv_feat_HR_scale(feat1)
        feat_shift_HR=self.conv_feat_HR_shift(feat1)
        
        out = self.HRconv(out)
        out = out * scale_HR.view(-1, self.GFM_nf, 1, 1) + shift_HR.view(-1, self.GFM_nf, 1, 1) + out
        out = out * feat_scale_HR + feat_shift_HR + out
        out = self.act(out)


        scale_last = self.cond_scale_last(fea)
        shift_last = self.cond_shift_last(fea)
        out = self.conv_last(out)
        out = out * scale_last.view(-1, 3, 1, 1) + shift_last.view(-1, 3, 1, 1) + out

        return out

