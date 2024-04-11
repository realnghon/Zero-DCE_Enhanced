import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#import pytorch_colors as colors
import numpy as np


class GuidedFilter(nn.Module):
    def __init__(self, radius=1, epsilon=1e-8):
        super(GuidedFilter, self).__init__()
        self.radius = radius
        self.epsilon = epsilon

    def forward(self, I, p):
        """
        Perform guided filtering on input image I with guidance image p.
        I: Input image to be filtered.
        p: Guidance image (usually the input image itself).
        """
        N = self.box_filter(torch.ones_like(I), self.radius)

        mean_I = self.box_filter(I, self.radius) / N
        mean_p = self.box_filter(p, self.radius) / N
        mean_Ip = self.box_filter(I * p, self.radius) / N
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = self.box_filter(I * I, self.radius) / N
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + self.epsilon)
        b = mean_p - a * mean_I

        mean_a = self.box_filter(a, self.radius) / N
        mean_b = self.box_filter(b, self.radius) / N

        q = mean_a * I + mean_b
        return q


    def box_filter(self, x, radius):
        # Apply average pooling to simulate box filter
        kernel_size = radius * 2 + 1
        box_kernel = torch.ones(x.size(1), 1, kernel_size, kernel_size, device=x.device, dtype=x.dtype)
        box_kernel = box_kernel / (kernel_size * kernel_size)
        return F.conv2d(x, box_kernel, padding=radius, groups=x.size(1))

class SpatialAttentionModule(nn.Module):
    # MODIFY: 空间注意力机制
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ChannelAttentionModule(nn.Module):
    # MODIFY: 通道注意力机制
    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(num_channels, num_channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(num_channels // reduction_ratio, num_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        return self.sigmoid(y)

class enhance_net_nopool(nn.Module):
    def __init__(self):
        super(enhance_net_nopool, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.guided_filter = GuidedFilter(radius=2, epsilon=1e-3)  # MODIFY: 初始化引导滤波模块


        number_f = 32
        self.e_conv1 = nn.Conv2d(3,number_f,3,1,1,bias=True)
        self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True)
        self.e_conv3 = nn.Conv2d(number_f,number_f,3,1,1,bias=True)
        self.e_conv4 = nn.Conv2d(number_f,number_f,3,1,1,bias=True)
        self.e_conv5 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True)
        self.e_conv6 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True)
        self.e_conv7 = nn.Conv2d(number_f*2,24,3,1,1,bias=True)

        # MODIFY: 添加注意力模块
        self.sa_module = SpatialAttentionModule()
        self.ca_module = ChannelAttentionModule(number_f)

        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))

        # MODIFY: 应用空间注意力和通道注意力机制
        sa = self.sa_module(x4)
        x4 = x4 * sa
        ca = self.ca_module(x4)
        x4 = x4 * ca

        x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))

        x_r = F.tanh(self.e_conv7(torch.cat([x1,x6],1)))
        r1,r2,r3,r4,r5,r6,r7,r8 = torch.split(x_r, 3, dim=1)

        x = x + r1*(torch.pow(x,2)-x)
        x = x + r2*(torch.pow(x,2)-x)
        x = x + r3*(torch.pow(x,2)-x)
        enhance_image_1 = x + r4*(torch.pow(x,2)-x)
        x = enhance_image_1 + r5*(torch.pow(enhance_image_1,2)-enhance_image_1)
        x = x + r6*(torch.pow(x,2)-x)
        x = x + r7*(torch.pow(x,2)-x)
        enhance_image = x + r8*(torch.pow(x,2)-x)

        # MODIFY: 自适应亮度调节
        local_brightness = F.avg_pool2d(x, kernel_size=6, stride=6)
        global_brightness = x.mean(dim=[2, 3], keepdim=True)
        percentile_50 = torch.kthvalue(local_brightness.view(-1), int(0.5 * local_brightness.numel()))[0]
        brightness_mask = (local_brightness > global_brightness) & (local_brightness < percentile_50)
        brightness_mask = F.interpolate(brightness_mask.float(), size=x.size()[2:], mode='nearest')
        enhance_image_1 = enhance_image_1 * (1 - brightness_mask) + x * brightness_mask

        # MODIFY: 边缘感知平滑
        enhance_image = self.guided_filter(x, enhance_image)

        r = torch.cat([r1,r2,r3,r4,r5,r6,r7,r8],1)
        return enhance_image_1, enhance_image, r