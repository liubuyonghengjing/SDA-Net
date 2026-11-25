import torch.nn as nn
import torch
import torch.nn.functional as F
import math

# TransformerBlock
class TransformerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4, ff_dim=2048, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.LayerNorm(in_channels)
        self.attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(in_channels)
        self.ff = nn.Sequential(
            nn.Linear(in_channels, ff_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ff_dim, in_channels)
        )

        self.project = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.size()
        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x_flat + attn_out

        x_ffn = self.ff(self.norm2(x))
        x = x + x_ffn

        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = self.project(x)  # 确保输出是 out_channels
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Linear(in_channel, in_channel // reduction, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channel // reduction, in_channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x).view(b, c))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x).view(b, c))))
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return out   # 注意：这里只输出权重 (B,C,1,1)

class SpatialAttention_ECA(nn.Module):
    def __init__(self, gamma=2, b=1):
        super(SpatialAttention_ECA, self).__init__()
        self.gamma = gamma
        self.b = b
        self.sigmoid = nn.Sigmoid()

    def get_kernel_size(self, num_elements):

        k_size = int(abs((math.log(num_elements, 2) + self.b) / self.gamma))
        k_size = k_size if k_size % 2 else k_size + 1
        return k_size

    def forward(self, x):
        b, c, h, w = x.size()

        # 1. 通道平均池化 (得到空间注意力输入)
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B,1,H,W)

        # 2. 展平成序列
        seq = avg_out.view(b, 1, -1)  # (B,1,H*W)

        # 3. 计算动态卷积核大小
        k_size = self.get_kernel_size(h * w)

        # 4. 构造卷积核 (均值卷积，非可学习参数)
        weight = torch.ones(1, 1, k_size, device=x.device) / k_size

        # 5. 用 F.conv1d 进行卷积
        y = F.conv1d(seq, weight, padding=k_size // 2)

        # 6. reshape 回空间维度
        attention = self.sigmoid(y.view(b, 1, h, w))
        return attention  # 权重 (B,1,H,W)

class CASA_Sequential(nn.Module):
    def __init__(self, channel, reduction=16, gamma=2, b=1):
        """
        串联方式: 先通道注意力 (CA)，再空间注意力 (SA)
        """
        super(CASA_Sequential, self).__init__()
        self.channel_attention = ChannelAttention(channel, reduction)
        self.spatial_attention = SpatialAttention_ECA(gamma, b)

    def forward(self, x):
        # 1️⃣ 通道注意力
        ca = self.channel_attention(x)
        x = x * ca  # 先加权通道特征

        # 2️⃣ 空间注意力
        sa = self.spatial_attention(x)
        out = x * sa  # 再加权空间特征

        return out
class DualBranchBottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64, use_transformer=False):
        super(DualBranchBottleneck, self).__init__()
        width = int(out_channel * (width_per_group / 64.)) * groups
        self.use_transformer = use_transformer

        if use_transformer:
            self.transformer = TransformerBlock(in_channel, out_channel * self.expansion)
        else:
            self.conv1 = nn.Conv2d(in_channel, width, kernel_size=1, stride=1, bias=False)
            self.bn1 = nn.BatchNorm2d(width)

            self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                                   padding=1, groups=groups, bias=False)

            self.bn2 = nn.BatchNorm2d(width)
            #self.scconv = SCConv(width, stride=stride)
            self.conv3 = nn.Conv2d(width, out_channel * self.expansion, kernel_size=1, stride=1, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)

            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample

            self.casa = CASA_Sequential(out_channel * self.expansion)  # CBAM模块

    def forward(self, x):
        identity = x
        if self.use_transformer:
            out = self.transformer(x)
            if self.downsample is not None:
                identity = self.downsample(identity)
            out += identity
            out = self.relu(out)
            return out

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        #out = self.scconv(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.casa(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

# ResNeXt Bottleneck Block
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, width_per_group=64, use_transformer=False):
        super(Bottleneck, self).__init__()
        width = int(planes * (width_per_group / 64.)) * groups
        self.use_transformer = use_transformer
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

        if use_transformer:
            self.transformer = TransformerBlock(inplanes, planes * self.expansion)
        else:
            self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(width)

            self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                                   padding=1, groups=groups, bias=False)
            self.bn2 = nn.BatchNorm2d(width)

            #self.scconv = SCConv(width, stride=stride)

            self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes * self.expansion)

            self.casa = CASA_Sequential(planes * self.expansion)  # CBAM模块

    def forward(self, x):
        identity = x
        if self.use_transformer:
            out = self.transformer(x)
            if self.downsample is not None:
                identity = self.downsample(identity)
            out += identity
            out = self.relu(out)
            return out

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        #out = self.scconv(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.casa(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

# ========= ResNeXtFusion with layer1+2+3+4 fusion =========

class ResNeXtFusion(nn.Module):
    def __init__(self, num_classes=1000, include_top=True,
                 groups=32, width_per_group=4):
        super(ResNeXtFusion, self).__init__()
        self.include_top = include_top
        self.in_channel = 64
        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # layer1 用双分支 Bottleneck
        self.layer1 = self._make_layer(DualBranchBottleneck, 64, 3)
        # 后面还是原始 Bottleneck
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

        # -------- 各层特征通道对齐 (1×1 Conv) --------
        # 让每层输出统一到同样的维度，例如 256
        self.align_channels = 256
        self.conv1x1_f1 = nn.Conv2d(64 * DualBranchBottleneck.expansion, self.align_channels, kernel_size=1, bias=False)
        self.conv1x1_f2 = nn.Conv2d(128 * Bottleneck.expansion, self.align_channels, kernel_size=1, bias=False)
        self.conv1x1_f3 = nn.Conv2d(256 * Bottleneck.expansion, self.align_channels, kernel_size=1, bias=False)
        self.conv1x1_f4 = nn.Conv2d(512 * Bottleneck.expansion, self.align_channels, kernel_size=1, bias=False)
        self.bn_f1 = nn.BatchNorm2d(self.align_channels)
        self.bn_f2 = nn.BatchNorm2d(self.align_channels)
        self.bn_f3 = nn.BatchNorm2d(self.align_channels)
        self.bn_f4 = nn.BatchNorm2d(self.align_channels)

        # ------------------ 融合卷积 ------------------
        fusion_channels = self.align_channels * 4
        self.fusion_conv = nn.Conv2d(fusion_channels, 512 * Bottleneck.expansion, kernel_size=1, bias=False)
        self.bn_fusion = nn.BatchNorm2d(512 * Bottleneck.expansion)
        self.relu_fusion = nn.ReLU(inplace=False)

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

    def _make_layer(self, block, channel, block_num, stride=1, use_transformer=False):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample,
                            stride=stride, groups=self.groups, width_per_group=self.width_per_group, use_transformer=use_transformer))
        self.in_channel = channel * block.expansion
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel,
                                groups=self.groups, width_per_group=self.width_per_group, use_transformer=use_transformer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        f1 = self.layer1(x)   # 1/4
        f2 = self.layer2(f1)  # 1/8
        f3 = self.layer3(f2)  # 1/16
        f4 = self.layer4(f3)  # 1/32

        # 1×1 Conv 对齐
        f1 = self.bn_f1(self.conv1x1_f1(f1))
        f2 = self.bn_f2(self.conv1x1_f2(f2))
        f3 = self.bn_f3(self.conv1x1_f3(f3))
        f4 = self.bn_f4(self.conv1x1_f4(f4))

        # 上采样
        f2_up = F.interpolate(f2, size=f1.shape[2:], mode='bilinear', align_corners=True)
        f3_up = F.interpolate(f3, size=f1.shape[2:], mode='bilinear', align_corners=True)
        f4_up = F.interpolate(f4, size=f1.shape[2:], mode='bilinear', align_corners=True)

        # 拼接融合
        fusion = torch.cat([f1, f2_up, f3_up, f4_up], dim=1)
        fusion = self.fusion_conv(fusion)
        fusion = self.bn_fusion(fusion)
        fusion = self.relu_fusion(fusion)

        if self.include_top:
            fusion = self.avgpool(fusion)
            fusion = torch.flatten(fusion, 1)
            fusion = self.fc(fusion)
        return fusion



def resnext50_32x4d(num_classes=1000, include_top=True):
    groups = 32
    width_per_group = 4
    return ResNeXtFusion(num_classes=num_classes, include_top=include_top,
                         groups=groups, width_per_group=width_per_group)

def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNeXtFusion(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top,
                   groups=groups, width_per_group=width_per_group)
