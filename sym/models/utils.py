import torch
import torch.nn as nn
import torch.nn.functional as F


class ReLUDropout(torch.nn.Dropout):
    def forward(self, input):
        return relu_dropout(
            input, p=self.p, training=self.training, inplace=self.inplace
        )


def relu_dropout(x, p=0, inplace=False, training=False):
    if not training or p == 0:
        return x.clamp_(min=0) if inplace else x.clamp(min=0)

    p1m = 1 - p
    mask = torch.rand_like(x) < p1m
    mask &= x > 0
    mask.logical_not_()
    return (
        x.masked_fill_(mask, 0).div_(p1m)
        if inplace
        else x.masked_fill(mask, 0).div(p1m)
    )


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1)
        self.stride = stride

        self.resample = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.resample = nn.Conv2d(in_planes, planes, 1, stride=stride)

    def forward(self, x):
        x = F.relu(self.bn1(x), inplace=True)
        residual = self.resample(x)
        x = self.conv1(x)
        x = self.conv2(F.relu(self.bn2(x)))
        x += residual
        return x


class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvTrBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvTrBnReLU3D, self).__init__()
        self.conv = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


def resample(x, scale_factor):
    return F.interpolate(
        x, scale_factor=scale_factor, mode="bilinear", align_corners=False
    )


def warp(x0, S, gamma):
    """Wrap the feature maps to other views given the depth.

    Arguments:
        x0: [N, C, H, W]
        S: [N, 4, 4]
        gamma: [D]

    Returns:
        3D cost volume of shape [N, C, D, H, W]
    """
    N, C, H, W = x0.shape
    D = len(gamma)
    with torch.no_grad():
        y, x = torch.meshgrid(
            [
                torch.arange(0, H, dtype=torch.float32, device=x0.device) + 0.5,
                torch.arange(0, W, dtype=torch.float32, device=x0.device) + 0.5,
            ]
        )
        # make x, y, z have shape [D*H*W]
        x = (x.flatten() * 2 / W - 1).repeat(D)
        y = (1 - y.flatten() * 2 / H).repeat(D)
        z = gamma.repeat_interleave(H * W)
        # S: [N, 4, 4]
        xyz = torch.stack([x * z, y * z, z, torch.ones_like(x)])[None]  # [1, 4, DHW]
        p = torch.matmul(S, xyz).view(N, 4, D, H, W)  # [N, 4, D, H, W]
        p = p[:, :2] / (p[:, 2:3] * p[:, 3:4]).clamp(min=1e-6)  # [N, 2, D, H, W]
        grid = torch.stack([p[:, 0], -p[:, 1]], 4).view(N, D * H, W, 2)
    return F.grid_sample(x0, grid, align_corners=False).view(N, C, D, H, W)


def depth_softargmin(cost, gamma):
    """Return a depth map with shape [N, 1, H, W].

    Arguments:
        cost: 3D cost volume tensor of shape [N, D, H, W]
        gamma: discrete depth values [D]

    Returns:
        a depth map with shape [N, 1, H, W]
    """
    return torch.sum(cost * gamma[:, None, None], 1, keepdim=True)
