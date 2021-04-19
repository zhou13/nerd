import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sym.config import CI, CM
from sym.models.hourglass_pose import Bottleneck2D, Hourglass, hg
from sym.models.utils import (
    BasicBlock,
    ConvBnReLU,
    ConvBnReLU3D,
    ConvTrBnReLU3D,
    depth_softargmin,
    resample,
    warp,
)
from sym.utils import benchmark


class MVSNet(nn.Module):
    def __init__(self):
        super(MVSNet, self).__init__()
        self.feature_network = FeatureNet()
        self.volume_network = VolumeNet()
        self.detection_network = DetectionNet()
        self.refine = RefineNet()
        self.fc = nn.Conv2d(64, 32, 1)
        # self.fc_ablation = nn.Conv3d(32, 64, 1)

        if CM.cat_depth_feature:
            depth_feature = torch.zeros([1, 32, CM.num_depth, 1, 1])
            div = CM.num_depth // 32
            for i in range(CM.num_depth):
                depth_feature[0, i // div, i, 0, 0] = (i % div + 1) / div
            self.register_buffer("depth_feature", depth_feature)

    def forward(self, image, S, w, gamma):
        D = len(gamma)

        # step 1. feature extraction
        x = self.feature_network(image)  # [N, 32, H, W]
        N, _, H, W = x.shape
        c = 1

        # step 2. differentiable homograph, build cost volume
        # during training, duplicate images for sampling symmetric axis
        c = S.shape[1]
        x2d = x.repeat_interleave(c, dim=0)

        x = self.fc(x).repeat_interleave(c, dim=0)
        vol = [x.unsqueeze(2).repeat(1, 1, D, 1, 1)]
        if CM.cat_depth_feature:
            vol.append(self.depth_feature.repeat(N * c, 1, 1, H, W))
        S = S.view(N * c, 4, 4)
        vol.append(warp(x, S, gamma))
        vol = torch.cat(vol, 1)

        # step 3. cost volume regularization
        cost, x3d = self.volume_network(vol)
        prob = F.softmax(cost, dim=1)
        depth = depth_softargmin(prob, gamma)

        if CM.do_refine:
            depth = self.refine(depth, x)

        confidence = None
        if not self.training and CM.save_confidence:
            with torch.no_grad():
                # prob_smooth: [N, D, H/4, W/4]
                prob_smooth = F.avg_pool3d(
                    prob[:, None], (5, 1, 1), stride=1, padding=(2, 0, 0)
                )[:, 0]
                index = torch.arange(D, device=x.device, dtype=torch.float)
                depth_index = depth_softargmin(prob, index).long()
                confidence = torch.gather(prob_smooth, 1, depth_index)[:, 0]

        cls = self.detection_network(x2d, x3d, w)
        return resample(depth, 4), cls, confidence


class FeatureNet(nn.Module):
    def __init__(self):
        super().__init__()
        if hasattr(CM, "backbone2d") and CM.backbone2d == "5x5":
            self.backbone = nn.Sequential(
                nn.Conv2d(4, 32, 5, 2, 2),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 5, 1, 2),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 5, 2, 2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 5, 1, 2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
        else:
            self.backbone = nn.Sequential(
                nn.Conv2d(4, 64, 5, 2, 2),
                BasicBlock(64, 64),
                BasicBlock(64, 64),
                BasicBlock(64, 64),
                BasicBlock(64, 64, 2),
                BasicBlock(64, 64),
                BasicBlock(64, 64),
                BasicBlock(64, 64),
                BasicBlock(64, 64),
            )

    def forward(self, x):
        return self.backbone(x)


class VolumeNet(nn.Module):
    def __init__(self):
        super().__init__()
        if CM.large_volumenet:
            self.conv0 = nn.Sequential(
                ConvBnReLU3D(32 * (2 + int(CM.cat_depth_feature)), 64, 1, 1, 0),
                ConvBnReLU3D(64, 64, 1, 1, 0),
                ConvBnReLU3D(64, 8),
            )
        else:
            self.conv0 = nn.Sequential(
                ConvBnReLU3D(32 * (2 + int(CM.cat_depth_feature)), 16, 1, 1, 0),
                ConvBnReLU3D(16, 8, 1, 1, 0),
                ConvBnReLU3D(8, 8),
            )

        self.conv1 = nn.Sequential(ConvBnReLU3D(8, 16, 3, 2), ConvBnReLU3D(16, 16))
        self.conv2 = nn.Sequential(ConvBnReLU3D(16, 32, 3, 2), ConvBnReLU3D(32, 32))
        self.conv3 = nn.Sequential(ConvBnReLU3D(32, 64, 3, 2), ConvBnReLU3D(64, 64))
        self.conv4 = nn.Sequential(ConvBnReLU3D(64, 128, 3, 2), ConvBnReLU3D(128, 128))
        self.donv4 = ConvTrBnReLU3D(128, 64, stride=2)
        self.donv3 = ConvTrBnReLU3D(64, 32, stride=2)
        self.donv2 = ConvTrBnReLU3D(32, 16, stride=2)
        self.donv1 = ConvTrBnReLU3D(16, 8, stride=2)
        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)

        self.fc0 = nn.Sequential(nn.MaxPool3d(16, 16), ConvBnReLU3D(8, 16))
        self.fc1 = nn.Sequential(nn.MaxPool3d(8, 8), ConvBnReLU3D(16, 16))
        self.fc2 = nn.Sequential(nn.MaxPool3d(4, 4), ConvBnReLU3D(32, 32))
        self.fc3 = nn.Sequential(nn.MaxPool3d(2, 2), ConvBnReLU3D(64, 64))
        self.fc4 = ConvBnReLU3D(128, 128)

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        # depth branch
        if CM.enable_depth:
            x = x4
            x = x3 + self.donv4(x)
            x = x2 + self.donv3(x)
            x = x1 + self.donv2(x)
            x = x0 + self.donv1(x)
        else:
            x = x0
        x = self.prob(x)

        # detection branch
        xp = torch.cat(
            [self.fc0(x0), self.fc1(x1), self.fc2(x2), self.fc3(x3), self.fc4(x4)],
            dim=1,
        )

        return x[:, 0], xp


class DetectionNet(nn.Module):
    def __init__(self):
        super().__init__()
        dim_fc = CM.detection.fc_channel
        if not hasattr(CM.detection, "fc_pool"):
            # backward compatible
            self.flatten3d = nn.Sequential(nn.Flatten())
            self.fc3d = nn.Sequential(
                nn.Linear(16384, dim_fc),
                nn.ReLU(inplace=True),
                nn.Dropout(p=CM.detection.dropout),
                nn.Linear(dim_fc, dim_fc),
                nn.ReLU(inplace=True),
                nn.Dropout(p=CM.detection.dropout),
                nn.Linear(dim_fc, CM.detection.n_level),
            )
        else:
            if CM.detection.fc_pool:
                self.flatten3d = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(16384, dim_fc),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=CM.detection.dropout),
                )
            else:
                self.flatten3d = nn.Sequential(
                    nn.Conv3d(256, dim_fc, 1),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveMaxPool3d((1, 1, 1)),
                    nn.Flatten(),
                    nn.Linear(dim_fc, dim_fc),
                    nn.ReLU(inplace=True),
                )
            self.fc3d = nn.Sequential(
                nn.Linear(dim_fc, dim_fc),
                nn.ReLU(inplace=True),
                nn.Linear(dim_fc, CM.detection.n_level),
            )

    def forward(self, x2d, x3d, w):
        N, c, _ = w.shape
        x = self.fc3d(self.flatten3d(x3d))
        return x.view(N, c, CM.detection.n_level)


class RefineNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(33, 32, 1)
        self.hg = Hourglass(Bottleneck2D, 1, 16, 4)
        self.fc = nn.Conv2d(32, 1, 1)

    def forward(self, depth, image):
        x = torch.cat([depth, image], 1)
        x = self.fc(self.hg(self.conv(x)))
        return depth + x * 0.02
