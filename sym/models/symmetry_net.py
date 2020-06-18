import itertools
import math
import random
import sys
from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import torch
import torch.nn as nn
import torch.nn.functional as F

from sym.config import CI, CM
from sym.models.mvsnet import MVSNet
from sym.utils import benchmark


class SymmetryNet(nn.Module):
    def __init__(self):
        super().__init__()

        # gamma is depth
        gamma = np.linspace(CI.depth_min, CI.depth_max, CM.num_depth, dtype=np.float32)
        gamma2 = (np.r_[gamma, gamma[-1] * 2] + np.r_[0, gamma]) / 2
        self.register_buffer("gamma", torch.from_numpy(gamma))
        self.register_buffer("gamma2", torch.from_numpy(gamma2))
        self.backbone = MVSNet()
        if CM.detection.enabled:
            self.detection_loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, input, mode):
        depth_gt = input["depth"]

        N, _, H, W = input["image"].shape
        xs, y, confidence = self.backbone(
            input["image"], input["S"], input["w"], self.gamma
        )

        losses = {}
        if CM.detection.enabled:
            n_sym = input["S"].shape[1]
            depth_gt = depth_gt.repeat_interleave(n_sym, dim=0)

        mask = (depth_gt > 0).float()
        mask_sum = mask.sum((2, 3)).clamp(min=1e-6)
        for i, x in enumerate(xs):
            if CM.loss == "L1":
                L = ((x - depth_gt).abs() * mask).sum((2, 3)) / mask_sum
            elif CM.loss == "L1_smooth":
                L = (l1_smooth(x, depth_gt) * mask).sum((2, 3)) / mask_sum
            elif CM.loss == "L2":
                L = ((x - depth_gt) ** 2 * mask).sum((2, 3)) / mask_sum
            else:
                raise NotImplementedError
            if mode != "test":
                if CM.detection.enabled:
                    L = L.view(N, -1)
                    for j in range(n_sym):
                        losses[f"dep{j}/{i}"] = L[:, j] * CM.detection.weight_depth[j]
                else:
                    losses[f"dep/{i}"] = L

        if CM.detection.enabled and mode != "test":
            y_gt = input["y"]
            L = self.detection_loss(y, y_gt)
            maskn = (y_gt == 0).float()
            maskp = (y_gt == 1).float()
            for i in range(CM.detection.n_level):
                assert maskn[..., i].sum().item() != 0
                assert maskp[..., i].sum().item() != 0
                lneg = (L[..., i] * maskn[..., i]).sum() / maskn[..., i].sum()
                lpos = (L[..., i] * maskp[..., i]).sum() / maskp[..., i].sum()
                losses[f"det{i}"] = (lneg + lpos)[None] * CM.detection.weight_detection

        preds = {}
        if mode != "train":
            if CM.detection.enabled:
                x = x.view(N, -1, H, W)
                preds["score"] = y.sigmoid()
            preds["depth"] = x
            if CM.save_confidence:
                preds["confidence"] = confidence

        return {
            "losses": losses,
            "preds": preds,
            "metrics": {},
        }

    def save_figures(self, input, preds, prefix):
        return


def l1_smooth(input, target, beta=0.04):
    n = torch.abs(input - target)
    cond = n < beta
    return torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
