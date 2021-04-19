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
from sym.models.hourglass_pose import hg
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
        self.detection_loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, input, mode):
        depth_gt = input["depth"]
        if CM.loss == "Ordinal":
            target = (depth_gt > self.gamma[:, None, None]).float()

        N, _, H, W = input["image"].shape
        depth, y, confidence = self.backbone(
            input["image"], input["S"], input["w"], self.gamma
        )

        losses = {}
        n_sample = input["S"].shape[1]
        depth_gt = depth_gt.repeat_interleave(n_sample, dim=0)

        mask = (depth_gt > 0).float()
        mask_sum = mask.sum((2, 3)).clamp(min=1e-6)

        if CM.enable_depth:
            if CM.loss == "L1":
                L_depth = ((depth - depth_gt).abs() * mask).sum((2, 3)) / mask_sum
            elif CM.loss == "L1_smooth":
                L_depth = (l1_smooth(depth, depth_gt) * mask).sum((2, 3)) / mask_sum
            elif CM.loss == "L2":
                L_depth = ((depth - depth_gt) ** 2 * mask).sum((2, 3)) / mask_sum
            elif CM.loss == "Ordinal":
                loss = F.binary_cross_entropy_with_logits(
                    depth, target, reduction="none"
                )
                L_depth = (loss * mask).sum((2, 3)) / mask_sum
            else:
                raise NotImplementedError

            if mode != "test":
                L_depth = L_depth.view(N, -1)
                for i in range(n_sample):
                    losses[f"dep{i}"] = (
                        L_depth[:, i] * CM.weight_depth[i] * CM.weight_depth_
                    )

        if mode != "test":
            y_gt = input["y"]
            L_detection = self.detection_loss(y, y_gt)
            maskn = (y_gt == 0).float()
            maskp = (y_gt == 1).float()
            for i in range(CM.detection.n_level):
                assert maskn[..., i].sum().item() != 0
                assert maskp[..., i].sum().item() != 0
                lneg = (L_detection[..., i] * maskn[..., i]).sum() / maskn[..., i].sum()
                lpos = (L_detection[..., i] * maskp[..., i]).sum() / maskp[..., i].sum()
                losses[f"det{i}"] = (lneg + lpos)[None] * CM.weight_detection

        preds = {}
        if mode != "train":
            depth = depth.view(N, -1, H, W)
            preds["score"] = y.sigmoid()
            preds["depth"] = depth
            if CM.save_confidence:
                preds["confidence"] = confidence

        return {
            "losses": losses,
            "preds": preds,
            "metrics": {},
        }

    def save_figures(self, input, preds, prefix):
        return

        plt.rcParams["figure.figsize"] = (24, 24)
        image = np.rollaxis(input["image"].cpu().numpy(), 0, 3)

        scores = preds["score"].cpu().numpy()
        ys = input["y"].cpu().numpy()
        w = input["w"].cpu().numpy()

        for idx, (w0, score, yy) in enumerate(zip(w, scores, ys)):
            D = 256
            x = -w0[0] / w0[2] * 2.18701 * D / 2 + D / 2
            y = w0[1] / w0[2] * 2.18701 * D / 2 + D / 2
            plt.imshow(image)
            plt.scatter(x, y, color="r")
            for xy in np.linspace(0, D, 10):
                plt.plot(
                    [x, xy, x, xy, x, 0, x, D - 1],
                    [y, 0, y, D - 1, y, xy, y, xy],
                    color="r",
                )
            plt.text(
                100,
                100,
                " ".join(map("{:.3f}".format, score))
                + "\n"
                + " ".join(map("{:.3f}".format, yy)),
                bbox=dict(facecolor="green"),
                fontsize=12,
            )
            plt.xlim(0, D)
            plt.ylim(D, 0)
            plt.savefig(f"{prefix}_{idx}_image.jpg"), plt.close()

        # depth_gt, depth_pd = input["depth"].cpu().numpy(), preds["depth"].cpu().numpy()
        # depth_pd = depth_pd[:1]
        # depth_pd[depth_gt == 0] = 0
        #
        # plt.figure(), plt.imshow(image), plt.colorbar()
        # plt.savefig(f"{prefix}__image.jpg"), plt.close()
        # plt.figure(), plt.imshow(depth_gt[0], vmin=CI.depth_min, vmax=CI.depth_max)
        # plt.colorbar(), plt.savefig(f"{prefix}_depth_GT.jpg"), plt.close()
        # plt.figure(), plt.imshow(depth_pd[0], vmin=CI.depth_min, vmax=CI.depth_max)
        # plt.colorbar(), plt.savefig(f"{prefix}_depth_PD.jpg"), plt.close()


def l1_smooth(input, target, beta=0.04):
    n = torch.abs(input - target)
    cond = n < beta
    return torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
