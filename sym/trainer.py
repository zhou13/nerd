import os
import os.path as osp
import shutil
import sys
import threading
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import torch

import sym.models.mvsnet as mvs
from sym.config import CI, CM, CO
from sym.utils import benchmark


class Trainer(object):
    def __init__(
        self, device, model, optimizer, train_loader, val_loader, batch_size, out
    ):
        self.device = device

        self.model = model
        self.optim = optimizer

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batch_size = batch_size

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.epoch = 0
        self.iteration = 0
        self.mean_loss = self.best_mean_loss = 1e1000

        self.loss_labels = None
        self.avg_metrics = None
        self.metrics = np.zeros(0)

    def _loss(self, result):
        losses = result["losses"]
        metrics = result["metrics"]
        combined = dict(losses, **metrics)

        if self.loss_labels is None:
            self.loss_labels = ["sum"] + list(losses.keys()) + list(metrics.keys())
            self.metrics = np.zeros([len(self.loss_labels)])
            bar = ["progress "] + list(map("{:7}".format, self.loss_labels))
            print("\n" + "| ".join(bar + ["speed"]))
            with open(f"{self.out}/loss.csv", "a") as fout:
                print(", ".join(bar), file=fout)

        total_loss = 0
        for i, name in enumerate(self.loss_labels):
            if name == "sum":
                continue
            L = combined[name].mean()
            self.metrics[i] += L.item()
            if name in losses:
                total_loss += L
        self.metrics[0] += total_loss.item()
        return total_loss

    def validate(self):
        tprint("Running validation...", " " * 75)
        training = self.model.training
        self.model.eval()

        n_image = self.batch_size * self.iteration
        n_image1 = self.batch_size * max(0, self.iteration - 1)
        save_checkpoint = (
            n_image // CI.checkpoint_interval != n_image1 // CI.checkpoint_interval
        )

        if save_checkpoint:
            viz = osp.join(self.out, "viz", f"{self.iteration * self.batch_size:09d}")
            npz = osp.join(self.out, "npz", f"{self.iteration * self.batch_size:09d}")
            osp.exists(viz) or os.makedirs(viz)
            osp.exists(npz) or os.makedirs(npz)

        total_loss = 0
        self.metrics[...] = 0
        with torch.no_grad():
            for batch_idx, input in enumerate(self.val_loader):
                result = self.model(input, "eval")
                total_loss += self._loss(result)
                for i in range(len(input["image"])):
                    index = batch_idx * self.batch_size + i
                    if save_checkpoint:
                        if CI.checkpoint_save_prediction:
                            np.savez(
                                f"{npz}/{index:06}.npz",
                                **{
                                    k: v[i].cpu().numpy()
                                    for k, v in result["preds"].items()
                                },
                            )
                        if index < CI.num_visualization:
                            self.model.save_figures(
                                {k: v[i] for k, v in input.items()},
                                {k: v[i] for k, v in result["preds"].items()},
                                f"{viz}/{index:06}",
                            )

        with open(f"{self.out}/loss.csv", "a") as fout:
            print(
                f"{self.epoch:03}/{self.iteration * self.batch_size // 1000:04}k| "
                + ", ".join(map("{:.5f}".format, self.metrics / len(self.val_loader))),
                file=fout,
            )
        pprint(
            f"{self.epoch:03}/{self.iteration * self.batch_size // 1000:04}k| "
            + "| ".join(map("{:.5f}".format, self.metrics / len(self.val_loader))),
            " " * 7,
        )
        self.mean_loss = total_loss / len(self.val_loader)

        torch.save(
            {
                "iteration": self.iteration,
                "arch": self.model.__class__.__name__,
                "optim_state_dict": self.optim.state_dict(),
                "model_state_dict": self.model.state_dict(),
                "best_mean_loss": self.best_mean_loss,
            },
            osp.join(self.out, "checkpoint_latest.pth"),
        )
        if save_checkpoint:
            shutil.copy(
                osp.join(self.out, "checkpoint_latest.pth"),
                osp.join(npz, "checkpoint.pth"),
            )
        if self.mean_loss < self.best_mean_loss:
            self.best_mean_loss = self.mean_loss
            shutil.copy(
                osp.join(self.out, "checkpoint_latest.pth"),
                osp.join(self.out, "checkpoint_best.pth"),
            )

        if training:
            self.model.train()

    def train_epoch(self):
        self.model.train()
        time = timer()
        for batch_idx, input in enumerate(self.train_loader):
            self.optim.zero_grad()
            self.metrics[...] = 0

            result = self.model(input, "train")

            loss = self._loss(result)
            if np.isnan(loss.item()):
                raise ValueError("loss is nan while training")
            loss.backward()
            self.optim.step()

            if self.avg_metrics is None:
                self.avg_metrics = self.metrics
            else:
                self.avg_metrics = self.avg_metrics * 0.95 + self.metrics * 0.05
            self.iteration += 1

            if self.iteration % 4 == 0:
                tprint(
                    f"{self.epoch:03}/{self.iteration * self.batch_size // 1000:04}k| "
                    + "| ".join(map("{:.5f}".format, self.avg_metrics))
                    + f"| {4 * self.batch_size / (timer() - time):04.1f} "
                )
                time = timer()
            n_image = self.batch_size * self.iteration
            n_image1 = self.batch_size * max(0, self.iteration - 1)
            if (
                n_image // CI.validation_interval != n_image1 // CI.validation_interval
                or n_image == CI.validation_debug
            ):
                self.validate()
                time = timer()

    def train(self):
        epoch_size = len(self.train_loader)
        start_epoch = self.iteration // epoch_size
        for self.epoch in range(start_epoch, CO.max_epoch):
            if self.epoch in CO.lr_decay_epoch:
                self.optim.param_groups[0]["lr"] /= 10
            self.train_epoch()


def tprint(*args):
    """Temporarily prints things on the screen"""
    print("\r\x1b[2K", end="")
    print(*args, end="", flush=True)


def pprint(*args):
    """Permanently prints things on the screen"""
    print("\r\x1b[2K", end="")
    print(*args, flush=True)
