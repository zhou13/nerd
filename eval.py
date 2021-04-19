#!/usr/bin/env python3
"""Compute vanishing points using corase-to-fine method on the evaluation dataset.
Usage:
    eval.py [options] <yaml-config> <checkpoint>
    eval.py ( -h | --help )

Arguments:
   <yaml-config>                 Path to the yaml hyper-parameter file
   <checkpoint>                  Path to the checkpoint

Options:
   -h --help                     Show this screen
   -d --devices <devices>        Comma seperated GPU devices [default: 0]
   -o --output <output>          Path to the output AA curve [default: error.npz]
   --visualize <outdir>          Output visualization related files
   --suffix <suffix>             File suffix of visualization [default: nerd]
   --split <split>               Split for testing [default: test_all]
   --noimshow                    Do not show result
"""

import math
import os
import os.path as osp
import pprint
import random
import shlex
import subprocess
import sys
import threading

import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import numpy as np
import numpy.linalg as LA
import skimage.io
import torch
from docopt import docopt
from tqdm import tqdm

import sym
from sym.config import CI, CM, CO, C
from sym.datasets import Pix3dDataset, ShapeNetDataset, to_label, w2S
from sym.models import SymmetryNet
from sym.utils import np_eigen_scale_invariant, np_kitti_error

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 20


def AA(x, y, threshold):
    index = np.searchsorted(x, threshold)
    x = np.concatenate([x[:index], [threshold]])
    y = np.concatenate([y[:index], [threshold]])
    return ((x[1:] - x[:-1]) * y[:-1]).sum() / threshold


def main():
    args = docopt(__doc__)
    config_file = args["<yaml-config>"]
    C.update(C.from_yaml(filename=config_file))
    CI.update(C.io)
    CM.update(C.model)
    CO.update(C.optim)
    pprint.pprint(C, indent=4)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    device_name = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = args["--devices"]
    if torch.cuda.is_available():
        device_name = "cuda"
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        print("Let's use", torch.cuda.device_count(), "GPU(s)!")
    else:
        print("CUDA is not available")
    device = torch.device(device_name)

    print("Working on", args["<checkpoint>"])
    checkpoint = torch.load(args["<checkpoint>"])
    model = sym.models.SymmetryNet().to(device)
    model = sym.utils.MyDataParallel(
        model, device_ids=list(range(args["--devices"].count(",") + 1))
    )
    missing, _ = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert len(missing) == 0
    model.eval()

    if CI.dataset == "ShapeNet":
        Dataset = ShapeNetDataset
    elif CI.dataset == "Pix3D":
        Dataset = Pix3dDataset
    else:
        raise NotImplementedError
    split = args["--split"]
    # split = "test_all"
    # if "only_car_plane_chair" in CI and CI.only_car_plane_chair:
    #     split = "test_unseen_all"
    # split = "test-1000"
    # if "only_car_plane_chair" in CI and CI.only_car_plane_chair:
    #     split = "test_unseen-1000"

    loader = torch.utils.data.DataLoader(
        Dataset(C.io.datadir, split=split),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    fpath = args["--output"]
    print("save to", fpath)

    thetas = [np.pi / 2] + CM.detection.theta

    err_normal = []
    err_depth_avg = []
    err_depth_SIL = []
    err_depth_AIO = []
    w_pd = []
    w_gt = []
    # D_pd = []
    # D_gt = []

    loader_tqdm = tqdm(loader)
    for batch_idx, input in enumerate(loader_tqdm):
        score = []
        depth = []

        w0 = input["w0"].cpu().numpy()[0]
        depth_gt = input["depth"].cpu().numpy()[0][0]
        H, W = depth_gt.shape
        Rt = input["RT"].cpu().numpy()[0]
        w = np.array([0, 0, 1])
        ww = []

        # print(input["fname"][0].cpu().numpy().tostring().decode("ascii"))

        for i in range(CM.detection.n_level):
            ws, S = sample_reflection(input, w, thetas[i])
            with torch.no_grad():
                input["S"] = torch.tensor(S[None]).float()
                input["w"] = torch.tensor(ws).float()[None]
                result = model(input, "test")
            score = result["preds"]["score"].cpu().numpy()[0, :, i]
            depth = result["preds"]["depth"].cpu().numpy()[0, :]
            del result

            # visualize(ws, score, w0)
            best_w = np.argmax(score)
            w = ws[best_w]
            ww.append(w)

        # rescale depth according to the ||w||_2
        depth_pd = depth[best_w] * abs(Rt[2][3])
        depth_gt = depth_gt * abs(Rt[2][3])
        w /= abs(Rt[2][3])
        w0 /= abs(Rt[2][3])

        mask = np.logical_and(depth_gt > 0, depth_pd > 0)
        diff = np.abs(depth_gt - depth_pd)[mask]

        if args["--visualize"]:
            fname = input["fname"].cpu().numpy()[0].tostring().decode("ascii")
            fname = fname.rstrip("\x00")
            if CI.dataset == "ShapeNet":
                fname = fname[::-1].replace("/", "_", 1)[::-1]

            fname_pd = f"{args['--visualize']}/{fname}_{args['--suffix']}.npz"
            fname_gt = f"{args['--visualize']}/{fname}_gt.npz"
            os.makedirs(osp.dirname(fname_pd), exist_ok=True)
            np.savez(fname_pd, w=w, ww=np.array(ww))
            np.savez(fname_gt, w=w0)
            # np.savez(fname_pd, depth=depth_pd, w=w)
            # np.savez(fname_gt, depth=depth_gt, w=w0)

        err_depth_avg += [np.average(diff)]
        err_depth_SIL += [np_eigen_scale_invariant(depth_pd, depth_gt, mask)]
        err_depth_AIO.append(np_kitti_error(depth_gt, depth_pd, mask))
        err_normal += [np.arccos(min(1, abs(w @ w0 / LA.norm(w0) / LA.norm(w))))]
        w_pd += [w]
        w_gt += [w0]

    err_normal = np.sort(np.array(err_normal)) / np.pi * 180
    err_depth_avg = np.sort(err_depth_avg)
    err_depth_SIL = np.sort(err_depth_SIL)
    err_depth_AIO = np.average(err_depth_AIO, axis=0)

    print("avg:", np.average(err_normal))
    print("med:", err_normal[len(err_normal) // 2])
    print("<0d: ", np.sum(err_normal < 0.5) / len(err_normal))
    print("<1d: ", np.sum(err_normal < 1) / len(err_normal))
    print("<2d: ", np.sum(err_normal < 2) / len(err_normal))
    print("<5d: ", np.sum(err_normal < 5) / len(err_normal))
    print(
        "AIO: ",
        np.sum(err_normal < 0.5) / len(err_normal),
        np.sum(err_normal < 1) / len(err_normal),
        np.sum(err_normal < 2) / len(err_normal),
        np.sum(err_normal < 5) / len(err_normal),
    )

    labels = ["abs_rel", "sq_rel", "rmse", "rmse_log", "sil", "a1", "a2", "a3"]
    for i, name in enumerate(labels):
        print(f"{name}:", err_depth_AIO[i])

    np.savez(
        fpath,
        err_depth_avg=err_depth_avg,
        err_depth_SIL=err_depth_SIL,
        err_depth_AIO=err_depth_AIO,
        err_normal=err_normal,
        w_pd=np.array(w_pd),
        w_gt=np.array(w_gt),
        # D_pd=np.array(D_pd),
        # D_gt=np.array(D_gt),
    )

    # if not args["--noimshow"]:
    #     y = (1 + np.arange(len(err_normal))) / len(err_normal)
    #     plt.figure()
    #     plt.plot(err_normal, y)
    #     plt.xlim([0, 5])
    #     plt.grid()
    #     plt.title("normal error")
    #     plt.legend()
    #     plt.show()


def sample_sphere(v, alpha, num_pts):
    def orth(v):
        x, y, z = v
        o = np.array([0.0, -z, y] if abs(x) < abs(y) else [-z, 0.0, x])
        o /= LA.norm(o)
        return o

    v1 = orth(v)
    v2 = np.cross(v, v1)
    v, v1, v2 = v[:, None], v1[:, None], v2[:, None]
    indices = np.linspace(1, num_pts, num_pts)
    phi = np.arccos(1 + (math.cos(alpha) - 1) * indices / num_pts)
    theta = np.pi * (1 + 5 ** 0.5) * indices
    r = np.sin(phi)
    w = (v * np.cos(phi) + r * (v1 * np.cos(theta) + v2 * np.sin(theta))).T
    return w


def sample_reflection(input, v, alpha):
    K = input["K"].cpu().numpy()[0]
    ws = sample_sphere(v / LA.norm(v), alpha, CM.detection.n_theta)
    ws /= ws[:, 2:]
    Ss = np.array([K @ w2S(w) @ LA.inv(K) for w in ws])
    return ws, Ss


# count = 0


def visualize(ws, score, w0):
    w0 /= LA.norm(w0) / 1.02
    ws /= LA.norm(ws, axis=1, keepdims=True) / 1.02
    ax = plt.figure(figsize=(10, 6)).add_subplot(111, projection="3d")

    # ax.set_box_aspect((1, 1, 1))
    # ax.set_box_aspect((1, 1, 1))
    ax.view_init(27, -22)
    ax.auto_scale_xyz([-1, 1], [-1, 1], [-1, 1])

    # draw a hemisphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi / 2, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color="g", alpha=0.3)

    # draw sampled points
    _ = ax.scatter(ws[:, 0], ws[:, 1], ws[:, 2], c=score)
    ax.scatter(w0[0], w0[1], w0[2], c="red", marker="^")
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-0, 1)
    # plt.colorbar(cb)
    # global count
    # ax.set_title(f"Coarse-to-fine Inference Round {count+1}", pad=10)
    # plt.savefig(f"{count}.pdf")
    # count += 1
    plt.show()


if __name__ == "__main__":
    main()
