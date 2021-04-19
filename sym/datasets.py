import json
import math
import os
import os.path as osp
import random
import sys
from glob import glob

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from sym.config import CI, CM


class ShapeNetDataset(Dataset):
    def __init__(self, rootdir, split):
        self.rootdir = rootdir
        self.split = split

        filelist = np.genfromtxt(f"{rootdir}/{split}.txt", dtype=str)
        random.seed(0)
        random.shuffle(filelist)
        filelist = [f for f in filelist if "03636649" not in f]  # remove lamps
        if (
            split == "train"
            and hasattr(CI, "only_car_plane_chair")
            and CI.only_car_plane_chair
        ):
            filelist = [
                f
                for f in filelist
                if "02691156" in f or "02958343" in f or "03001627" in f
            ]
        self.filelist = [f"{rootdir}/{f}" for f in filelist]
        self.filelist2 = filelist
        self.size = len(self.filelist)
        print(f"n{split}:", self.size)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        prefix = self.filelist[idx]
        image = cv2.imread(f"{prefix}.png", -1).astype(np.float32) / 255.0
        # plt.imshow(image)
        # plt.show()
        image = np.rollaxis(image, 2).copy()
        depth = cv2.imread(f"{prefix}_depth0001.exr", -1).astype(np.float32)[:, :, 0:1]
        depth[depth > 20] = 0
        depth[depth < 0] = 0
        depth[depth != depth] = 0
        depth = np.rollaxis(depth, 2).copy()
        with open(f"{prefix}.json") as f:
            js = json.load(f)
        Rt, K_ = np.array(js["RT"]), np.array(js["K"])
        K = np.eye(4)
        K[:3, :3] = K_[np.ix_([0, 1, 3], [0, 1, 2])]
        KRt = K @ Rt

        oprefix = self.filelist2[idx]
        fname = np.zeros([60], dtype="uint8")
        fname[: len(oprefix)] = np.frombuffer(oprefix.encode(), "uint8")

        depth_scale = 1 / abs(Rt[2][3])

        S0 = [
            KRt @ np.diagflat([1, -1, 1, 1]) @ LA.inv(KRt),
            KRt @ np.diagflat([-1, 1, 1, 1]) @ LA.inv(KRt),
            KRt @ np.diagflat([-1, -1, 1, 1]) @ LA.inv(KRt),
        ]
        result = {
            "fname": torch.tensor(fname).byte(),
            "image": torch.tensor(image).float(),
            "depth": torch.tensor(depth).float() * depth_scale,
            "K": torch.tensor(K).float(),
            "RT": torch.tensor(Rt).float(),
        }

        w0, ws = sample_plane(Rt)
        S = [K @ w2S(w) @ LA.inv(K) for w in ws]
        y = [to_label(w, w0) for w in ws]
        result["S"] = torch.tensor(S).float()
        result["y"] = torch.tensor(y).float()
        result["w"] = torch.tensor(ws).float()
        result["w0"] = torch.tensor(w0).float()

        return result


class Pix3dDataset(Dataset):
    def __init__(self, rootdir, split):
        self.rootdir = rootdir
        self.split = split

        with open(f"{rootdir}/pix3d_info.json", "r") as fin:
            data_lists = json.load(fin)
        data_valid = set(np.loadtxt(f"{rootdir}/pix3d-valid.txt", dtype=str))

        data_lists = [
            d
            for d in data_lists
            if not d["truncated"] and not d["occluded"] and d["img"][:-4] in data_valid
        ]
        random.seed(0)
        random.shuffle(data_lists)
        if self.split == "train":
            data_lists = [data_lists[i] for i in range(len(data_lists)) if i % 10 != 0]
            self.size = len(data_lists) * 2
        elif self.split == "valid":
            data_lists = [data_lists[i] for i in range(len(data_lists)) if i % 10 == 0]
            self.size = len(data_lists)
        else:
            raise NotImplementedError

        self.data_lists = data_lists

        print(f"n{split}:", self.size)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        do_flip = False
        if self.split == "train":
            do_flip = idx % 2 == 1
            idx //= 2

        data_item = self.data_lists[idx]
        fimage = osp.join(self.rootdir, data_item["img"])
        fdepth = osp.join(self.rootdir, data_item["depth"])
        fmask = osp.join(self.rootdir, data_item["mask"])

        image = cv2.imread(fimage, -1).astype(np.float32) / 255.0
        image = np.rollaxis(image, 2).copy()
        mask = cv2.imread(fmask).astype(np.float32)[None, :, :, 0] / 255.0
        image = np.concatenate([image, mask])
        depth = cv2.imread(fdepth, -1).astype(np.float32)[:, :, 0:1]
        depth[depth > 20] = 0
        depth[depth < 0] = 0
        depth[depth != depth] = 0

        depth = np.rollaxis(depth, 2).copy()
        Rt, K = np.array(data_item["Rt"]), np.array(data_item["K"])
        K[:3] *= -1

        if do_flip:
            K = np.diagflat([-1, 1, 1, 1]) @ K
            image = image[:, :, ::-1].copy()
            depth = depth[:, :, ::-1].copy()

        KRt = K @ Rt
        S0 = [KRt @ np.diagflat([-1, 1, 1, 1]) @ LA.inv(KRt)]

        w0, ws = sample_plane(Rt, np.array([1, 0, 0, 0]))
        S = [K @ w2S(w) @ LA.inv(K) for w in ws]
        y = [to_label(w, w0) for w in ws]
        depth_scale = 1 / (w0 @ Rt[:3, 3])

        # print(S0)
        # print("Rt", Rt)
        # print("S0", Rt @ np.diagflat([-1, 1, 1, 1]) @ LA.inv(Rt))
        # print("S0", w2S(w0))
        # print("w0", w0)
        # print("depth_scale", depth_scale)
        S0 = [K @ w2S(w0) @ LA.inv(K)]
        # print(S0)

        oprefix = data_item["mask"][5:-4]
        fname = np.zeros([60], dtype="uint8")
        fname[: len(oprefix)] = np.frombuffer(oprefix.encode(), "uint8")

        result = {
            "fname": torch.tensor(fname).byte(),
            "image": torch.tensor(image).float(),
            "depth": torch.tensor(depth).float() * depth_scale,
            "S0": torch.tensor(S0).float(),
            "K": torch.tensor(K).float(),
            "RT": torch.tensor(Rt).float(),
        }
        result["S"] = torch.tensor(S).float()
        result["y"] = torch.tensor(y).float()
        result["w"] = torch.tensor(ws).float()
        result["w0"] = torch.tensor(w0).float()

        return result


def sample_plane(Rt, plane=np.array([0, 1, 0, 0]), plane2=np.array([1, 0, 0, 0])):
    w0_ = LA.inv(Rt).T @ plane
    # find plane normal s.t. w0 @ x + 1 = 0
    w0 = w0_[:3] / w0_[3]
    # normalize so that w[2]=1
    w0 = w0 / w0[2]

    if CM.detection.sample_hard_negative:
        # sample around second symmetry axis (hard negative)
        w1_ = LA.inv(Rt).T @ plane2
        w1 = w1_[:3] / w1_[3]
        w1 = w1 / w1[2]
        ws = [
            sample_symmetry(w0, 0, math.pi / 2),
            sample_symmetry(w1, 0, CM.detection.theta[0]),
        ]
    else:
        while True:
            w = sample_symmetry(w0, 0, math.pi / 2)
            if sum(to_label(w, w0)) == 0:
                break
        ws = [w]
    for theta1, theta0 in zip(CM.detection.theta, CM.detection.theta[1:] + [0]):
        ws.append(sample_symmetry(w0, theta0 * 1.001, theta1 * 0.999))
    return w0, ws


def sample_symmetry(w0, theta0, theta1, delta=1):
    w = sample_sphere(w0 / LA.norm(w0), theta0, theta1)
    return w / (delta * w[2])


def sample_sphere(v, theta0, theta1):
    def orth(v):
        x, y, z = v
        o = np.array([0.0, -z, y] if abs(x) < abs(y) else [-z, 0.0, x])
        o /= LA.norm(o)
        return o

    costheta = random.uniform(math.cos(theta1), math.cos(theta0))
    phi = random.random() * math.pi * 2
    v1 = orth(v)
    v2 = np.cross(v, v1)
    r = math.sqrt(1 - costheta ** 2)
    w = v * costheta + r * (v1 * math.cos(phi) + v2 * math.sin(phi))
    return w / LA.norm(w)


def w2S(w):
    S = np.eye(4)
    S[:3, :3] = np.eye(3) - 2 * np.outer(w, w) / np.sum(w ** 2)
    S[:3, 3] = -2 * w / np.sum(w ** 2)
    return S


def to_label(w, w0):
    theta = math.acos(np.clip(abs(w @ w0) / LA.norm(w) / LA.norm(w0), -1, 1))
    return [theta < theta0 for theta0 in CM.detection.theta]
