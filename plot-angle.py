#!/usr/bin/env python3
import json
import os
from collections import defaultdict
from glob import glob

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({"font.size": 16})
# plt.rcParams["figure.figsize"] = (8, 6)
plt.rcParams["font.family"] = "Times New Roman"
del mpl.font_manager.weight_dict["roman"]
mpl.font_manager._rebuild()

for err_key in ["err_normal"]:
    plt.figure()
    plt.title("Normal Error of Symmetry Planes")
    for npz in sorted(glob(f"results/*.npz")):
        with np.load(npz) as f:
            err = f[err_key]
        _, label = os.path.split(npz)

        print(npz, len(err))
        plt.plot(err, (1 + np.arange(len(err))) / len(err), label=label[1:-4])
        plt.grid(True)
        plt.xlim([0, 5])
        plt.ylabel("Percentage")
        plt.legend(loc="lower right")

plt.show()
