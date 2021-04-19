#!/usr/bin/env python3
import json
import os
from collections import defaultdict
from glob import glob

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({"font.size": 16})
plt.rcParams["font.family"] = "Times New Roman"
del mpl.font_manager.weight_dict["roman"]
mpl.font_manager._rebuild()

error_types = [
    ("err_depth_avg", "Depth Error (Average L1)"),
]
xlims = [0.08, 0.01]

for ii, ((err_key, err_name), xlim) in enumerate(zip(error_types, xlims)):
    plt.figure()
    plt.title(err_name , fontsize=20)
    for fname in sorted(glob(f"results/*.npz")):
        with np.load(fname) as f:
            if err_key not in f:
                print(err_key, fname)
                continue
            err = f[err_key]
        _, label = os.path.split(fname)

        plt.plot(err, (1 + np.arange(len(err))) / len(err), label=label[1:-4])
        plt.grid(True)
        plt.xlim([0, xlim])
        plt.ylabel("Percentage")
        plt.legend(loc="lower right")

plt.show()
