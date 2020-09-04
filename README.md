# SymmetryNet: Learning to Detect 3D Reflection Symmetry for Single-View Reconstruction

This repository contains the official PyTorch implementation of the paper:  *[Yichao Zhou](https://yichaozhou.com), [Shichen Liu](https://shichenliu.github.io/), [Yi Ma](https://people.eecs.berkeley.edu/~yima/). "[Learning to Detect 3D Reflection Symmetry for Single-View Reconstruction](https://arxiv.org/abs/2006.10042)"*.

## Introduction

SymmetryNet is a *geometry-based* end-to-end deep learning framework that detects the plane of reflection symmetry and uses it to help the prediction of depth maps by finding the intra-image pixel-wise correspondence.

## Main Results

### Qualitative Measures

Coming soon.

### Quantitative Measures

Coming soon.

## Code Structure

Below is a quick overview of the function of key files.

```bash
########################### Data ###########################
data/
    shapenet-r2n2/              # default folder for R2N2 data
logs/                           # default folder for storing the output during training
########################### Code ###########################
config/                         # neural network hyper-parameters and configurations
    shapenet-aio.yaml           # default config for symmetry detection & depth estimation
misc/                           # misc scripts that are not important
    find-radius.py              # script for generating figure grids
sym/                            # sym module so you can "import sym" in other scripts
    models/                     # neural network architectures
        symmetry_net.py         # wrapper for loss
        mvsnet.py               # 3D hourglass
    config.py                   # global variables for configuration
    datasets.py                 # reading the training data
    trainer.py                  # general trainer
train.py                        # script for training the neural network
eval.py                         # script for evaluating a dataset from a checkpoint
plot-angle.py                   # script for ploting angle error curves
plot-depth.py                   # script for ploting depth error curves
```

## Reproducing Results

### Installation

For the ease of reproducibility, you are suggested to install [miniconda](https://docs.conda.io/en/latest/miniconda.html) (or [anaconda](https://www.anaconda.com/distribution/) if you prefer) before following executing the following commands.

```bash
git clone https://github.com/zhou13/symmetrynet
cd symmetrynet
conda create -y -n symmetrynet
source activate symmetrynet
conda install -y pyyaml docopt matplotlib scikit-image opencv tqdm
# Replace cudatoolkit=10.2 with your CUDA version: https://pytorch.org/get-started/
conda install -y pytorch cudatoolkit=10.2 -c pytorch
mkdir data logs results
```

### Downloading the Processed Datasets
Make sure `curl` is installed on your system and execute
```bash
cd data
../misc/gdrive-download.sh 1zdHsSb-xHhY8imIy3uWt_XfBW_YQX4Vh shapenet-r2n2.zip
unzip *.zip
rm *.zip
cd ..
```

If `gdrive-download.sh` does not work for you, you can download the pre-processed datasets
manually from our [Google Drive](https://drive.google.com/file/d/1zdHsSb-xHhY8imIy3uWt_XfBW_YQX4Vh) and proceed accordingly.

### Training
Execute the following commands to train the neural networks from scratch with four GPUs (specified by `-d 0,1,2,3`):
```bash
python ./train.py -d 0,1,2,3 --identifier baseline config/shapenet-aio.yaml
```

The checkpoints and logs will be written to `logs/` accordingly.

### Pre-trained Models

You can download our reference pre-trained models from [Google Drive](https://drive.google.com/file/d/1U1zN_LcvgpoV9yhMRQtxYo_QzBJicp9e).  This pretrained model has slightly better performance than the ones reported in the paper, because it is trained with more epochs.

### Evaluation

To evaluate the models with coarse-to-fine inference for symmetry plane prediction and depth map estimation, execute

``` bash
python eval.py -d 0 --output results/symmetrynet.npz logs/<your-checkpoint>/config.yaml logs/<your-checkpoint>/checkpoint_latest.pth.tar
```

The error statistics are printed on the screen and the error metrics are stored in `results/symmetrynet.npz`. To visualize the error distribution and plot the error-percentage curves, execute

``` bash
python plot-angle.py
python plot-depth.py
```


## Acknowledgement

This work is supported by a research grant from Sony Research.

## Citing SymmetryNet

If you find SymmetryNet useful in your research, please consider citing:

```
@article{zhou2020learning,
    author = {Zhou, Yichao and Liu, Shichen and Ma, Yi},
    title = {Learning to Detect 3D Reflection Symmetry for Single-View Reconstruction},
    year = {2020},
    archivePrefix = "arXiv", 
    note = {arXiv:2006.10042 [cs.CV]},
}
```
