# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This Script Assumes Python 3.10, CUDA 12.1

conda deactivate

# Set environment variables
export ENV_NAME=BEVDiffLoc
export PYTHON_VERSION=3.10
export PYTORCH_VERSION=2.1.2
export CUDA_VERSION=12.1

# Create a new conda environment and activate it
conda create -n $ENV_NAME python=$PYTHON_VERSION
conda activate $ENV_NAME

# Install PyTorch, torchvision, and PyTorch3D using conda
conda install pytorch=$PYTORCH_VERSION torchvision pytorch-cuda=$CUDA_VERSION -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d

# Install pip packages
pip install hydra-core --upgrade
pip install omegaconf opencv-python einops visdom
pip install accelerate==0.24.0
pip install matplotlib==3.8.2
pip install pandas==2.2.0
pip install transforms3d==0.4.1
pip install open3d==0.18.0
pip install h5py==3.10.0
pip install tensorboardX==2.6.2.2
pip install timm==0.9.12
pip install faiss-gpu
pip install numpy
pip install opencv-python
pip install scikit-image
pip install scikit-learn
pip install tqdm
pip install argparse
pip install imgaug
