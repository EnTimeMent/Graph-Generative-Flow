## Installation

### Requirements
* Python 3.6+
* PyTorch 1.6+
* CUDA 10.2+
* Linux

### Code Installation

Install conda from [here](https://repo.anaconda.com/miniconda/) or 
```shell
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Create a conda virtual environment
```shell
# Create a conda virtual environment.
conda create -n graphglow python=3.6 -y
conda activate graphglow
```

Conda install
```shell
# Conda install
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch -y
conda install -c conda-forge tqdm matplotlib -y
conda install -c anaconda scikit-learn scipy -y
```
pip install -U protobuf
pip install tensorflow==1.15
pip install tensorboardX
