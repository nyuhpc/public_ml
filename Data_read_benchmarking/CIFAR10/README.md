Based on https://realpython.com/storing-images-in-python/

get data set from https://www.cs.toronto.edu/~kriz/cifar.html

mkdir -p /scratch/$USER/datasets/CIFAR10
cd /scratch/$USER/datasets/CIFAR10
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar xfvz cifar-10-python.tar.gz

Install packages

module load anaconda3/5.3.1
conda create -p $(pwd)/penv python=3.7
conda activate ...
conda install -y pillow lmdb python-lmdb hdf5 h5py matplotlib numpy pathlib

