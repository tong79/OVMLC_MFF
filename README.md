## Installation
The codebase is built on PyTorch 1.1.0 and tested on Ubuntu 16.04 environment (Python3.6, CUDA9.0, cuDNN7.5).

For installing, follow these intructions
 
```
conda create -n mlzsl python=3.6
conda activate mlzsl
conda install pytorch=1.1 torchvision=0.3 cudatoolkit=9.0 -c pytorch
pip install matplotlib scikit-image scikit-learn opencv-python yacs joblib natsort h5py tqdm pandas
```
Install warmup scheduler

```
cd pytorch-gradual-warmup-lr; python setup.py install; cd ..

```




