# MFF_OVMLC
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

## Training and Evaluation

### NUS-WIDE

### Step 1: Data preparation

1) Download pre-computed features from [here](https://drive.google.com/drive/folders/1jvJ0FnO_bs3HJeYrEJu7IcuilgBipasA?usp=sharing) and store them at `features` folder inside `BiAM/datasets/NUS-WIDE` directory.
2) [Optional] You can extract the features on your own by using the original NUS-WIDE dataset from [here](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html) and run the below script:

```
python feature_extraction/extract_nus_wide.py

```

### Step 2: Training from scratch

To train and evaluate multi-label zero-shot learning model on full NUS-WIDE dataset, please run:

```
sh scripts/train_nus.sh
```

### Step 3: Evaluation using pretrained weights

To evaluate the multi-label zero-shot model on NUS-WIDE. You can download the pretrained weights from [here](https://drive.google.com/drive/folders/1o03bqr_yNPblwAPjv2J83tMsHEDiEKPk?usp=sharing) and store them at `NUS-WIDE` folder inside `pretrained_weights` directory.

```
sh scripts/evaluate_nus.sh
```

### OPEN-IMAGES

### Step 1: Data preparation

1) Please download the annotations for [training](https://storage.googleapis.com/openimages/2018_04/train/train-annotations-human-imagelabels.csv), [validation]( https://storage.googleapis.com/openimages/2018_04/validation/validation-annotations-human-imagelabels.csv), and [testing](https://storage.googleapis.com/openimages/2018_04/test/test-annotations-human-imagelabels.csv) into this folder.

2) Store the annotations inside `BiAM/datasets/OpenImages`.

3) To extract the features for OpenImages-v4 dataset run the below scripts for crawling the images and extracting features of them:

```
## Crawl the images from web
python ./datasets/OpenImages/download_imgs.py  #`data_set` == `train`: download images into `./image_data/train/`
python ./datasets/OpenImages/download_imgs.py  #`data_set` == `validation`: download images into `./image_data/validation/`
python ./datasets/OpenImages/download_imgs.py  #`data_set` == `test`: download images into `./image_data/test/`

## Run feature extraction codes for all the 3 splits
python feature_extraction/extract_openimages_train.py
python feature_extraction/extract_openimages_test.py
python feature_extraction/extract_openimages_val.py

```

### Step 2: Training from scratch

To train and evaluate multi-label zero-shot learning model on full OpenImages-v4 dataset, please run:

```
sh scripts/train_openimages.sh
sh scripts/evaluate_openimages.sh

```

### Step 3: Evaluation using pretrained weights

To evaluate the multi-label zero-shot model on OpenImages. 

