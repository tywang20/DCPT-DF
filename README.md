# Deep Convolutional Pooling Transformer for Deepfake Detection (DCPT)
Source code of the paper [Deep Convolutional Pooling Transformer for Deepfake Detection](https://arxiv.org/pdf/2209.05299) accepted to ACM Transactions on Multimedia Computing, Communications and Applications (TOMM).

## To test the model
```
python test.py
```

## To train the model
```
python train.py
```

## Important files
Main model architecture at [model/network.py](model/network.py).

Directory architecture for the datasets can be seen in [dataset/](dataset) folder.


## Dependancies
Our implementation is tested on the following libraries with Python 3.8.5, torch1.7.1, and CUDA 11.0.

Install other dependencies using the following command.

```
pip install albumentations
pip install -U albumentations[imgaug]
pip install einops
pip install timm
pip install kornia
```

## Citation
If you find our work useful, please properly cite the following:
```
@article{DCPT2023Wang,
author = {Wang, Tianyi and Cheng, Harry and Chow, Kam Pui and Nie, Liqiang},
title = {Deep Convolutional Pooling Transformer for Deepfake Detection},
year = {2023},
volume = {19},
number = {6},
issn = {1551-6857},
doi = {10.1145/3588574},
journal = {ACM Trans. Multimedia Comput. Commun. Appl.}
}
```