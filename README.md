# Deep Convolutional Pooling Transformer for Deepfake Detection (DCPT)
Source code of the paper "Deep Convolutional Pooling Transformer for Deepfake Detection".

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
