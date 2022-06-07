
# yolov5-obb

## Installation

## Requirements
* Linux **(Recommend)**, Windows **(not Recommend, Please refer to this [issue](https://github.com/hukaixuan19970627/yolov5_obb/issues/224) if you have difficulty in generating utils/nms_rotated_ext.cpython-XX-XX-XX-XX.so)**
* Python 3.7+ 
* PyTorch ≥ 1.7 
* CUDA 9.0 or higher

I have tested the following versions of OS and softwares：
* OS：Ubuntu 16.04/18.04, Windows 10
* CUDA: 10.0/10.1/10.2/11.3

## Fixing Error
1. If you are using window 10, you would get installation error sometime. (Please Make Sure to follow these steps):
2. You need pytorch of version <1.10; version 1.11 does not have TCH;
3. Make sure you have only one cuda veriosion installed or mentioned in your PATH;
4. for some reason, cl does not recognize const double eps=1E-8; on line 24 of poly_nms_cuda.cu throwing an error. As a hack, I have replaced eps with the value 1E-8    at the only two places it is used (the sig function right below), and the whole thing worked for me.
## Install 
**CUDA Driver Version ≥ CUDA Toolkit Version(runtime version) = torch.version.cuda**

a. Create a conda virtual environment and activate it, e.g.,
```
conda create -n Py39_Torch1.10_cu11.3 python=3.9 -y 
source activate Py39_Torch1.10_cu11.3
```
b. Make sure your CUDA runtime api version ≤ CUDA driver version. (for example 11.3 ≤ 11.4)
```
nvcc -V
nvidia-smi
```
c. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), Make sure cudatoolkit version same as CUDA runtime api version, e.g.,
```
pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
nvcc -V
python
>>> import torch
>>> torch.version.cuda
>>> exit()
```
d. Clone the yolov5-obb repository.
```
git clone https://github.com/hukaixuan19970627/yolov5_obb.git
cd yolov5_obb
```
e. Install yolov5-obb.

```python 
pip install -r requirements.txt
cd utils/nms_rotated
python setup.py develop  #or "pip install -v -e ."
```

## Install DOTA_devkit. 
**(Custom Install, it's just a tool to split the high resolution image and evaluation the obb)**

```
cd yolov5_obb/DOTA_devkit
sudo apt-get install swig
swig -c++ -python polyiou.i
python setup.py build_ext --inplace
```

## Training

# Train a model

**1. Prepare custom dataset files**

1.1 Make sure the labels format is [poly classname diffcult], e.g., You can set **diffcult=0**
```
  x1      y1       x2        y2       x3       y3       x4       y4       classname     diffcult

1686.0   1517.0   1695.0   1511.0   1711.0   1535.0   1700.0   1541.0   large-vehicle      1
```
![image](https://user-images.githubusercontent.com/72599120/159213229-b7c2fc5c-b140-4f10-9af8-2cbc405b0cd3.png)


1.2 Split the dataset. 
```shell
cd yolov5_obb
python DOTA_devkit/ImgSplit_multi_process.py
```
or Use the orignal dataset. 
```shell
cd yolov5_obb
```

1.3 Make sure your dataset structure same as:
```
parent
├── yolov5
└── datasets
    └── DOTAv1.5
        ├── train_split_rate1.0_subsize1024_gap200
        ├── train_split_rate1.0_subsize1024_gap200
        └── test_split_rate1.0_subsize1024_gap200
            ├── images
                 |────1.jpg
                 |────...
                 └────10000.jpg
            ├── labelTxt
                 |────1.txt
                 |────...
                 └────10000.txt

```

**Note:**
* DOTA is a high resolution image dataset, so it needs to be splited before training/testing to get better performance.

**2. Train**

2.1 Train with specified GPUs. (for example with GPU=3)

```shell
python train.py --device 3
```

2.2 Train with multiple(4) GPUs. (DDP Mode)

```shell
python -m torch.distributed.launch --nproc_per_node 4 train.py --device 0,1,2,3
```

2.3 Train the orignal dataset demo.
```shell
python train.py --data 'data/yolov5obb_demo.yaml' --epochs 10 --batch-size 1 --img 1024 --device 0
```

2.4 Train the splited dataset demo.
```shell
python train.py --data 'data/yolov5obb_demo_split.yaml' --epochs 10 --batch-size 2 --img 1024 --device 0
```

# Inferenece with pretrained models. (Splited Dataset)
This repo provides the validation/testing scripts to evaluate the trained model.

