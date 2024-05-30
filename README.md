# ADAPT
When Adversarial Training Meets Prompt Tuning: Adversarial Dual Prompt Tuning for Unsupervised Domain Adaptation 

## Overview
This repository is a PyTorch implementation of the paper.  

## Framework
![Framework](https://github.com/Liuziyi1999/ADAPT/blob/main/assets/framework.png)

## How to Install
Our code is built based on the source code of [CoOp](https://github.com/KaiyangZhou/CoOp). So you need to install some dependent environments.
```# install clip
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git

# clone dapl
git clone https://github.com/LeapLabTHU/DAPrompt.git

# install dassl
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd dassl
pip install -r requirements.txt
pip install .
cd ..
```
You may follow the installation guide from [CLIP](https://github.com/KaiyangZhou/CoOp) and [dassl](https://github.com/KaiyangZhou/Dassl.pytorch).

## Dataset
- Manually download the datasets [Office](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view), [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view), from the official websites.
- [VisDA](https://ai.bu.edu/visda-2017/) is a dataset from VisDA 2017 challenge. It contains two domains, i.e., 152397 synthetic images and 55388 real images.
- [Mini-DomainNet](https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/DATASETS.md#miniDomainNet) is a subset of [DomainNet](http://ai.bu.edu/M3SDA/), which is known as the most challenging dataset for UDA.

## How to Run
We provide the running scripts in ```scripts/```. Make sure you change the path in ```DATA``` and run the commands under ```ADAPT/scripts/```.

### Training
The commond is in the file ADAPT/scripts/main.sh, which contains six input arguments:

- ```DATASET``` takes as input a dataset name, like ```visda```、```Office``` or ```Office-Home```. The valid names are the files' names in ```ADAPT/configs/datasets/```. The names of dataset, source domain and target domain is defined in the file. The visual backbone is also defined in the yaml. file for the common visual backbone is related to dataset. You may follow these files to establish new datasets;
- ```CFG``` means which config file to use, such as ```vit_b16``` (see ```ADAPT/configs/trainers/ADAPT/```). The implemntation details are included in this file. You may modify the hyper-parameters in the file;

Below we provide examples on how to run DAPL on VisDA-2017. The file ```ADAPT/scripts/main.sh``` defines the path to dataset in the line 6. You may set it as the true path to your dataset. If you want to train DAPL on the VisDA-2017 dataset, you may run the below command in the path ```ADAPT/scripts```:

``` 
bash main.sh visda17 vit_b16 1.0 0.5 1.0 t0
```

### Load a pre-trained Model
We have upload a pretrained weight. You can load it and evaluate in the target domain. The command is
```
bash eval.sh visda17 ep25-32-csc 1.0 0.5 1.0 t0
```

### Other information
Currently only the shallow version is available, the deep version will be updated after the paper is published.

### Acknowledgement
Thanks for the following projects:
- [CLIP](https://github.com/openai/CLIP)
- [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch)
- [CoOp](https://github.com/KaiyangZhou/CoOp)
- [DAPL](https://github.com/LeapLabTHU/DAPrompt)
- [Maple](https://github.com/muzairkhattak/multimodal-prompt-learning)



