# Instance-Wise Adaptive Tuning and Caching for Vision-Language Models
Official implementation of ['Instance-Wise Adaptive Tuning and Caching for
Vision-Language Models'].

<div align="center">
  <img width=900 src="ATC_module.png"/>
</div>

## Requirements
### Installation
Create a conda environment and install dependencies:
```bash
conda create -n ATC python=3.9
conda activate ATC

pip install -r requirements.txt

# Install the according versions of torch and torchvision
conda install pytorch torchvision cudatoolkit
```

### Dataset
Follow [DATASET.md](https://github.com/gaopengcuhk/Tip-Adapter/blob/main/DATASET.md) to install ImageNet and other 10 datasets referring to CoOp.

## Get Started
### Configs
The running configurations can be modified in `configs/dataset.yaml`, including shot numbers, visual encoders, and hyperparamters. 

For simplicity, we provide the hyperparamters achieving the overall best performance on 1\~16 shots for a dataset, which accord with the scores reported in the paper. If respectively tuned for different shot numbers, the 1\~16-shot performance can be further improved. You can edit the `search_scale`, `search_step`, `init_beta` and `init_alpha` for fine-grained tuning.

Note that the default `load_cache` and `load_pre_feat` are `False` for the first running, which will store the cache model and val/test features in `configs/dataset/`. For later running, they can be set as `True` for faster hyperparamters tuning.

### Running
For ImageNet dataset:
```bash
CUDA_VISIBLE_DEVICES=0 python main_imagenet.py --config configs/imagenet.yaml
```
For other 10 datasets:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/dataset.yaml
```

