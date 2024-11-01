# Fairness Without Harm: An Influence-Guided Active Sampling Approach

This code is a PyTorch implementation of our paper "[Fairness Without Harm: An Influence-Guided Active Sampling Approach]" accepted by NeurIPS 2024. - [Jinlong Pang](https://jlpang863.github.io/), [Jialu Wang](https://people.ucsc.edu/~jwang470/), [Zhaowei Zhu](https://users.soe.ucsc.edu/~zhaoweizhu/), [Yuanshun Yao](https://www.kevyao.com/), [Chen Qian](https://users.soe.ucsc.edu/~qian/), [Yang Liu](http://www.yliuu.com/).

## Prerequisites
You can install the replicable Python environment by using
```
pip install -r requirements.txt
```

## Guideline

Here is the main code for executing our methods on different datasets:

### CelebA Dataset: 
  Use `bash run_celeba.sh` to directly train a new model on the CelebA Dataset. 

  Here is an example with detailed setting:

  ```
  $ CUDA_VISIBLE_DEVICES=0 python3 run_celeba.py --runs 0 --warm_epoch 0 --epoch 10 --metric dp --label_ratio 0.05 --val_ratio 0.1 --strategy 2
  ```

### Adult Dataset: 
  Use `bash run_adult.sh` to directly train a new model on the Adult Dataset. 

  Here is an example with detailed setting:

  ```
  $ CUDA_VISIBLE_DEVICES=0 python3 run_adult.py --group_key age  --warm_epoch 0  --metric dp --label_ratio 0.2 --val_ratio 0.1 --strategy 2 
  ```

### Compas Dataset:
  Use `bash run_compas.sh` to directly train a new model on the Compas Dataset. 

  Here is an example with detailed setting:

  ```
  $ CUDA_VISIBLE_DEVICES=0 python3 run_compas.py --runs 0 --epoch 50 --metric dp --label_ratio 0.2  --val_ratio 0.2 --strategy 2 --warm_epoch 50
  ```



