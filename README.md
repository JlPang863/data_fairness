# Fairness Without Harm: An Influence-Guided Active Sampling Approach

This code is a PyTorch implementation of our paper "[Fairness Without Harm: An Influence-Guided Active Sampling Approach](https://arxiv.org/abs/2402.12789)" accepted by NeurIPS 2024. - [Jinlong Pang](https://jlpang863.github.io/), [Jialu Wang](https://people.ucsc.edu/~jwang470/), [Zhaowei Zhu](https://users.soe.ucsc.edu/~zhaoweizhu/), [Yuanshun Yao](https://www.kevyao.com/), [Chen Qian](https://users.soe.ucsc.edu/~qian/), [Yang Liu](http://www.yliuu.com/).

## Prerequisites
You can install the replicable Python environment by using
```
pip install -r requirements.txt
```

## Guideline

The main code for executing our methods on different datasets are provided as follows.

### CelebA dataset: 
  You can use `bash run_celeba.sh` to directly train a new model on the CelebA dataset. For a custom configuration, you can use the following example:

  ```
  python3 src/run_celeba.py --runs 0 --warm_epoch 0 --epoch 10 --metric dp --label_ratio 0.05 --val_ratio 0.1 --strategy 2
  ```

### Adult dataset: 
  You can use `bash run_adult.sh` to directly train a new model on the Adult dataset.  Alternatively, you can use the following command to specify settings:
  ```
  python3 src/run_adult.py --group_key age  --warm_epoch 0  --metric dp --label_ratio 0.2 --val_ratio 0.1 --strategy 2 
  ```

### Compas dataset:
  To train a new model on the Compas dataset, you can run `bash run_compas.sh`. Or specify settings with the following command:

  ```
  python3 src/run_compas.py --runs 0 --epoch 50 --metric dp --label_ratio 0.2  --val_ratio 0.2 --strategy 2 --warm_epoch 50
  ```

### Experimental results

You can access the experimental results by running:

```
python read_results.py
```


### Citation

If you used this repository, please cite our work:
```
@article{pang2024fair,
  title={Fair Classifiers Without Fair Training: An Influence-Guided Data Sampling Approach},
  author={Pang, Jinlong and Wang, Jialu and Zhu, Zhaowei and Yao, Yuanshun and Qian, Chen and Liu, Yang},
  journal={arXiv preprint arXiv:2402.12789},
  year={2024}
}
```