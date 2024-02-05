# This is the working repo for data fairness project


## Training

Here is the main code for executing our methods on different datasets:

### CelebA Dataset: 
  Use `bash run_celeba.sh` to directly train a new model on the CelebA Dataset. 

  Here is an example with detailed setting:
  ```
  $ CUDA_VISIBLE_DEVICES=0 python3 run_celeba.py --runs 0 --warm_epoch 0 --epoch 10 --metric dp --label_ratio 0.05 --val_ratio 0.1 --strategy 2```

### Adult Dataset: 
  Use `bash run_adult.sh` to directly train a new model on the Adult Dataset. 

  Here is an example with detailed setting:
  ```
  $ CUDA_VISIBLE_DEVICES=0 python3 run_adult.py --group_key age  --warm_epoch 0  --metric dp --label_ratio 0.2 --val_ratio 0.1 --strategy 2 ```

### Compas Dataset:
  Use `bash run_compas.sh` to directly train a new model on the Compas Dataset. 

  Here is an example with detailed setting:

  ```
  $ CUDA_VISIBLE_DEVICES=0 python3 run_compas.py --runs 0 --epoch 50 --metric dp --label_ratio 0.2  --val_ratio 0.2 --strategy 2 --warm_epoch 50
  ```

## Update
02/04/2024 update for ICML subsmission 

- Add arguments for all datasets
- Disparity mitigation with our algorithm
  - Check code: `train(args)` in `./fair_learn/run_celeba.py` is uncommented. 
  - `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 run --cpu 8 --type v100-32g -- python3 run_celeba.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_epoch 0 --conf entropy  --metric dp --fe_sel 3` (MLX lab)
  - `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 python3 run_celeba.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_epoch 0 --conf entropy  --metric dp --fe_sel 3` (plain)
  - Recommend to check a cleaner version at `fair_learn (active-sampling)`
