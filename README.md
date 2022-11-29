# This is the working repo for fair active sampling



11/22/2022 update for rebuttal

- Add learning-based T estimator
- Fairness evaluation with learning-based T estimator
  - Check code: `train_cl(args)` in `./fair_learn/run_celeba.py` is uncommented. 
  - `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 python3 run_celeba.py --method plain  --lmd 0.0 --mu 1.0  --warm_epoch 0 --conf no_conf`
  - This part is hard-coded. It only supports the experiment with learning-based T. For fairness evaluation with our methods, please go to the stable version (fair-eval v1). 
- Disparity mitigation with our algorithm
  - Check code: `train(args)` in `./fair_learn/run_celeba.py` is uncommented. 
  - `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 run --cpu 8 --type v100-32g -- python3 run_celeba.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_epoch 0 --conf entropy  --metric dp --fe_sel 3` (MLX lab)
  - `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 python3 run_celeba.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_epoch 0 --conf entropy  --metric dp --fe_sel 3` (plain)
  - Recommend to check a cleaner version at `fair_learn (active-sampling)`