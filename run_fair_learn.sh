# CUDA_VISIBLE_DEVICES=0 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf entropy  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 2  > ./logs/fair_train/s2_dp_02_new256_100round_case1_remove_unfair.log &

# CUDA_VISIBLE_DEVICES=1 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf entropy  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 3  > ./logs/fair_train/s3_dp_02_new256_100round_case1_remove_unfair.log &

# CUDA_VISIBLE_DEVICES=0 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf entropy  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 4  > ./logs/fair_train/s4_dp_02_new256_100round_case1_remove_unfair.log &

# CUDA_VISIBLE_DEVICES=1 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf entropy  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 5  > ./logs/fair_train/s5_dp_02_new256_100round_case1_remove_unfair.log &


# CUDA_VISIBLE_DEVICES=2 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf entropy  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 2 --remove_pos > ./logs/fair_train/s2_dp_02_new256_100round_case1_remove_unfair_posloss.log &

# CUDA_VISIBLE_DEVICES=2 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf entropy  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 3 --remove_pos > ./logs/fair_train/s3_dp_02_new256_100round_case1_remove_unfair_posloss.log &

# CUDA_VISIBLE_DEVICES=3 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf entropy  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 4 --remove_pos > ./logs/fair_train/s4_dp_02_new256_100round_case1_remove_unfair_posloss.log &

# CUDA_VISIBLE_DEVICES=3 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf entropy  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 5 --remove_pos > ./logs/fair_train/s5_dp_02_new256_100round_case1_remove_unfair_posloss.log &


# --------- baseline ------------
CUDA_VISIBLE_DEVICES=4 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf entropy  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 1 > ./logs/fair_train/s1_dp_02_new256_100round_case1_random_sel.log &