# CUDA_VISIBLE_DEVICES=0 nohup python3 run_celeba.py --method plain  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf entropy  --metric dp --label_ratio 0.05 --val_ratio 0.1 --strategy 5 --remove_pos --train_conf > s5_dp_05_new256_100round_case1_remove_unfair_posloss_train_conf.log &


CUDA_VISIBLE_DEVICES=0 python3 run_celeba.py --method plain  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf no_conf  --metric dp --label_ratio 0.05 --val_ratio 0.1 --strategy 2 --remove_pos --remove_pos