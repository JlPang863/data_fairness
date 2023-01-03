
# # label_key="Arched_Eyebrows Attractive Bags_Under_Eyes Bald"
# label_key="Smiling"
# # conf="entropy no_conf peer"
# conf="no_conf entropy"
# # metric="dp dp_cov plain"
# metric="dp"
# # metric="dp dp_cov plain eop eop_cov eod eod_cov"
# # warm="0 1"
# warm="0"
# fe_sel="0 1 2 3 4 5 6"

# for MYLABEL in $label_key
# do
# for MYCONF in $conf
# do
# for MYWARM in $warm
# do
        
#     for MYMETRIC in $metric
#     do
#         for MYSEL in $fe_sel
#         do
#             LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 nohup run --cpu 8 --type v100-32g -- python3 run_celeba.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_epoch $MYWARM --conf $MYCONF --metric $MYMETRIC --label_key $MYLABEL --fe_sel $MYSEL &
#             sleep 1m
#         done
#     done    
#     sleep 1h
# done
# done
# done
# wait



# CUDA_VISIBLE_DEVICES=0 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 nohup python3 run_celeba.py --method plain  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf no_conf  --metric dp --label_ratio 0.001 --val_ratio 0.1 --strategy 1 > s1.log &

# CUDA_VISIBLE_DEVICES=1 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 nohup python3 run_celeba.py --method plain  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf no_conf  --metric dp --label_ratio 0.001 --val_ratio 0.1 --strategy 2 > s2.log &

# CUDA_VISIBLE_DEVICES=2 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 nohup python3 run_celeba.py --method plain  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf no_conf  --metric dp --label_ratio 0.001 --val_ratio 0.1 --strategy 3 > s3.log &

# CUDA_VISIBLE_DEVICES=3 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 nohup python3 run_celeba.py --method plain  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf no_conf  --metric dp --label_ratio 0.001 --val_ratio 0.1 --strategy 4 > s4.log &

# CUDA_VISIBLE_DEVICES=4 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 nohup python3 run_celeba.py --method plain  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf no_conf  --metric dp --label_ratio 0.001 --val_ratio 0.1 --strategy 5 > s5.log &


# CUDA_VISIBLE_DEVICES=0 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 nohup python3 run_celeba.py --method plain  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf no_conf  --metric dp --label_ratio 0.05 --val_ratio 0.1 --strategy 1 > s1_dp_05_new1024_20round.log &

# CUDA_VISIBLE_DEVICES=1 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 nohup python3 run_celeba.py --method plain  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf no_conf  --metric dp --label_ratio 0.05 --val_ratio 0.1 --strategy 2 > s2_dp_05_new1024_20round.log &

# CUDA_VISIBLE_DEVICES=2 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 nohup python3 run_celeba.py --method plain  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf no_conf  --metric dp --label_ratio 0.05 --val_ratio 0.1 --strategy 3 > s3_dp_05_new1024_20round.log &

# CUDA_VISIBLE_DEVICES=3 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 nohup python3 run_celeba.py --method plain  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf no_conf  --metric dp --label_ratio 0.05 --val_ratio 0.1 --strategy 4 > s4_dp_05_new1024_20round.log &

# CUDA_VISIBLE_DEVICES=4 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 nohup python3 run_celeba.py --method plain  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf no_conf  --metric dp --label_ratio 0.05 --val_ratio 0.1 --strategy 5 > s5_dp_05_new1024_20round.log &

# CUDA_VISIBLE_DEVICES=0 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 nohup python3 run_celeba.py --method plain  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf no_conf  --metric dp --label_ratio 0.05 --val_ratio 0.1 --strategy 1 > s1_dp_05_new1024_100round_case1_new.log &

# # -----------------------
# CUDA_VISIBLE_DEVICES=1 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 nohup python3 run_celeba.py --method plain  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf no_conf  --metric dp --label_ratio 0.05 --val_ratio 0.1 --strategy 2 > s2_dp_05_new1024_100round_case1_remove_unfair_posloss.log &

# CUDA_VISIBLE_DEVICES=3 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 nohup python3 run_celeba.py --method plain  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf no_conf  --metric dp --label_ratio 0.05 --val_ratio 0.1 --strategy 3 > s3_dp_05_new1024_100round_case1_remove_unfair_posloss.log &

# CUDA_VISIBLE_DEVICES=4 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 nohup python3 run_celeba.py --method plain  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf no_conf  --metric dp --label_ratio 0.05 --val_ratio 0.1 --strategy 4 > s4_dp_05_new1024_100round_case1_remove_unfair_posloss.log &

# CUDA_VISIBLE_DEVICES=6 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 nohup python3 run_celeba.py --method plain  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf no_conf  --metric dp --label_ratio 0.05 --val_ratio 0.1 --strategy 5 > s5_dp_05_new1024_100round_case1_remove_unfair_posloss.log &
# # -----------------------


# -----------------------
CUDA_VISIBLE_DEVICES=0 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 nohup python3 run_celeba.py --method plain  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf entropy  --metric dp --label_ratio 0.05 --val_ratio 0.1 --strategy 2 > s2_dp_05_new1024_100round_case1_remove_unfair_posloss_entropy.log &

CUDA_VISIBLE_DEVICES=3 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 nohup python3 run_celeba.py --method plain  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf entropy  --metric dp --label_ratio 0.05 --val_ratio 0.1 --strategy 3 > s3_dp_05_new1024_100round_case1_remove_unfair_posloss_entropy.log &

CUDA_VISIBLE_DEVICES=4 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 nohup python3 run_celeba.py --method plain  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf entropy  --metric dp --label_ratio 0.05 --val_ratio 0.1 --strategy 4 > s4_dp_05_new1024_100round_case1_remove_unfair_posloss_entropy.log &

CUDA_VISIBLE_DEVICES=2 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 nohup python3 run_celeba.py --method plain  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf entropy  --metric dp --label_ratio 0.05 --val_ratio 0.1 --strategy 5 > s5_dp_05_new1024_100round_case1_remove_unfair_posloss_entropy.log &
# -----------------------



# CUDA_VISIBLE_DEVICES=0 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 nohup python3 run_celeba.py --method plain  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf no_conf  --metric dp --label_ratio 0.001 --val_ratio 0.1 --strategy 1 --tol 0.5 > s1_dp_tol_05.log &

# CUDA_VISIBLE_DEVICES=5 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 nohup python3 run_celeba.py --method plain  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf no_conf  --metric dp --label_ratio 0.001 --val_ratio 0.1 --strategy 2 --tol 0.2 > s2_dp_tol_02.log &

# CUDA_VISIBLE_DEVICES=6 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 nohup python3 run_celeba.py --method plain  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf no_conf  --metric dp --label_ratio 0.001 --val_ratio 0.1 --strategy 3 --tol 0.2 > s3_dp_tol_02.log &

# CUDA_VISIBLE_DEVICES=7 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 nohup python3 run_celeba.py --method plain  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf no_conf  --metric dp --label_ratio 0.001 --val_ratio 0.1 --strategy 4 --tol 0.2 > s4_dp_tol_02.log &

# CUDA_VISIBLE_DEVICES=4 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 nohup python3 run_celeba.py --method plain  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf no_conf  --metric dp --label_ratio 0.001 --val_ratio 0.1 --strategy 5 --tol 0.5 > s5_dp_tol_05.log &


# CUDA_VISIBLE_DEVICES=5 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 nohup python3 run_celeba.py --method plain  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf no_conf  --metric dp --label_ratio 0.001 --val_ratio 0.1 --strategy 1 > s1_dp_minabsinfl.log &

# CUDA_VISIBLE_DEVICES=6 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 nohup python3 run_celeba.py --method plain  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf no_conf  --metric dp --label_ratio 0.001 --val_ratio 0.1 --strategy 2 > s2_dp_minabsinfl.log &

# CUDA_VISIBLE_DEVICES=7 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 nohup python3 run_celeba.py --method plain  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf no_conf  --metric dp --label_ratio 0.001 --val_ratio 0.1 --strategy 3 > s3_dp_minabsinfl.log &