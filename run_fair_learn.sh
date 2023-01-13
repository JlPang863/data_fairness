# # --------- baseline ------------
# CUDA_VISIBLE_DEVICES=0 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf no_conf  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 1 --sel_round $SR > ./logs/fair_train/s1_dp_02_new256_100round_sel_$SR\_case1_random_sel_no_conf.log &

sel_round="5 10 15 20 25"

for SR in $sel_round
do
    
    CUDA_VISIBLE_DEVICES=0 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf entropy  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 2 --sel_round $SR Org --conf_fair_only --train_conf > ./logs/fair_train/s2_dp_02_new256_100round_sel_$SR\_case1_remove_unfairOrg_train_conf_ConfFariOnly.log &

    # CUDA_VISIBLE_DEVICES=0 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf entropy  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 3 --sel_round $SR Org --conf_fair_only --train_conf > ./logs/fair_train/s3_dp_02_new256_100round_sel_$SR\_case1_remove_unfairOrg_train_conf_ConfFariOnly.log &



    CUDA_VISIBLE_DEVICES=0 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf entropy  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 4 --sel_round $SR Org --conf_fair_only --train_conf > ./logs/fair_train/s4_dp_02_new256_100round_sel_$SR\_case1_remove_unfairOrg_train_conf_ConfFariOnly.log &

    CUDA_VISIBLE_DEVICES=1 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf entropy  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 5 --sel_round $SR Org --conf_fair_only --train_conf > ./logs/fair_train/s5_dp_02_new256_100round_sel_$SR\_case1_remove_unfairOrg_train_conf_ConfFariOnly.log &




    CUDA_VISIBLE_DEVICES=1 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf entropy  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 2 --sel_round $SR --remove_posOrg --conf_fair_only --train_conf > ./logs/fair_train/s2_dp_02_new256_100round_sel_$SR\_case1_remove_unfair_poslossOrg_train_conf_ConfFariOnly.log &

    # CUDA_VISIBLE_DEVICES=2 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf entropy  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 3 --sel_round $SR --remove_posOrg --conf_fair_only --train_conf > ./logs/fair_train/s3_dp_02_new256_100round_sel_$SR\_case1_remove_unfair_poslossOrg_train_conf_ConfFariOnly.log &



    CUDA_VISIBLE_DEVICES=2 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf entropy  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 4 --sel_round $SR --remove_posOrg --conf_fair_only --train_conf > ./logs/fair_train/s4_dp_02_new256_100round_sel_$SR\_case1_remove_unfair_poslossOrg_train_conf_ConfFariOnly.log &

    CUDA_VISIBLE_DEVICES=2 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf entropy  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 5 --sel_round $SR --remove_posOrg --conf_fair_only --train_conf > ./logs/fair_train/s5_dp_02_new256_100round_sel_$SR\_case1_remove_unfair_poslossOrg_train_conf_ConfFariOnly.log &

    # -------------

    CUDA_VISIBLE_DEVICES=3 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf entropy  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 2 --sel_round $SR  --conf_fair_only --train_conf > ./logs/fair_train/s2_dp_02_new256_100round_sel_$SR\_case1_remove_unfair_train_conf_ConfFariOnly.log &

    # CUDA_VISIBLE_DEVICES=4 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf entropy  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 3 --sel_round $SR  --conf_fair_only --train_conf > ./logs/fair_train/s3_dp_02_new256_100round_sel_$SR\_case1_remove_unfair_train_conf_ConfFariOnly.log &



    CUDA_VISIBLE_DEVICES=3 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf entropy  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 4 --sel_round $SR  --conf_fair_only --train_conf > ./logs/fair_train/s4_dp_02_new256_100round_sel_$SR\_case1_remove_unfair_train_conf_ConfFariOnly.log &

    CUDA_VISIBLE_DEVICES=4 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf entropy  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 5 --sel_round $SR  --conf_fair_only --train_conf > ./logs/fair_train/s5_dp_02_new256_100round_sel_$SR\_case1_remove_unfair_train_conf_ConfFariOnly.log &




    CUDA_VISIBLE_DEVICES=4 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf entropy  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 2 --sel_round $SR --remove_pos --conf_fair_only --train_conf > ./logs/fair_train/s2_dp_02_new256_100round_sel_$SR\_case1_remove_unfair_posloss_train_conf_ConfFariOnly.log &

    # CUDA_VISIBLE_DEVICES=6 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf entropy  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 3 --sel_round $SR --remove_pos --conf_fair_only --train_conf > ./logs/fair_train/s3_dp_02_new256_100round_sel_$SR\_case1_remove_unfair_posloss_train_conf_ConfFariOnly.log &



    CUDA_VISIBLE_DEVICES=5 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf entropy  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 4 --sel_round $SR --remove_pos --conf_fair_only --train_conf > ./logs/fair_train/s4_dp_02_new256_100round_sel_$SR\_case1_remove_unfair_posloss_train_conf_ConfFariOnly.log &

    CUDA_VISIBLE_DEVICES=5 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf entropy  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 5 --sel_round $SR --remove_pos --conf_fair_only --train_conf > ./logs/fair_train/s5_dp_02_new256_100round_sel_$SR\_case1_remove_unfair_posloss_train_conf_ConfFariOnly.log &

    sleep 45m

    


done

# SR="25"

#     CUDA_VISIBLE_DEVICES=0 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf entropy  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 2 --sel_round $SR  > ./logs/fair_train/s2_dp_02_new256_100round_sel_$SR\_case1_remove_unfair.log &

#     CUDA_VISIBLE_DEVICES=0 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf entropy  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 3 --sel_round $SR  > ./logs/fair_train/s3_dp_02_new256_100round_sel_$SR\_case1_remove_unfair.log &


#     CUDA_VISIBLE_DEVICES=1 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf entropy  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 4 --sel_round $SR  > ./logs/fair_train/s4_dp_02_new256_100round_sel_$SR\_case1_remove_unfair.log &

#     CUDA_VISIBLE_DEVICES=1 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf entropy  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 5 --sel_round $SR  > ./logs/fair_train/s5_dp_02_new256_100round_sel_$SR\_case1_remove_unfair.log &



#     CUDA_VISIBLE_DEVICES=2 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf entropy  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 2 --sel_round $SR --remove_pos > ./logs/fair_train/s2_dp_02_new256_100round_sel_$SR\_case1_remove_unfair_posloss.log &

#     CUDA_VISIBLE_DEVICES=2 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf entropy  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 3 --sel_round $SR --remove_pos > ./logs/fair_train/s3_dp_02_new256_100round_sel_$SR\_case1_remove_unfair_posloss.log &


#     CUDA_VISIBLE_DEVICES=3 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf entropy  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 4 --sel_round $SR --remove_pos > ./logs/fair_train/s4_dp_02_new256_100round_sel_$SR\_case1_remove_unfair_posloss.log &

#     CUDA_VISIBLE_DEVICES=3 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf entropy  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 5 --sel_round $SR --remove_pos > ./logs/fair_train/s5_dp_02_new256_100round_sel_$SR\_case1_remove_unfair_posloss.log &