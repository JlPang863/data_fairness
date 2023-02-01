# # --------- baseline ------------
# CUDA_VISIBLE_DEVICES=0 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_step 2000 --conf no_conf  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 1 --sel_round $SR > ./logs/fair_train/s1_dp_02_new256_100round_sel_$SR\_case1_random_sel_no_conf.log &

# CUDA_VISIBLE_DEVICES=0 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_step 2000 --conf entropy  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 2 --sel_round $SR --remove_posOrg --conf_fair_only --train_conf > ./logs/fair_train/s2_dp_02_new256_100round_sel_$SR\_case1_remove_unfair_poslossOrg_train_conf_ConfFariOnly.log &
    


# for SR in $sel_round
# do
    
#     CUDA_VISIBLE_DEVICES=0 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_step 2000 --conf no_conf  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 2 --sel_round $SR --remove_posOrg --train_conf > ./logs/fair_train/s2_dp_02_new256_100round_sel_$SR\_case1_remove_unfair_poslossOrg_train_conf_no_conf.log &

#     # CUDA_VISIBLE_DEVICES=0 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_step 2000 --conf no_conf  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 3 --sel_round $SR --remove_posOrg --train_conf > ./logs/fair_train/s3_dp_02_new256_100round_sel_$SR\_case1_remove_unfair_poslossOrg_train_conf_no_conf.log &



#     CUDA_VISIBLE_DEVICES=0 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_step 2000 --conf no_conf  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 4 --sel_round $SR --remove_posOrg --train_conf > ./logs/fair_train/s4_dp_02_new256_100round_sel_$SR\_case1_remove_unfair_poslossOrg_train_conf_no_conf.log &

#     CUDA_VISIBLE_DEVICES=1 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_step 2000 --conf no_conf  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 5 --sel_round $SR --remove_posOrg --train_conf > ./logs/fair_train/s5_dp_02_new256_100round_sel_$SR\_case1_remove_unfair_poslossOrg_train_conf_no_conf.log &




#     CUDA_VISIBLE_DEVICES=1 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_step 2000 --conf entropy  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 2 --sel_round $SR --remove_posOrg --train_conf > ./logs/fair_train/s2_dp_02_new256_100round_sel_$SR\_case1_remove_unfair_poslossOrg_train_conf.log &

#     # CUDA_VISIBLE_DEVICES=2 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_step 2000 --conf entropy  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 3 --sel_round $SR --remove_posOrg --train_conf > ./logs/fair_train/s3_dp_02_new256_100round_sel_$SR\_case1_remove_unfair_poslossOrg_train_conf.log &



#     CUDA_VISIBLE_DEVICES=2 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_step 2000 --conf entropy  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 4 --sel_round $SR --remove_posOrg --train_conf > ./logs/fair_train/s4_dp_02_new256_100round_sel_$SR\_case1_remove_unfair_poslossOrg_train_conf.log &

#     CUDA_VISIBLE_DEVICES=2 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_step 2000 --conf entropy  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 5 --sel_round $SR --remove_posOrg --train_conf > ./logs/fair_train/s5_dp_02_new256_100round_sel_$SR\_case1_remove_unfair_poslossOrg_train_conf.log &

#     # -------------

#     CUDA_VISIBLE_DEVICES=3 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_step 2000 --conf entropy  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 2 --sel_round $SR --remove_pos --train_conf > ./logs/fair_train/s2_dp_02_new256_100round_sel_$SR\_case1_remove_unfair_posloss_train_conf_no_conf.log &

#     # CUDA_VISIBLE_DEVICES=4 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_step 2000 --conf entropy  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 3 --sel_round $SR --remove_pos --train_conf > ./logs/fair_train/s3_dp_02_new256_100round_sel_$SR\_case1_remove_unfair_posloss_train_conf_no_conf.log &



#     CUDA_VISIBLE_DEVICES=3 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_step 2000 --conf entropy  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 4 --sel_round $SR --remove_pos --train_conf > ./logs/fair_train/s4_dp_02_new256_100round_sel_$SR\_case1_remove_unfair_posloss_train_conf_no_conf.log &

#     CUDA_VISIBLE_DEVICES=4 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_step 2000 --conf entropy  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 5 --sel_round $SR --remove_pos --train_conf > ./logs/fair_train/s5_dp_02_new256_100round_sel_$SR\_case1_remove_unfair_posloss_train_conf_no_conf.log &




#     CUDA_VISIBLE_DEVICES=4 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_step 2000 --conf entropy  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 2 --sel_round $SR --remove_pos --train_conf > ./logs/fair_train/s2_dp_02_new256_100round_sel_$SR\_case1_remove_unfair_posloss_train_conf.log &

#     # CUDA_VISIBLE_DEVICES=6 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_step 2000 --conf entropy  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 3 --sel_round $SR --remove_pos --train_conf > ./logs/fair_train/s3_dp_02_new256_100round_sel_$SR\_case1_remove_unfair_posloss_train_conf.log &



#     CUDA_VISIBLE_DEVICES=5 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_step 2000 --conf entropy  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 4 --sel_round $SR --remove_pos --train_conf > ./logs/fair_train/s4_dp_02_new256_100round_sel_$SR\_case1_remove_unfair_posloss_train_conf.log &

#     CUDA_VISIBLE_DEVICES=5 nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_step 2000 --conf entropy  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy 5 --sel_round $SR --remove_pos --train_conf > ./logs/fair_train/s5_dp_02_new256_100round_sel_$SR\_case1_remove_unfair_posloss_train_conf.log &

#     sleep 45m

    


# done

# SR="25"

conf="entropy peer"
# conf="no_conf"
# sel_round="5 10"
sel_round="5"
train_ratio="0.02 0.05 0.1 0.2 0.4 0.6"
val_ratio="0.2 0.4 0.6 0.8"
exp="1 2 3"
strategy="6"
conf_method="TV V"




i=0
j=0
for MYEXP in $exp
do

# for SR in $sel_round
# do

for TR in $train_ratio
do


for VR in $val_ratio
do

for MYCONF in $conf
do

    for CM in $conf_method
    do
        for STG in $strategy
        do
            echo GPU: $i. Task: $j. Rrunning for ./logs/fair_train/s$STG\_dp_02_new256_100round_sel_$SR\_$MYCONF\_exp$MYEXP\_$CM.log
            # CUDA_VISIBLE_DEVICES=$i nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_step 0 --conf $MYCONF  --metric dp --label_ratio $TR --val_ratio $VR --strategy $STG --sel_round 5 --remove_pos --exp $MYEXP --conf_method $CM > ./logs/fair_train/s$STG\_dp_02_new256_100round_val_$VR\_$TR\_$MYCONF\_exp$MYEXP\_$CM.log &
            # sleep 1s
            
            j=$((j+1))
            if [[ $j -eq 2 ]]
            then
                i=$((i+1))
                j=0
            fi
            if [[ $i -eq 7 ]]
            then
                i=0
                echo wait
                wait
            fi
            
        done
    done


done

done
done
done

# conf="entropy peer"
conf="no_conf"
sel_round="5"
val_ratio="0.2 0.4 0.6 0.8"
train_ratio="0.02 0.05 0.1 0.2 0.4 0.6"
exp="1 2 3"
strategy="6"
conf_method="TV"




for MYEXP in $exp
do

for TR in $train_ratio
do


# for SR in $sel_round
# do
for VR in $val_ratio
do

for MYCONF in $conf
do

    for CM in $conf_method
    do
        for STG in $strategy
        do
            echo GPU: $i. Task: $j. Rrunning for ./logs/fair_train/s$STG\_dp_02_new256_100round_sel_$SR\_$MYCONF\_exp$MYEXP\_$CM.log
            # CUDA_VISIBLE_DEVICES=$i nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_step 0 --conf $MYCONF  --metric dp --label_ratio $TR --val_ratio $VR --strategy $STG --sel_round 5 --remove_pos --exp $MYEXP --conf_method $CM > ./logs/fair_train/s$STG\_dp_02_new256_100round_val_$VR\_$TR\_$MYCONF\_exp$MYEXP\_$CM.log &
            # sleep 1s

            j=$((j+1))
            if [[ $j -eq 2 ]]
            then
                i=$((i+1))
                j=0
            fi
            if [[ $i -eq 7 ]]
            then
                i=0
                echo wait
                wait
            fi
            
        done
    done


done
done
done
done