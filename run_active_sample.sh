# CUDA_VISIBLE_DEVICES=0 nohup python3 run_celeba.py --method plain  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf entropy  --metric dp --label_ratio 0.05 --val_ratio 0.1 --strategy 5 --remove_pos --train_conf > s5_dp_05_new256_100round_case1_remove_unfair_posloss_train_conf.log &


# CUDA_VISIBLE_DEVICES=0 python3 run_celeba.py --method plain  --warm_epoch 0  --metric dp --label_ratio 0.05 --val_ratio 0.1 --strategy 2 


# sel_layers="2"
# strategy="1"
# # label_key="Smiling Straight_Hair Attractive"
# label_key="Pale_Skin Young Big_Nose"
# metric="dp eop eod"

# i=0
# j=0

# for LABEL in $label_key
# do

# for LAYER in $sel_layers
# do

# for STG in $strategy
# do

# for MTC in $metric
# do

# echo GPU: $i. Task: $j. Rrunning for ./logs/fair_sampling/$LABEL\_s$STG\_$MTC\_$LAYER.log

# CUDA_VISIBLE_DEVICES=$i nohup python3 run_celeba_res18.py --method plain  --warm_epoch 0  --metric $MTC --label_ratio 0.02 --val_ratio 0.1 --strategy $STG --sel_layers $LAYER --label_key $LABEL > ./logs/fair_sampling/res18/res18_$LABEL\_s$STG\_$MTC\_$LAYER.log & 

# j=$((j+1))
# if [[ $j -eq 1 ]]
# then
#     i=$((i+1))
#     j=0
# fi
# if [[ $i -eq 8 ]]
# then
#     i=0
#     echo wait
#     wait
# fi

# done
# done
# done
# done


# sel_layers="2 4"
# strategy="2 5"
# # label_key="Smiling Straight_Hair Attractive"
# # metric="dp eop eod"



# for LAYER in $sel_layers
# do

# for LABEL in $label_key
# do



# for STG in $strategy
# do

# for MTC in $metric
# do

# echo GPU: $i. Task: $j. Rrunning for ./logs/fair_sampling/$LABEL\_s$STG\_$MTC\_$LAYER.log

# CUDA_VISIBLE_DEVICES=$i nohup python3 run_celeba_res18.py --method plain  --warm_epoch 0  --metric $MTC --label_ratio 0.02 --val_ratio 0.1 --strategy $STG --sel_layers $LAYER --label_key $LABEL > ./logs/fair_sampling/res18/res18_$LABEL\_s$STG\_$MTC\_$LAYER.log & 


# j=$((j+1))
# if [[ $j -eq 1 ]]
# then
#     i=$((i+1))
#     j=0
# fi
# if [[ $i -eq 8 ]]
# then
#     i=0
#     echo wait
#     wait
# fi

# done
# done
# done
# done


# # transformer
# wait

sel_layers="4"
strategy="1"
label_key="Smiling Straight_Hair Attractive Pale_Skin Young Big_Nose"
# label_key="Pale_Skin Young Big_Nose"
metric="dp eop eod"

i=0
j=0

for LABEL in $label_key
do

for LAYER in $sel_layers
do

for STG in $strategy
do

for MTC in $metric
do

echo GPU: $i. Task: $j. Rrunning for ./logs/fair_sampling/$LABEL\_s$STG\_$MTC\_$LAYER.log

CUDA_VISIBLE_DEVICES=$i nohup python3 run_celeba.py --method plain  --warm_epoch 2  --metric $MTC --label_ratio 0.02 --val_ratio 0.1 --strategy $STG --sel_layers $LAYER --label_key $LABEL  > ./logs/fair_sampling/vit/$LABEL\_s$STG\_$MTC\_$LAYER.log & 

j=$((j+1))
if [[ $j -eq 2 ]]
then
    i=$((i+1))
    j=0
fi
if [[ $i -eq 8 ]]
then
    i=0
    echo wait
    wait
fi

done
done
done
done


# sel_layers="2 4"
strategy="2 5"
# label_key="Smiling Straight_Hair Attractive"
# metric="dp eop eod"



for LAYER in $sel_layers
do

for LABEL in $label_key
do



for STG in $strategy
do

for MTC in $metric
do

echo GPU: $i. Task: $j. Rrunning for ./logs/fair_sampling/$LABEL\_s$STG\_$MTC\_$LAYER.log

CUDA_VISIBLE_DEVICES=$i nohup python3 run_celeba.py --method plain  --warm_epoch 2  --metric $MTC --label_ratio 0.02 --val_ratio 0.1 --strategy $STG --sel_layers $LAYER --label_key $LABEL > ./logs/fair_sampling/vit/$LABEL\_s$STG\_$MTC\_$LAYER.log & 


j=$((j+1))
if [[ $j -eq 2 ]]
then
    i=$((i+1))
    j=0
fi
if [[ $i -eq 8 ]]
then
    i=0
    echo wait
    wait
fi

done
done
done
done

