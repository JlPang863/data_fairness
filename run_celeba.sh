# CUDA_VISIBLE_DEVICES=0 nohup python3 run_celeba.py --method plain  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf entropy  --metric dp --label_ratio 0.05 --val_ratio 0.1 --strategy 5 --remove_pos --train_conf > s5_dp_05_new256_100round_case1_remove_unfair_posloss_train_conf.log &


# CUDA_VISIBLE_DEVICES=0 python3 run_celeba.py --method plain  --warm_epoch 0  --metric dp --label_ratio 0.05 --val_ratio 0.1 --strategy 2 
####################################################################################################################################################################################################
### table:default setting

sel_layers="4"
# strategy="1 6 8 7 2"
strategy="7"
label_key="Smiling Attractive"
# label_key="Smiling  Young "
metric="dp eop eod"
# metric="eod"
# type='1024'
type='default'
runs="1"
warm_epoch="2"
epoch="10"

#start point
gpu_start_index="2"
gpu_final_index="8"

i=$gpu_start_index
i_final=$gpu_final_index
j=0


for LABEL in $label_key
do

for LAYER in $sel_layers
do

for STG in $strategy
do

for MTC in $metric
do


echo GPU: $i. Task: $j. Rrunning for ./logs/fair_sampling/celeba-runs$runs/$LABEL\_s$STG\_$MTC\_$LAYER\_$type.log

CUDA_VISIBLE_DEVICES=$i nohup python3 run_celeba.py --method plain --epoch $epoch --runs $runs --warm_epoch $warm_epoch --metric $MTC --label_ratio 0.02 --val_ratio 0.1 --strategy $STG --sel_layers $LAYER --label_key $LABEL > ./logs/fair_sampling/celeba-runs$runs/$LABEL\_s$STG\_$MTC\_$LAYER\_$type.log & 

j=$((j+1))
if [[ $j -eq 1 ]]
then
    i=$((i+1))
    j=0
fi
if [[ $i -eq $i_final ]]
then
    i=$gpu_index
    echo wait
    wait
fi


done
done
done
done

# ####################################################################################################################################################################################################
# '''
# sensitivity analysis
# '''
# # sel_layers="4"
# # strategy="1"
# # # label_key="Smiling Straight_Hair Attractive"
# # #label_key="Smiling Attractive Young Big_Nose"
# # label_key="Young"
# # metric="eop"
# # #metric="dp"
# # gpu_index="0"
# # i=$gpu_index
# # j=0

# sel_layers="4"
# strategy="6"
# # label_key="Smiling Straight_Hair Attractive"
# #label_key="Smiling Attractive Young Big_Nose"
# label_key="Smiling"
# metric="dp"
# gpu_index="0"
# label_budget="256"


# #start point
# gpu_start_index="3"
# gpu_final_index="4"

# i=$gpu_start_index
# i_final=$gpu_final_index
# j=0


# for LABEL in $label_key
# do

# for LAYER in $sel_layers
# do

# for STG in $strategy
# do

# for MTC in $metric
# do

# for LABEL_BUDGET in $label_budget
# do

# echo GPU: $i. Task: $j. Rrunning for ./logs/fair_sampling/celeba/$LABEL\_s$STG\_$MTC\_$LAYER\_$LABEL_BUDGET.log

# CUDA_VISIBLE_DEVICES=$i nohup python3 run_celeba.py --method plain  --warm_epoch 0 --label_budget $LABEL_BUDGET --metric $MTC --label_ratio 0.02 --val_ratio 0.1 --strategy $STG --sel_layers $LAYER --label_key $LABEL > ./logs/fair_sampling/celeba/$LABEL\_s$STG\_$MTC\_$LAYER\_$LABEL_BUDGET.log & 

# j=$((j+1))
# if [[ $j -eq 2 ]]
# then
#     i=$((i+1))
#     j=0
# fi
# if [[ $i -eq $i_final ]]
# then
#     i=$gpu_index
#     echo wait
#     wait
# fi

# done
# done
# done
# done
# done



##################################################################################################
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

############################################################################################################
# # transformer
# wait

# sel_layers="4"
# strategy="1"
# label_key="Smiling Straight_Hair Attractive Pale_Skin Young Big_Nose"
# # label_key="Pale_Skin Young Big_Nose"
# metric="dp eop eod"
# TOL="0.02"

# i=1
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

# CUDA_VISIBLE_DEVICES=$i nohup python3 run_celeba.py --method plain  --warm_epoch 2  --metric $MTC --label_ratio 0.02 --val_ratio 0.1 --strategy $STG --sel_layers $LAYER --label_key $LABEL --tol $TOL --without_label > ./logs/fair_sampling/celeba/$LABEL\_s$STG\_$MTC\_$LAYER\_tol$TOL\_wolb.log & 

# j=$((j+1))
# if [[ $j -eq 3 ]]
# then
#     i=$((i+1))
#     j=0
# fi
# if [[ $i -eq 8 ]]
# then
#     i=1
#     echo wait
#     wait
# fi

# done
# done
# done
# done


# # sel_layers="2 4"
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

# CUDA_VISIBLE_DEVICES=$i nohup python3 run_celeba.py --method plain  --warm_epoch 2  --metric $MTC --label_ratio 0.02 --val_ratio 0.1 --strategy $STG --sel_layers $LAYER --label_key $LABEL  --tol $TOL --without_label > ./logs/fair_sampling/celeba/$LABEL\_s$STG\_$MTC\_$LAYER\_tol$TOL\_wolb.log & 


# j=$((j+1))
# if [[ $j -eq 3 ]]
# then
#     i=$((i+1))
#     j=0
# fi
# if [[ $i -eq 8 ]]
# then
#     i=1
#     echo wait
#     wait
# fi

# done
# done
# done
# done

