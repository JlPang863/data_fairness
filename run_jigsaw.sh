

# sel_layers="4"
# #strategy="1 2 5 6"
# strategy="7"
# metric="dp eop eod"
# # label_ratio="0.05" #default setting 
# warm_epoch="50"
# new_prob="0.5"
# gpu_index="0"
# i=$gpu_index
# j=0


####JTT
sel_layers="4"
strategy="1 6 8 7 2"
# strategy="8 2"
metric="dp eop eod"
# metric="dp eop"
# label_ratio="0.05" #default setting 
label_ratio="0.1" # JTT
warm_epoch="10"
epoch="50"
new_prob="0.9"
# new_prob="0.5"
runs="2"

type='default'
# type='budget'
#start point
gpu_start_index="2"
gpu_final_index="8"

i=$gpu_start_index
i_final=$gpu_final_index
j=0



for LAYER in $sel_layers
do

for STG in $strategy
do

for MTC in $metric
do


echo GPU: $i. Task: $j. Rrunning for ./logs/fair_sampling/jigsaw-runs$runs/label_s$STG\_$MTC\_$LAYER\_$type.log

# example
#CUDA_VISIBLE_DEVICES=6 python3 run_jigsaw.
#CUDA_VISIBLE_DEVICES=6 python3 run_jigsaw.py --strategy 2 --new_prob 0.5 --label_ratio 0.05 --warm_epoch 50

CUDA_VISIBLE_DEVICES=$i nohup python3 run_jigsaw.py --metric $MTC --epoch $epoch --runs $runs --label_ratio $label_ratio --val_ratio 0.2 --new_prob $new_prob --warm_epoch $warm_epoch --strategy $STG --sel_layers $LAYER > ./logs/fair_sampling/jigsaw-runs$runs/label_s$STG\_$MTC\_$LAYER\_$type.log  &



j=$((j+1))
if [[ $j -eq 4 ]]
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


##########################################################################################

# # sel_layers="4"
# # #strategy="1 2 5 6"
# # strategy="6 5"
# # metric="dp eop eod"
# # label_ratio="0.05"
# # warm_epoch="50"
# # new_prob="0.5"
# # gpu_index="0"
# # label_budget="128 256 512 1024 2048"
# # i=$gpu_index
# # j=0


# # use for testing the sentisitivy of label budget
# sel_layers="4"
# strategy="6 5"
# metric="dp"
# label_ratio="0.05"
# warm_epoch="50"
# new_prob="0.5"
# gpu_index="0"
# label_budget="512"

# #start point
# gpu_start_index="2"
# gpu_final_index="4"

# i=$gpu_start_index
# i_final=$gpu_final_index
# j=0



# for LAYER in $sel_layers
# do

# for STG in $strategy
# do

# for MTC in $metric
# do

# for LABEL_BUDGET in $label_budget
# do
# # echo GPU: $i. Task: $j. Rrunning for ./logs/fair_sampling/$LABEL\_s$STG\_$MTC\_$LAYER.log

# # example
# #CUDA_VISIBLE_DEVICES=6 python3 run_jigsaw.
# #CUDA_VISIBLE_DEVICES=6 python3 run_jigsaw.py --strategy 2 --new_prob 0.5 --label_ratio 0.05 --warm_epoch 50

# CUDA_VISIBLE_DEVICES=$i nohup python3 run_jigsaw.py --metric $MTC --label_ratio $label_ratio --label_budget $LABEL_BUDGET --val_ratio 0.2 --new_prob $new_prob --warm_epoch $warm_epoch --strategy $STG --sel_layers $LAYER > ./logs/fair_sampling/jigsaw/label_s$STG\_$MTC\_$LAYER\_$LABEL_BUDGET.log  &



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

# # wait

