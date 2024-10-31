

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
metric="dp eop eod"
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


echo GPU: $i. Task: $j. Running for ./logs/fair_sampling/jigsaw-runs$runs/label_s$STG\_$MTC\_$LAYER\_$type.log

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

