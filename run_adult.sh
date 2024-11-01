

dataset="adult"
sel_layers="4"
strategy="1 6 8 7 2"
metric="dp eop eod"
label_ratio="0.2"
new_prob="0.9"
warm_epoch="100" #default
epoch="60"

group_key="age"
type='default'
val_ratios='0.01 0.05 0.1 0.2' #default value 0.2
runs="0"

#start point
gpu_start_index="0"
gpu_final_index="1"

i=$gpu_start_index
i_final=$gpu_final_index
j=0



for LAYER in $sel_layers
do

for STG in $strategy
do

for val_ratio in $val_ratios
do 

for MTC in $metric
do

echo GPU: $i. Task: $j. Running for ./logs/fair_sampling/adult-$group_key-runs$runs/label_s$STG\_$MTC\_$LAYER\_$type\_$val_ratio.log &

CUDA_VISIBLE_DEVICES=$i nohup python3 run_adult.py --metric $MTC --epoch $epoch --label_ratio $label_ratio --runs $runs --group_key $group_key --val_ratio $val_ratio --new_prob $new_prob --warm_epoch $warm_epoch --strategy $STG --sel_layers $LAYER > ./logs/fair_sampling/adult-$group_key-runs$runs/label_s$STG\_$MTC\_$LAYER\_$type\_$val_ratio.log  &


j=$((j+1))
if [[ $j -eq 6 ]]
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



