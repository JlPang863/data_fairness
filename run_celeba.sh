
sel_layers="4"
strategy="1 6 8 7 2"
label_key="Smiling  Young Big_Nose Attractive" 
metric="dp eop eod"
type='default'
runs="0" # random seed
warm_epoch="2"
epoch="5"
val_ratios='0.02 0.05' #default value 0.1

#start point
gpu_start_index="7"
gpu_final_index="8"
num_of_exper_single_gpu=1
i=$gpu_start_index
i_final=$gpu_final_index
j=0


for LABEL in $label_key
do

for LAYER in $sel_layers
do

for STG in $strategy
do

for val_ratio in $val_ratios
do

for MTC in $metric
do


echo GPU: $i. Task: $j. Running for ./logs/fair_sampling/celeba-runs$runs/$LABEL\_s$STG\_$MTC\_$LAYER\_$type\_$val_ratio.log

CUDA_VISIBLE_DEVICES=$i nohup python3 run_celeba.py --method plain --epoch $epoch --runs $runs --warm_epoch $warm_epoch --metric $MTC --label_ratio 0.02 --val_ratio $val_ratio --strategy $STG --sel_layers $LAYER --label_key $LABEL > ./logs/fair_sampling/celeba-runs$runs/$LABEL\_s$STG\_$MTC\_$LAYER\_$type\_$val_ratio.log & 

j=$((j+1))
if [[ $j -eq $num_of_exper_single_gpu ]]
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
done
