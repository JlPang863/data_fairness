
sel_layers="4"
# strategy="1 6 8 7 2"
strategy="1"
metric="dp eop eod"
# metric="dp"
label_ratio="0.2"  # default setting
# label_ratio="0.2" #JTT
warm_epoch="20"
type='default'
# type='budget'
epoch="50"
runs="0"

# sel_layers="4"
# strategy="6 5"
# metric="dp"
# label_ratio="0.2"
# warm_epoch="10"
# type='budget'
# val_ratios='0.01 0.05 0.1 0.2 0.25 0.5' #default value 0.2
val_ratios='0.2' #default value 0.2


#start point
gpu_start_index="0"
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

for val_ratio in $val_ratios
do


echo GPU: $i. Task: $j. Rrunning for ./logs/fair_sampling/compas-runs$runs/label_s$STG\_$MTC\_$LAYER\_$type\_$val_ratio.log 

CUDA_VISIBLE_DEVICES=$i nohup python3 run_compas.py --metric $MTC --epoch $epoch --label_ratio $label_ratio --val_ratio $val_ratio --strategy $STG --sel_layers $LAYER --warm_epoch $warm_epoch > ./logs/fair_sampling/compas-runs$runs/label_s$STG\_$MTC\_$LAYER\_$type\_$val_ratio.log & 

j=$((j+1))
if [[ $j -eq 5 ]]
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


# wait


# sel_layers="4"
# strategy="2 3 4 5"
# #label_key="two_year_recid"
# #metric="dp eop eod"



# for LAYER in $sel_layers
# do




# for STG in $strategy
# do

# for MTC in $metric
# do

# CUDA_VISIBLE_DEVICES=$i nohup python3 run_compas.py --metric $MTC --label_ratio $label_ratio --val_ratio 0.2 --strategy $STG --sel_layers $LAYER  --warm_epoch 5 > ./logs/fair_sampling/compas/label_s$STG\_$MTC\_$LAYER.log & 

# j=$((j+1))
# if [[ $j -eq 2 ]]
# then
#     i=$((i+1))
#     j=0
# fi
# if [[ $i -eq 8 ]]
# then
#     i=4
#     echo wait
#     wait
# fi


# done
# done
# done

