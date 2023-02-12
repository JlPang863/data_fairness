
sel_layers="4"
strategy="1"
metric="dp eop eod"

i=0
j=0



for LAYER in $sel_layers
do

for STG in $strategy
do

for MTC in $metric
do

# echo GPU: $i. Task: $j. Rrunning for ./logs/fair_sampling/$LABEL\_s$STG\_$MTC\_$LAYER.log

CUDA_VISIBLE_DEVICES=$i nohup python3 run_compas.py --metric $MTC --label_ratio 0.05 --val_ratio 0.2 --strategy $STG --sel_layers $LAYER --warm_epoch 5 > ./logs/fair_sampling/compas/label_s$STG\_$MTC\_$LAYER.log & 

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


sel_layers="4"
strategy="2 5"
# label_key="Smiling Straight_Hair Attractive"
# metric="dp eop eod"



for LAYER in $sel_layers
do




for STG in $strategy
do

for MTC in $metric
do

CUDA_VISIBLE_DEVICES=$i nohup python3 run_compas.py --metric $MTC --label_ratio 0.1 --val_ratio 0.2 --strategy $STG --sel_layers $LAYER  --warm_epoch 5 > ./logs/fair_sampling/compas/label_s$STG\_$MTC\_$LAYER.log & 

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

