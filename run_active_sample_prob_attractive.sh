

sel_layers="4"
strategy="5"
label_key="Attractive"
metric="dp eop eod"
TOL="0.02"
prob_all="0.9 0.95 0.99"
# ratio_org="0.1 0.3 0.5 0.7 0.9"
ratio_org="0.9"

# all_gpu=(0 2 4 6)
all_gpu=(1 3 4 7)



i=0
j=0
for RATIO in $ratio_org
do
for LABEL in $label_key
do

for LAYER in $sel_layers
do

for STG in $strategy
do

for MTC in $metric
do

for PROB in $prob_all
do

echo GPU: ${all_gpu[$i]}. Task: $j. Rrunning for ./logs/fair_sampling/$LABEL\_s$STG\_$MTC\_$LAYER.log

# CUDA_VISIBLE_DEVICES=$i nohup python3 run_celeba.py --method plain  --warm_epoch 2  --metric $MTC --label_ratio 0.02 --val_ratio 0.1 --strategy $STG --sel_layers $LAYER --label_key $LABEL --new_prob $PROB --aux_data imagenet  > ./logs/fair_sampling/celeba/$LABEL\_s$STG\_$MTC\_$LAYER\_prob_$PROB\_warm2batch_imgaux.log & 

# CUDA_VISIBLE_DEVICES=$i nohup python3 run_celeba.py --method plain  --warm_epoch 2  --metric $MTC --label_ratio 0.02 --val_ratio 0.1 --strategy $STG --sel_layers $LAYER --label_key $LABEL --new_prob $PROB --half_ablation  > ./logs/fair_sampling/celeba/$LABEL\_s$STG\_$MTC\_$LAYER\_prob_$PROB\_warm2batch_half_ablation.log & 


CUDA_VISIBLE_DEVICES=${all_gpu[$i]} nohup python3 run_celeba.py --method plain  --warm_epoch 2  --metric $MTC --label_ratio 0.02 --val_ratio 0.1 --strategy $STG --sel_layers $LAYER --label_key $LABEL --new_prob $PROB  --aux_data scut  --ratio_org $RATIO  > ./logs/fair_sampling/celeba/$LABEL\_s$STG\_$MTC\_$LAYER\_prob_$PROB\_warm2batch_scut_org_$RATIO.log & 

j=$((j+1))
if [[ $j -eq 3 ]]
then
    i=$((i+1))
    j=0
fi
if [[ $i -eq 4 ]]
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
