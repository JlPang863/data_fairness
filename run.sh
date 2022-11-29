
# label_key="Arched_Eyebrows Attractive Bags_Under_Eyes Bald"
label_key="Smiling"
# conf="entropy no_conf peer"
conf="no_conf entropy"
# metric="dp dp_cov plain"
metric="dp"
# metric="dp dp_cov plain eop eop_cov eod eod_cov"
# warm="0 1"
warm="0"
fe_sel="0 1 2 3 4 5 6"

for MYLABEL in $label_key
do
for MYCONF in $conf
do
for MYWARM in $warm
do
        
    for MYMETRIC in $metric
    do
        for MYSEL in $fe_sel
        do
            LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 nohup run --cpu 8 --type v100-32g -- python3 run_celeba.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_epoch $MYWARM --conf $MYCONF --metric $MYMETRIC --label_key $MYLABEL --fe_sel $MYSEL &
            sleep 1m
        done
    done    
    sleep 1h
done
done
done
wait
