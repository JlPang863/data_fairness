conf="no_conf"
sel_round="5 10 15 20 25"
exp="1 2 3"
strategy="2"
conf_method="TV V"




for MYEXP in $exp
do



for SR in $sel_round
do

i=2

for MYCONF in $conf
do

    for CM in $conf_method
    do
        for STG in $strategy
        do
            echo running for ./logs/fair_train/s$STG\_dp_02_new256_100round_sel_$SR\_$MYCONF\_exp$MYEXP\_$CM.log
            CUDA_VISIBLE_DEVICES=$i nohup python3 run_celeba_fair_learn.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_epoch -1 --conf $MYCONF  --metric dp --label_ratio 0.02 --val_ratio 0.1 --strategy $STG --sel_round $SR --remove_pos --exp $MYEXP --conf_method $CM > ./logs/fair_train/s$STG\_dp_02_new256_100round_sel_$SR\_$MYCONF\_exp$MYEXP\_$CM.log &
            
        done
    done

i=$((i+1))
# echo $i
done
wait 

done
done

