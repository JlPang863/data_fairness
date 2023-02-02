
# conf="entropy peer"
# sel_round="5 10 15 20 25"
# exp="1 2 3"
# strategy="2"
# conf_method="TV V"

# conf="entropy peer"
# # sel_round="5 10"
# sel_round="5"
# exp="1 2 3"
# strategy="6"
# conf_method="TV V"


# conf="no_conf"
# sel_round="5 10 15 20 25"
# exp="1 2 3"
# strategy="2"
# conf_method="TV"
# conf="no_conf"
# conf_method="TV"
# conf="entropy peer"
# conf_method="TV"
# sel_round="5"
# val_ratio="0.2 0.4 0.6 0.8"
# exp="1 2 3"
# strategy="6"



conf="entropy peer"
# conf="no_conf"
# sel_round="5 10"
sel_round="5"
train_ratio="0.02 0.05 0.1 0.2 0.4 0.6"
val_ratio="0.2 0.4 0.8"
exp="1 2 3"
strategy="6"
conf_method="TV V"

for TR in $train_ratio
do


for VR in $val_ratio
do

for MYEXP in $exp
do



# for SR in $sel_round
# do

i=0

for MYCONF in $conf
do

    for CM in $conf_method
    do
        for STG in $strategy
        do
            echo $VR\_$TR\_$MYCONF\_exp$MYEXP\_$CM
            # cat ./logs/fair_train/s$STG\_dp_02_new256_100round_warm1000_sel_$SR\_$MYCONF\_exp$MYEXP\_$CM.log | grep "95.15" | grep "test" | awk '{ print $20, $27 }'
            # cat ./logs/fair_train/s$STG\_dp_02_new256_100round_warm2500_sel_$SR\_$MYCONF\_exp$MYEXP\_$CM.log | grep "95.15" | grep "test" | awk '{ print $20, $27 }'
            # cat ./logs/fair_train/s$STG\_dp_02_new256_100round_sel_$SR\_$MYCONF\_exp$MYEXP\_$CM.log | grep "98.79" | grep "test" | awk '{ print $19, $26 }'

            # cat ./logs/fair_train/s$STG\_dp_02_new256_100round_val_$VR\_$MYCONF\_exp$MYEXP\_$CM.log | grep "98.79" | grep "test" |  awk '{ acc +=  $19; fr += $26 } END {print acc/NR, fr/NR}'
            # cat ./logs/fair_train/s$STG\_dp_02_new256_100round_val_$VR\_$TR\_$MYCONF\_exp$MYEXP\_$CM.log | grep "98.79" | grep "test" |  awk '{ acc +=  $19; fr += $26 } END {print acc/NR, fr/NR}'
            cat ./logs/fair_train/s$STG\_dp_02_new256_100round_val_$VR\_$TR\_$MYCONF\_exp$MYEXP\_$CM.log | grep "97.57\|95.05\|92.52"  | grep "test" |  awk '{ acc +=  $20; fr += $27 } END {print acc/NR, fr/NR}'

            

            

            
            
        done
    done

i=$((i+1))

done
wait 

done
done
done


