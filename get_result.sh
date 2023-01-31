
# conf="entropy peer"
# sel_round="5 10 15 20 25"
# exp="1 2 3"
# strategy="2"
# conf_method="TV V"

conf="entropy peer"
# sel_round="5 10"
sel_round="5"
exp="1 2 3"
strategy="6"
conf_method="TV V"


# conf="no_conf"
# sel_round="5 10 15 20 25"
# exp="1 2 3"
# strategy="2"
# conf_method="TV"


for MYEXP in $exp
do



for SR in $sel_round
do

i=0

for MYCONF in $conf
do

    for CM in $conf_method
    do
        for STG in $strategy
        do
            echo $SR\_$MYCONF\_exp$MYEXP\_$CM
            # cat ./logs/fair_train/s$STG\_dp_02_new256_100round_warm1000_sel_$SR\_$MYCONF\_exp$MYEXP\_$CM.log | grep "95.15" | grep "test" | awk '{ print $20, $27 }'
            # cat ./logs/fair_train/s$STG\_dp_02_new256_100round_warm2500_sel_$SR\_$MYCONF\_exp$MYEXP\_$CM.log | grep "95.15" | grep "test" | awk '{ print $20, $27 }'
            cat ./logs/fair_train/s$STG\_dp_02_new256_100round_sel_$SR\_$MYCONF\_exp$MYEXP\_$CM.log | grep "98.79" | grep "test" | awk '{ print $20, $27 }'

            
            
        done
    done

i=$((i+1))

done
wait 

done
done


