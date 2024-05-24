date=$1
expt=$3
while [ $date -le $2 ]; do
 python getz500_plev.py $3 $date 1>> z500err_plev_stochastic_1_4.txt 2>> z500err.err
 date=`./incdate.sh $date 6`
done
