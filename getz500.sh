date=$1
expt=$3
while [ $date -le $2 ]; do
 python getz500b.py $3 $date stochastic_1_4 1>> z500err_stochastic_1_4.txt 2>> z500err.err
 date=`./incdate.sh $date 6`
done
