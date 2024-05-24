#!/bin/sh
#SBATCH -q urgent
#SBATCH -t 08:00:00
#SBATCH -A da-cpu
#SBATCH -N 20  
#SBATCH --ntasks-per-node=80
#SBATCH -p hercules
#SBATCH -J run_neuralgcm
#SBATCH -e run_neuralgcm.out
#SBATCH -o run_neuralgcm.out

#datapath=/work2/noaa/gsienkf/whitaker/neuralgcm_enkf_psonly3
#analdate=2021082918
#analdatep1=2021083000
#datapath2=${datapath}/${analdate}
#datapathp1=${datapath}/${analdatep1}
#nanals=80
#scriptsdir=$PWD
#corespernode=$SLURM_CPUS_ON_NODE
#NODES=$SLURM_NNODES
#FHMIN=3
#FHMAX=9
#FHOUT=3

runspernode=`expr $nanals \/ $NODES`
coresperrun=`expr $corespernode \/ $runspernode`
export OMP_NUM_THREADS=$coresperrun

module purge
YYYYMMDD=`echo $analdate | cut -c1-8`

mkdir -p $datapath2
mkdir -p $datapathp1
cd $datapath2
model_type='stochastic_1_4'
ln -fs /work2/noaa/gsienkf/whitaker/python/run_neuralgcm/neural_gcm_${model_type}_deg_v0.pkl .
ln -fs /work2/noaa/gsienkf/whitaker/python/run_neuralgcm/neural_gcm_dynamic_forcing_${model_type}_deg.pkl .
ln -fs /work2/noaa/gsienkf/whitaker/python/run_neuralgcm/era5_init_1p4deg_${YYYYMMDD}.nc .
ln -fs /work2/noaa/gsienkf/whitaker/python/run_neuralgcm/era5_init_0p7deg_${YYYYMMDD}.nc .
nanal=1
while [ $nanal -le $nanals ]; do
ISEED=$((analdate*1000 + nanal*10 + 1))
charnanal="mem`printf %03i $nanal`"
/bin/rm -f sfg_${analdatep1}*${charnanal}
if [ $LEVS -eq 32 ]; then # sigma version
  srun -N 1 -n 1 -c $coresperrun --ntasks-per-node=$runspernode --cpu-bind=cores /work/noaa/gsienkf/whitaker/miniconda3/bin/python ${scriptsdir}/run_neuralgcm_sigma.py ${model_type} $analdate $nanal $ISEED sanl_${analdate}_fhr06_${charnanal} &
elif [ $LEVS -eq 37 ]; then # pressure level version
  srun -N 1 -n 1 -c $coresperrun --ntasks-per-node=$runspernode --cpu-bind=cores /work/noaa/gsienkf/whitaker/miniconda3/bin/python ${scriptsdir}/run_neuralgcm_plevs.py ${model_type} $analdate $nanal $ISEED sanl_${analdate}_fhr06_${charnanal} &
else
  echo "incorrect value for LEVS"
  exit 1
fi
nanal=$((nanal+1))
done
wait

/bin/mv -f sfg_${analdatep1}*mem* ${datapathp1}

nanal=1
anyfilemissing='no'
while [ $nanal -le $nanals ]; do
    export charnanal="mem`printf %03i $nanal`"
    fhr=$FHMIN
    outfiles=""
    while [ $fhr -le $FHMAX ]; do
       charhr="fhr`printf %02i $fhr`"
       outfiles="${outfiles} ${datapath}/${analdatep1}/sfg_${analdatep1}_${charhr}_${charnanal}"
       fhr=$((fhr+FHOUT))
    done
    filemissing='no'
    for outfile in $outfiles; do
      ls -l $outfile
      if [ ! -s $outfile ]; then 
        echo "${outfile} is missing"
        filemissing='yes'
        anyfilemissing='yes'
      else
        echo "${outfile} is OK"
      fi
    done 
    nanal=$((nanal+1))
done

if [ $anyfilemissing == 'yes' ]; then
    echo "there are output files missing!"
    exit 1
else
    echo "all output files seem OK"
    date
    exit 0
fi
