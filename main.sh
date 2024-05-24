#!/bin/sh

# main driver script
# gsi gain or gsi covariance GSI EnKF (based on ensemble mean background)
# optional high-res control fcst replayed to ens mean analysis

# allow this script to submit other scripts with LSF
unset LSB_SUB_RES_REQ 

echo "nodes = $NODES"

idate_job=1

while [ $idate_job -le ${ndates_job} ]; do

source $datapath/fg_only.sh # define fg_only variable.

export startupenv="${datapath}/analdate.sh"
source $startupenv
# substringing to get yr, mon, day, hr info
export yr=`echo $analdate | cut -c1-4`
export mon=`echo $analdate | cut -c5-6`
export day=`echo $analdate | cut -c7-8`
export hr=`echo $analdate | cut -c9-10`
# previous analysis time.
export FHOFFSET=`expr $ANALINC \/ 2`
export analdatem1=`${incdate} $analdate -$ANALINC`
# next analysis time.
export analdatep1=`${incdate} $analdate $ANALINC`
# beginning of current assimilation window
export analdatem3=`${incdate} $analdate -$FHOFFSET`
# beginning of next assimilation window
export analdatep1m3=`${incdate} $analdate $FHOFFSET`
export hrp1=`echo $analdatep1 | cut -c9-10`
export hrm1=`echo $analdatem1 | cut -c9-10`

#------------------------------------------------------------------------
mkdir -p $datapath

echo "BaseDir: ${basedir}"
echo "EnKFBin: ${enkfbin}"
echo "DataPath: ${datapath}"

############################################################################
# Main Program

env
echo "starting the cycle (${idate_job} out of ${ndates_job})"

export datapath2="${datapath}/${analdate}/"
/bin/cp -f ${scriptsdir}/gdas*abias* $datapath2

# setup node parameters used in psop_ncio.sh and compute_ensmean_fcst.sh
export mpitaskspernode=`python -c "import math; print(int(math.ceil(float(${nanals})/float(${NODES}))))"`
if [ $mpitaskspernode -lt 1 ]; then
  export mpitaskspernode 1
fi
export OMP_NUM_THREADS=`expr $corespernode \/ $mpitaskspernode`
echo "mpitaskspernode = $mpitaskspernode threads = $OMP_NUM_THREADS"
export nprocs=$nanals

export datapathp1="${datapath}/${analdatep1}/"
export datapathm1="${datapath}/${analdatem1}/"
mkdir -p $datapathp1
export CDATE=$analdate

date
echo "analdate minus 1: $analdatem1"
echo "analdate: $analdate"
echo "analdate plus 1: $analdatep1"

# make log dir for analdate
export current_logdir="${datapath2}/logs"
echo "Current LogDir: ${current_logdir}"
mkdir -p ${current_logdir}

export PREINP="${RUN}.t${hr}z."
export PREINP1="${RUN}.t${hrp1}z."
export PREINPm1="${RUN}.t${hrm1}z."

if [ $fg_only ==  'false' ]; then

niter=1
alldone="no"
while [ $alldone == 'no' ] && [ $niter -le $nitermax ]; do
   echo "$analdate starting ens mean computation `date`"
   sh ${scriptsdir}/compute_ensmean_fcst.sh >  ${current_logdir}/compute_ensmean_fcst.out 2>&1
   errstatus=$?
   if [ $errstatus -ne 0 ]; then
       echo "failed computing ensemble mean, try again..."
       alldone="no"
       if [ $niter -eq $nitermax ]; then
           echo "giving up"
           exit 1
       fi
   else
       echo "$analdate done computing ensemble mean `date`"
       alldone="yes"
   fi
   niter=$((niter+1))
done

# compute forward obs operator for ps obs (create ncdiag files)
# (psop_ncio)
if [ $modelspace_vloc ==  ".true." ]; then # run on ens mean, save jacobian
   pushd ${datapath2}
   cat > psop.nml << EOF
&nam_psop
  nlevt=${nlevt},fhmin=$FHMIN,fhmax=$FHMAX,fhout=${FHOUT},
  datestring="${analdate}",obsfile="${obs_datapath}/psobs_${analdate}.txt",zthresh=1000,ps_ind=128,
/
EOF
   cat psop.nml
   ${execdir}/psop_ncio_ensmean.x > ${current_logdir}/psop_ncio.out 2>&1
   popd
else # run on every member (and ensemble mean)
   pushd ${datapath2}
   nanalsp1=`expr $nanals + 1`
   #export mpitaskspernode=`expr $nanalsp1 \/ $NODES + 1`
   #export nprocs=$nanalsp1
   #export OMP_NUM_THREADS=`expr $corespernode \/ $mpitaskspernode`
   export mpitaskspernode=$corespernode
   export nprocs=$nanalsp1
   export OMP_NUM_THREADS=1
   cat > psop.nml << EOF
&nam_psop
  nlevt=${nlevt},fhmin=$FHMIN,fhmax=$FHMAX,fhout=${FHOUT},nanals=${nanals},
  datestring="${analdate}",obsfile="${obs_datapath}/psobs_${analdate}.txt",zthresh=1000,
/
EOF
   cat psop.nml
   export PGM="${execdir}/psop_ncio.x"
   echo "mpitaskspernode = $mpitaskspernode threads = $OMP_NUM_THREADS"
   ${scriptsdir}/runmpi > ${current_logdir}/psop_ncio.out 2>&1
   popd
fi


# run enkf analysis.
if [ $noDA == "true" ]; then
    nanal=1
    while [ $nanal -le $nanals ]; do
       charnanal=`printf %03i $nanal`
       /bin/cp -f ${datapath2}/sfg_${analdate}_fhr06_mem${charnanal} ${datapath2}/sanl_${analdate}_fhr06_mem${charnanal}
       nanal=$((nanal+1))
    done
    /bin/cp -f ${datapath2}/sfg_${analdate}_fhr06_ensmean ${datapath2}/sanl_${analdate}_fhr06_ensmean
else
    echo "$analdate run enkf `date`"
    sh ${scriptsdir}/runenkf.sh > ${current_logdir}/run_enkf.out 2>&1
    # once enkf has completed, check log files.
    enkf_done=`cat ${current_logdir}/run_enkf.log`
    if [ $enkf_done == 'yes' ]; then
      echo "$analdate enkf analysis completed successfully `date`"
    else
      echo "$analdate enkf analysis did not complete successfully, exiting `date`"
      exit 1
    fi
fi

# compute ensemble mean analyses.
if [ $write_ensmean == ".false." ]; then
   echo "$analdate starting ens mean analysis computation `date`"
   sh ${scriptsdir}/compute_ensmean_enkf.sh > ${current_logdir}/compute_ensmean_anal.out 2>&1
   echo "$analdate done computing ensemble mean analyses `date`"
fi

fi # skip to here if fg_only = true

echo "$analdate run enkf ens first guess `date`"


sh ${scriptsdir}/run_neuralgcm.sh > ${current_logdir}/run_neuralgcm.out 2>&1
errstatus=$?
if [ $errstatus -ne 0 ]; then
    echo "failed to run enkf ens first guess, giving up..."
    exit 1
else
    echo "$analdate done running enkf ens first guess `date`"
fi

#sh ${scriptsdir}/run_fg_ens.sh > ${current_logdir}/run_fg_ens.out  2>&1
#ens_done=`cat ${current_logdir}/run_fg_ens.log`
#if [ $ens_done == 'yes' ]; then
#  echo "$analdate enkf first-guess completed successfully `date`"
#else
#  echo "$analdate enkf first-guess did not complete successfully, exiting `date`"
#  exit 1
#fi

if [ $cold_start == 'false' ]; then

# cleanup
# only save full ensemble data to hpss if checkdate.py returns 0
# a subset will be saved if save_hpss_subset="true" and save_hpss="true"
date_check=`python ${homedir}/checkdate.py ${analdate}`
if [ $date_check -eq 0 ]; then
   export save_hpss_full="true"
else
   export save_hpss_full="false"
fi
if [ $do_cleanup == 'true' ]; then
   sh ${scriptsdir}/clean.sh > ${current_logdir}/clean.out 2>&1
fi # do_cleanup = true

wait # wait for backgrounded processes to finish

cd $homedir
if [ $save_hpss == 'true' ]; then
   cat ${machine}_preamble_hpss hpss.sh > job_hpss.sh
   sbatch --export=ALL job_hpss.sh
fi
#sbatch --export=machine=${machine},analdate=${analdate},datapath2=${datapath2},hsidir=${hsidir},save_hpss_full=${save_hpss_full},save_hpss_subset=${save_hpss_subset} job_hpss.sh

fi # skip to here if cold_start = true

echo "$analdate all done"

# next analdate: increment by $ANALINC
export analdate=`${incdate} $analdate $ANALINC`

echo "export analdate=${analdate}" > $startupenv
echo "export analdate_end=${analdate_end}" >> $startupenv
echo "export fg_only=false" > $datapath/fg_only.sh
echo "export cold_start=false" >> $datapath/fg_only.sh

cd $homedir

if [ $analdate -le $analdate_end ]; then
  idate_job=$((idate_job+1))
else
  idate_job=$((ndates_job+1))
fi

done # next analysis time


if [ $analdate -le $analdate_end ]  && [ $resubmit == 'true' ]; then
   echo "current time is $analdate"
   if [ $resubmit == 'true' ]; then
      echo "resubmit script"
      echo "machine = $machine"
      cat ${machine}_preamble config.sh > job.sh
      sbatch --export=ALL job.sh
   fi
fi

exit 0
