#!/bin/sh

export nprocs=`expr $cores \/ $enkf_threads`
export mpitaskspernode=`expr $corespernode \/ $enkf_threads`
export OMP_NUM_THREADS=$enkf_threads
export OMP_STACKSIZE=512M
export MKL_NUM_THREADS=1
source $MODULESHOME/init/sh
module list

iaufhrs2=`echo $iaufhrs | sed 's/,/ /g'`

for nfhr in $iaufhrs2; do
  charfhr="fhr"`printf %02i $nfhr`
  # check output files.
  nanal=1
  filemissing='no'
  while [ $nanal -le $nanals ]; do
     charnanal="mem"`printf %03i $nanal`
     analfile="${datapath2}/sanl_${analdate}_${charfhr}_${charnanal}"
     if [ ! -s $analfile ]; then
        filemissing='yes'
     fi
     nanal=$((nanal+1))
  done
done


if [ $filemissing == 'yes' ]; then

echo "computing enkf update..."

date
cd ${datapath2}

if [ $modelspace_vloc == ".true." ]; then
 lobsdiag_forenkf=.true.
else
 lobsdiag_forenkf=.false.
fi 

cat <<EOF > enkf.nml
 &nam_enkf
  datestring="$analdate",datapath="$datapath2",univaroz=.false.,numiter=0,
  analpertwtnh=$analpertwtnh,analpertwtsh=$analpertwtsh,analpertwttr=$analpertwttr,
  analpertwtnh_rtpp=$analpertwtnh_rtpp,analpertwtsh_rtpp=$analpertwtsh_rtpp,analpertwttr_rtpp=$analpertwttr_rtpp,
  covinflatemax=$covinflatemax,covinflatemin=$covinflatemin,pseudo_rh=$pseudo_rh,
  corrlengthnh=$corrlengthnh,corrlengthsh=$corrlengthsh,corrlengthtr=$corrlengthtr,
  lnsigcutoffnh=$lnsigcutoffnh,lnsigcutoffsh=$lnsigcutoffsh,lnsigcutofftr=$lnsigcutofftr,
  lnsigcutoffsatnh=$lnsigcutoffsatnh,lnsigcutoffsatsh=$lnsigcutoffsatsh,lnsigcutoffsattr=$lnsigcutoffsattr,
  lnsigcutoffpsnh=$lnsigcutoffpsnh,lnsigcutoffpssh=$lnsigcutoffpssh,lnsigcutoffpstr=$lnsigcutoffpstr,
  nlons=$LONB,nlats=$LATB,smoothparm=$SMOOTHINF,letkf_bruteforce_search=${letkf_bruteforce_search},
  readin_localization=.false.,saterrfact=$saterrfact,sprd_tol=$sprd_tol,
  covl_minfact=$covl_minfact,covl_efold=$covl_efold,paoverpb_thresh=$paoverpb_thresh,letkf_flag=$letkf_flag,denkf=$denkf,
  getkf_inflation=$getkf_inflation,letkf_novlocal=$letkf_novlocal,modelspace_vloc=$modelspace_vloc,save_inflation=.false.,
  reducedgrid=${reducedgrid},nlevs=$LEVS,nanals=$nanals,deterministic=$deterministic,imp_physics=$imp_physics,
  lobsdiag_forenkf=${lobsdiag_forenkf},write_spread_diag=.false.,netcdf_diag=.true.,
  sortinc=$sortinc,nhr_anal=$iaufhrs,nhr_state=$enkfstatefhrs,getkf=$getkf,taperanalperts=${taperanalperts},
  use_correlated_oberrs=${use_correlated_oberrs},use_gfs_ncio=.true.,nccompress=T,paranc=F,write_fv3_incr=${write_fv3_increment},
  adp_anglebc=.true.,angord=4,newpc4pred=.true.,use_edges=.false.,emiss_bc=.true.,biasvar=-500,nobsl_max=$nobsl_max,
  ${ENKFVARS}
  ${WRITE_INCR_ZERO}
 /
 &satobs_enkf
 /
 &END
 &ozobs_enkf
 /
 &END
EOF

cat enkf.nml

cp ${scriptsdir}/vlocal_eig_L${LEVS}.dat ${datapath2}/vlocal_eig.dat

/bin/rm -f ${datapath2}/enkf.log
/bin/mv -f ${current_logdir}/ensda.out ${current_logdir}/ensda.out.save
export PGM=$enkfbin
echo "OMP_NUM_THREADS = $OMP_NUM_THREADS"

# use same number of tasks on every node.
export nprocs=`expr $cores \/ $OMP_NUM_THREADS`
export mpitaskspernode=`expr $corespernode \/ $OMP_NUM_THREADS`
echo "running with $OMP_NUM_THREADS threads ..."
${scriptsdir}/runmpi > ${current_logdir}/ensda.out 2>&1

if [ ! -s ${datapath2}/enkf.log ]; then
   echo "no enkf log file found"
   exit 1
fi

else
echo "enkf update already done..."
fi # filemissing='yes'

# check output files again.
nanal=1
filemissing='no'
while [ $nanal -le $nanals ]; do
   charnanal="mem"`printf %03i $nanal`
   analfile=${datapath2}/sanl_${analdate}_${charfhr}_${charnanal}
   if [ ! -s $analfile ]; then
     filemissing='yes'
   fi
   nanal=$((nanal+1))
done

if [ $filemissing == 'yes' ]; then
    echo "there are output files missing!"
    exit 1
else
    echo "all output files seem OK `date`"
fi
exit 0
