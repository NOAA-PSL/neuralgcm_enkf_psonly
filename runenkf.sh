#!/bin/sh
# need symlinks for satbias_angle, satbias_in, satinfo
#if [ -z $biascorrdir ]; then # cycled bias correction files
#    export GBIAS=${datapathm1}/${PREINPm1}abias
#    export GBIAS_PC=${datapathm1}/${PREINPm1}abias_pc
#    export GBIASAIR=${datapathm1}/${PREINPm1}abias_air
#    #if [ "$cold_start_bias" == "true" ]; then
#    if [ -s ${datapath2}/${PREINP}abias ]; then
#      # if bias correction files have already been created for this analysis time, use them
#      export GBIAS=${datapath2}/${PREINP}abias
#      export GBIAS_PC=${datapath2}/${PREINP}abias_pc
#      export GBIASAIR=${datapath2}/${PREINP}abias_air
#    fi
#else # externally specified bias correction files.
#    export GBIAS=${biascorrdir}/${analdate}//${PREINP}abias
#    export GBIAS_PC=${biascorrdir}/${analdate}//${PREINP}abias_pc
#    export GBIASAIR=${biascorrdir}/${analdate}//${PREINP}abias_air
#fi
#export GSATANG=$fixgsi/global_satangbias.txt # not used, but needs to exist

#ln -fs $GBIAS   ${datapath2}/satbias_in
#ln -fs $GBIAS_PC   ${datapath2}/satbias_pc
#ln -fs $GSATANG ${datapath2}/satbias_angle
#ln -fs ${gsipath}/fix/global_satinfo.txt ${datapath2}/satinfo
#ln -fs ${SATINFO} ${datapath2}/satinfo
#ls -l ${datapath2}/satinfo
#ln -fs ${gsipath}/fix/global_convinfo.txt ${datapath2}/convinfo
ln -fs ${CONVINFO} ${datapath2}/convinfo
ln -fs ${ANAVINFO_ENKF} ${datapath2}/anavinfo
ls -l ${datapath2}/convinfo
#ln -fs ${gsipath}/fix/global_ozinfo.txt ${datapath2}/ozinfo
#ln -fs ${gsipath}/fix/global_scaninfo.txt ${datapath2}/scaninfo
#ln -fs ${current_logdir}/satinfo.out ${datapath2}/fort.207
#ln -fs ${current_logdir}/ozinfo.out ${datapath2}/fort.206
#ln -fs ${current_logdir}/convinfo.out ${datapath2}/fort.205

# remove previous analyses
if [ $cleanup_anal == 'true' ]; then
   /bin/rm -f ${datapath2}/sanl_*mem*
   /bin/rm -f ${datapath2}/sanl_*ensmean
fi

niter=1
alldone='no'
echo "${analdate} compute enkf analysis increment `date`"
while [ $alldone == 'no' ] && [ $niter -le $nitermax ]; do
    echo "${scriptsdir}/${ensda}"
    sh ${scriptsdir}/${ensda} 
    exitstat=$?
    if [ $exitstat -eq 0 ] ; then
       alldone='yes'
    else
       echo "some files missing, try again .."
       niter=$((niter+1))
    fi
done
if [ $alldone == 'no' ]; then
    echo "Tried ${nitermax} times to run ensda and failed: ${analdate}"
    echo "no" > ${current_logdir}/run_enkf.log
else
    echo "yes" > ${current_logdir}/run_enkf.log
fi
echo "${analdate} done computing enkf analysis increment `date`"
