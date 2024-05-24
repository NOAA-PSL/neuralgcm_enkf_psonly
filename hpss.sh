# need envars:  machine, analdate, datapath2, hsidir, save_hpss_full, save_hpss_subset

hr=`echo $analdate | cut -c9-10`
analdatem1=`${incdate} $analdate -6`
exitstat=0
if [ $machine == "gaea" ]; then
   htar=/sw/rdtn/hpss/default/bin/htar
   hsi=/sw/rdtn/hpss/default/bin/hsi
else
   source $MODULESHOME/init/sh
   module load hpss
   htar=`which htar`
   hsi=`which hsi`
fi
#env
$hsi ls -l $hsidir
$hsi mkdir ${hsidir}/
cd ${datapath2}

#cd fgens2
#$htar -cvf ${hsidir}/${analdate}_sfgens.tar sfg*fhr06*mem* sfg*fhr06*ensmean
#cd ..
#$hsi ls -l ${hsidir}/${analdate}_sfgens.tar
#if [ $? -ne 0 ]; then
#   echo "htar sfgens failed"
#fi

if [ $save_hpss_full == "true" ]; then
   echo "htar fgens, fgens2"
   /bin/rm -rf gsitmp*
   /bin/rm -rf sanl*mem*
   cd fgens # fgens has FV3 restart files
   $htar -cvf ${hsidir}/${analdate}_fgens.tar * &
   cd ../fgens2 # fgens2 has nemsio files
   $htar -cvf ${hsidir}/${analdate}_fgens2.tar * &
   cd ..
   wait
   $hsi ls -l ${hsidir}/${analdate}_fgens.tar
   if [ $? -eq 0 ]; then
      /bin/rm -rf fgens
   else
      echo "htar fgens failed, not deleting data"
      existat=1
   fi
   $hsi ls -l ${hsidir}/${analdate}_fgens2.tar
   if [ $? -eq 0 ]; then
      /bin/rm -rf fgens2
   else
      echo "htar fgens2 failed, not deleting data"
      existat=1
   fi
else
   echo 'not saving fgens,fgens2 data to hpss, just clean up...'
   /bin/rm -rf fgens
   /bin/rm -rf fgens2
fi

# save restarts at 00UTC
#if [ $analdatem1 -ge 2016010400 ] && [ -s restarts ] && [ $hr == "06" ];  then
#   $htar -cvf ${hsidir}/${analdatem1}_restarts.tar restarts
#   $hsi ls -l ${hsidir}/${analdatem1}_restarts.tar
#   if [  $? -eq 0 ]; then
#      echo "hsi restarts done, deleting data..."
#      /bin/rm -rf restarts
#   else
#      echo "hsi restarts failed ${analdate}..."
#      exitstat=1
#   fi
#fi
 
# remove unwanted files and directories.
nanal=0 
while [ $nanal -le $nanals ]; do
   if [ $nanal -eq 0 ]; then
      charnanal="ensmean"
   else
      charnanal="mem"`printf %03i $nanal`
   fi
   # remove GSI temp dirs
   /bin/rm -rf gsitmp_${charnanal} 
   # remove remaining 'every member' files
   if [ $nanal -gt 0 ]; then
      /bin/rm -rf *${charnanal}*
   fi
   nanal=$[$nanal+1]
done 
cd ..

# now save what's left to HPSS
if  [ $save_hpss_subset == "true" ]; then
   #cd ${datapath2}
   #$htar -cvf ${hsidir}/${analdate}_analens.tar analens
   #$hsi ls -l ${hsidir}/${analdate}_analens.tar
   #if [  $? -eq 0 ]; then
   #   echo "hsi analens done, deleting data..."
   #   /bin/rm -rf analens
   #else
   #   echo "hsi analens failed ${analdate}..."
   #   exitstat=1
   #fi
   #cd ..
   # exclude long forecast directory
   $htar -cvf ${hsidir}/${analdate}_subset.tar ${analdate}/${RUN}* ${analdate}/*ensmean* ${analdate}/*control* ${analdate}/logs
fi
$hsi ls -l ${hsidir}/${analdate}_subset.tar
exitstat=$?
if [  $exitstat -ne 0 ]; then
   echo "hsi subset failed ${analdate} with exit status $exitstat..."
   exit 1
else
   # remove files to save space
   cd ${analdate}
   /bin/rm -f diag*cris* diag*airs* diag*iasi*
   /bin/rm -f *chgres
   /bin/rm -rf ensmean
   #if [ $hr != '00' ]; then
   #    /bin/rm -rf control
   #fi
fi

exit $exitstat
