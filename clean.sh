echo "clean up files `date`"
cd $datapath2

# every 06z save 20 member + ens mean restarts.
#if ($analdatem1 >= 2016010400 && $ensmean_restart == 'true' && $hr == '06') then
#    /bin/rm -rf restarts
#    mkdir -p restarts/ensmean
#    /bin/mv -f ensmean/INPUT restarts/ensmean
#    nanal=1
#    while ($nanal <= 20) 
#       charmem="mem`printf %03i $nanal`"
#       /bin/cp -R ${charmem} restarts
#       /bin/rm -f restarts/*/PET* restarts/*/log*
#       @ nanal = $nanal + 1
#    end
#fi

# move every member files to a temp dir.
/bin/rm -rf fgens fgens2
mkdir fgens
mkdir fgens2
/bin/rm -f mem*/*nc mem*/*txt mem*/*grb mem*/*dat mem*/co2* diag*mem*
/bin/rm -f ${charnanal}/*nc ${charnanal}/*txt ${charnanal}/*grb ${charnanal}/*dat ${charnanal}/co2*
/bin/mv -f mem* fgens
/bin/mv -f sfg*mem* fgens2
/bin/mv -f bfg*mem* fgens2

#mkdir analens
#/bin/mv -f sanl_*mem* analens # save analysis ensemble
#echo "files moved to analens `date`"
/bin/rm -f sanl_*mem* # don't save analysis ensemble

echo "files moved to fgens, fgens2 `date`"

/bin/rm -f hostfile*
/bin/rm -f fort*
/bin/rm -f *log
/bin/rm -f *lores *mem*orig
/bin/rm -f ozinfo convinfo satinfo scaninfo anavinfo
/bin/rm -rf *tmp* nodefile* machinefile*
/bin/rm -rf hybridtmp*
if [ $save_hpss == "false" ] && [ $save_hpss_full == "true" ]; then
  echo "keeping fgens,fgens2"
else
  /bin/rm -rf fgens fgens2
fi
echo "unwanted files removed `date`"
wait
