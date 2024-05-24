# sh submit_job.sh <machine>
machine=$1
cat ${machine}_preamble config.sh > job.sh
sbatch job.sh
