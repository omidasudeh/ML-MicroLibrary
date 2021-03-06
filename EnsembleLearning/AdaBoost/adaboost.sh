#!/bin/bash
#SBATCH --account=soc-np
#SBATCH --partition=soc-shared-np
#SBATCH --time=01:30:00
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --error=adaboost_job_log/job.%J.err
#SBATCH --output=adaboost_job_log/job.%J.out
module load python3
module load pandas
jbID=$1
for ((iter=$jbID; iter<$((501)); iter+=50)); do
    cmd="python3 ./adaboost.py $iter"
    echo $cmd
    $cmd
    echo "%%%%%%%%%%"
done