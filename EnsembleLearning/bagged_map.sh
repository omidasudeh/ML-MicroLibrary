#!/bin/bash
mkdir -p bagged_trees_predictions
mkdir -p bagged_job_log
for ((job=1; job<$((51)); job+=1)); do
    cmd="sbatch ./bagged.sh $job"
    echo $cmd
    $cmd
done