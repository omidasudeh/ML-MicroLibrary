#!/bin/bash
for ((job=1; job<$((51)); job+=1)); do
    cmd="sbatch ./adaboost.sh $job"
    echo $cmd
    $cmd
done