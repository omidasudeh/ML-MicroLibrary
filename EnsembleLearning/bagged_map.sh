#!/bin/bash
for ((job=1; job<$((51)); job+=1)); do
    cmd="sbatch ./begged.sh $job"
    echo $cmd
    $cmd
done