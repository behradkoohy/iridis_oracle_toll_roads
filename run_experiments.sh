#!/bin/bash

dbpath="results.db"

source venv/bin/activate

python CreateDatabase.py $dbpath

model_name="PSOTimeOnly"
description="Calculate-the-optimal-solution-using-PSO-Time-Only"
vot_seed=0

for i in {10..100}
do
  sbatch experiment_deamon.slurm $dbpath $i $model_name $description $vot_seed $i
done
