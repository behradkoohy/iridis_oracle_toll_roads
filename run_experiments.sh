#!/bin/bash

dbpath="big_results_1000cars_1000ts.db"

#source venv/bin/activate
module load conda

source activate benchmarks

#python CreateDatabase.py $dbpath

#model_name="PSOTimeOnly"
#description="none"
#vot_seed=0
#for i in {1..100}
#do
#  sbatch experiment_deamon.slurm $dbpath $i $model_name $description $vot_seed $i
#done
models=(
    "PSOTimeOnly"
    "PSOFixedPriceTTOpt"
    "PSOFixedPriceSCOpt"
    "PSOFixedPriceCCOpt"
    "PSOFTimestepPriceTTOpt"
    "PSOFTimestepPriceSCOpt"
    "PSOFTimestepPriceCCOpt"
#    "PSOFUnboundPriceTTOpt"
#    "PSOFUnboundPriceSCOpt"
#    "PSOFUnboundPriceCCOpt"
#    "PSOLinearPriceTTOpt"
#    "PSOLinearPriceSCOpt"
#    "PSOLinearPriceCCOpt"
)

description="none"
vot_seed=0
id_counter=0

for model_name in "${models[@]}"
do
  for ((i=0; i<50; i++))
  do
    job_id=$(sbatch experiment_deamon.slurm $dbpath $id_counter $model_name $description $i $i)
    #id_counter=$((id_counter + 1))
    echo "scancel $job_id" >> cancel_running_jobs.sh
    id_counter=$((id_counter + 1))
  done
done
